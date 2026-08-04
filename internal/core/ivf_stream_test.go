// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package core

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestIVFIncrementalBootstrapAndAssignment(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricIP)
	options.NList = 3
	builder, err := NewIVFBuilder(2, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	inputs := []Candidate{
		{Key: 1, Vector: []float32{1, 0}},
		{Key: 2, Vector: []float32{0, 2}},
		{Key: 3, Vector: []float32{-1, 0}},
		{Key: 4, Vector: []float32{0, 1.5}},
	}
	for _, input := range inputs {
		{
			err := index.Add(context.Background(), input.Key, input.Vector)
			require.NoError(t, err)
		}
	}
	require.True(t, index.Len() == 4)
	require.True(t, index.NList() == 3)

	for key, want := range map[uint64]int{1: 0, 2: 1, 3: 2, 4: 1} {
		{
			got, found := index.ListForKey(key)
			require.True(t, found)
			require.Equal(t, want, got)
		}
	}
	{
		got := index.model.counts
		require.Equal(t, []int{1, 2, 1}, got)
	}
	{
		got := index.TrainingCost()
		require.Equal(t, float64(-9), got)
	}
	require.True(t, index.TrainingIterations() == 0,
		"bootstrap unexpectedly reports a completed training round")
	require.False(t, index.TrainingConverged(),
		"bootstrap unexpectedly reports a completed training round")

	query := []float32{0, 1}
	got, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: len(inputs)},
		NProbe:        index.NList(),
	})
	require.NoError(t, err)

	want, err := TopK(context.Background(), MetricIP, query, inputs, len(inputs))
	require.NoError(t, err)
	require.Equal(t, want, got)

	path := filepath.Join(t.TempDir(), "streamed.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameIVFIndex(t, reopened, index)
}

func TestIVFIncrementalFixedCentroidsAndOwnership(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricL2, 2)
	centroids := index.Centroids()
	vector := []float32{10, 11, 12}
	{
		err := index.Add(context.Background(), 1000, vector)
		require.NoError(t, err)
	}

	vector[0] = -100
	stored, found := index.Vector(1000)
	require.True(t, found)
	require.True(t, stored[0] == 10)
	require.Equal(t, centroids, index.Centroids(),
		"incremental add changed a full trained centroid set")

	list, found := index.ListForKey(1000)
	require.True(t, found,
		"incremental key has no list")

	candidates, err := index.List(list)
	require.NoError(t, err)
	require.True(t, candidates[len(candidates)-1].Key == 1000)
}

func TestIVFIncrementalValidationIsAtomic(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricL2, 2)
	originalLen := index.Len()
	{
		err := index.Add(nil, 1000, []float32{1, 2, 3})
		require.Error(t, err,
			"nil context succeeded")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Add(ctx, 1000, []float32{1, 2, 3})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Add(context.Background(), 1000, []float32{1, 2})
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		err := index.Add(context.Background(), 1000, []float32{1, float32(math.NaN()), 3})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}
	{
		err := index.Add(context.Background(), index.keys[0], []float32{1, 2, 3})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	require.Equal(t, originalLen, index.Len())

	var nilIndex *IVFIndex
	{
		err := nilIndex.Add(context.Background(), 1, []float32{1})
		require.Error(t, err,
			"nil index add succeeded")
	}
}

func TestIVFConcurrentStreamingSearchAndSave(t *testing.T) {
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 8
	options.NIterations = 4
	builder, err := NewIVFBuilder(3, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	dir := t.TempDir()
	errCh := make(chan error, 32)
	var writers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		writers.Add(1)
		go func(worker int) {
			defer writers.Done()
			for value := 0; value < 100; value++ {
				key := uint64(worker*100 + value + 1)
				vector := []float32{float32(worker), float32(value), float32(worker + value)}
				if err := index.Add(context.Background(), key, vector); err != nil {
					errCh <- err
					return
				}
				if _, found := index.Vector(key); !found {
					errCh <- fmt.Errorf("key %d missing immediately after add", key)
					return
				}
			}
		}(worker)
	}

	stopReaders := make(chan struct{})
	var readers sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stopReaders:
					return
				default:
				}
				nprobe := max(1, index.NList())
				_, err := index.SearchIVF(context.Background(), []float32{1, 2, 3}, IVFSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, NProbe: nprobe,
				})
				if err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for generation := 0; generation < 20; generation++ {
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.ivf", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenIVFIndex(context.Background(), path); err != nil {
				errCh <- err
				return
			}
		}
	}()

	writers.Wait()
	close(stopReaders)
	readers.Wait()
	close(errCh)
	for err := range errCh {
		require.NoError(t, err)
	}
	require.True(t, index.Len() == 800)

	path := filepath.Join(dir, "final.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)
	require.Equal(t, index.Len(), reopened.Len())
	require.Equal(t, index.NList(), reopened.NList())
}
