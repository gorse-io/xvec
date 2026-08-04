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
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestSparseHNSWIncrementalMatchesOneShotBuild(t *testing.T) {
	t.Parallel()
	inputs := sparseHNSWBuildInputs(180)
	options := DefaultSparseHNSWBuildOptions()
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x3141592653589793

	streamed := buildSparseHNSW(t, options, inputs[:120])
	for _, input := range inputs[120:] {
		{
			err := streamed.AddSparse(context.Background(), input.key, input.vector)
			require.NoError(t, err)
		}
	}
	oneShot := buildSparseHNSW(t, options, inputs)
	assertSameSparseHNSWIndex(t, streamed, oneShot)

	path := filepath.Join(t.TempDir(), "streamed.shnsw")
	{
		err := streamed.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, reopened, oneShot)

	next := sparseHNSWInput{key: 999999, vector: SparseVector{
		Indices: []uint32{7, 111, 307},
		Values:  []float32{3.25, 7.5, 1.125},
	}}
	{
		err := reopened.AddSparse(context.Background(), next.key, next.vector)
		require.NoError(t, err)
	}

	all := append(slices.Clone(inputs), next)
	want := buildSparseHNSW(t, options, all)
	assertSameSparseHNSWIndex(t, reopened, want)
}

func TestSparseHNSWIncrementalEmptyAndOwnership(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 4
	options.EFConstruction = 16
	options.Seed = 7
	builder, err := NewSparseHNSWBuilder(options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	vector := SparseVector{Indices: []uint32{1, 9}, Values: []float32{1, 2}}
	{
		err := index.AddSparse(context.Background(), 17, vector)
		require.NoError(t, err)
	}

	vector.Indices[0] = 8
	vector.Values[0] = -100
	stored, found := index.SparseVector(17)
	require.True(t, found)
	require.Equal(t, SparseVector{Indices: []uint32{1, 9}, Values: []float32{1, 2}}, stored)

	entry, found := index.EntryPoint()
	require.True(t, found)
	require.True(t, entry == 17)
	require.True(t, index.Len() == 1)

	results, err := index.SearchSparse(context.Background(), stored, 1)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 17, Score: 5}}, results)
}

func TestSparseHNSWIncrementalFailuresAreAtomic(t *testing.T) {
	t.Parallel()
	index := persistedSparseHNSWIndex(t, 300)
	before, err := encodeSparseHNSWIndex(context.Background(), index)
	require.NoError(t, err)

	valid := SparseVector{Indices: []uint32{1, 101, 201}, Values: []float32{1, 2, 3}}
	{
		err := index.AddSparse(nil, 100000, valid)
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.AddSparse(canceled, 100000, valid)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.AddSparse(context.Background(), 100000, SparseVector{Indices: []uint32{1}, Values: nil})
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		err := index.AddSparse(context.Background(), 100000, SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 2}})
		require.ErrorIs(t, err, ailego.ErrInvalidSparseOrder)
	}
	{
		err := index.AddSparse(context.Background(), 100000, SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.NaN())}})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}

	firstKey := sparseHNSWBuildInputs(1)[0].key
	{
		err := index.AddSparse(context.Background(), firstKey, valid)
		require.ErrorIs(t, err, ErrDuplicateKey)
	}

	midClone := newCancelAfterChecks(5)
	{
		err := index.AddSparse(midClone, 100000, valid)
		require.ErrorIs(t, err, context.Canceled)
	}

	midTraversal := newCancelAfterChecks(7)
	{
		err := index.AddSparse(midTraversal, 100000, valid)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := encodeSparseHNSWIndex(context.Background(), index)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, before),
		"failed incremental sparse add changed graph generation")

	var nilIndex *SparseHNSWIndex
	{
		err := nilIndex.AddSparse(context.Background(), 1, SparseVector{})
		require.Error(t, err,
			"nil index add succeeded")
	}

	next := sparseHNSWInput{key: 100000, vector: valid}
	{
		err := index.AddSparse(context.Background(), next.key, next.vector)
		require.NoError(t, err)
	}

	inputs := append(sparseHNSWBuildInputs(300), next)
	want := buildSparseHNSW(t, index.options, inputs)
	assertSameSparseHNSWIndex(t, index, want)
}

func TestSparseHNSWConcurrentStreamingSearchAndSave(t *testing.T) {
	options := DefaultSparseHNSWBuildOptions()
	options.M = 6
	options.EFConstruction = 30
	options.Seed = 123
	builder, err := NewSparseHNSWBuilder(options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	dir := t.TempDir()
	errCh := make(chan error, 32)
	var writers sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		writers.Add(1)
		go func(worker int) {
			defer writers.Done()
			for value := 0; value < 50; value++ {
				key := uint64(worker*100 + value + 1)
				vector := SparseVector{
					Indices: []uint32{uint32(worker + 1), uint32(100 + value), uint32(1000 + worker*100 + value)},
					Values:  []float32{float32(worker + 1), float32(value + 1), float32(worker + value + 1)},
				}
				if err := index.AddSparse(context.Background(), key, vector); err != nil {
					errCh <- err
					return
				}
				if _, found := index.SparseVector(key); !found {
					errCh <- fmt.Errorf("key %d missing immediately after sparse add", key)
					return
				}
			}
		}(worker)
	}

	stopReaders := make(chan struct{})
	var readers sync.WaitGroup
	query := SparseVector{Indices: []uint32{1, 100, 1000}, Values: []float32{1, 2, 3}}
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
				if _, err := index.SearchSparse(context.Background(), query, 10); err != nil {
					errCh <- err
					return
				}
				if entry, found := index.EntryPoint(); found {
					if _, found := index.Level(entry); !found {
						errCh <- fmt.Errorf("entry key %d has no level", entry)
						return
					}
				}
			}
		}()
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for generation := 0; generation < 10; generation++ {
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.shnsw", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenSparseHNSWIndex(context.Background(), path); err != nil {
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
	require.True(t, index.Len() == 200)

	path := filepath.Join(dir, "final.shnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, reopened, index)
}
