// Copyright 2026-present the xvec project
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
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIVFBuildPartitionsVectors(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 2
	options.NIterations = 20
	options.Seed = 7
	builder, err := NewIVFBuilder(2, options)
	require.NoError(t, err)

	input := []Candidate{
		{Key: 10, Vector: []float32{0, 0}},
		{Key: 11, Vector: []float32{0, 1}},
		{Key: 20, Vector: []float32{10, 10}},
		{Key: 21, Vector: []float32{10, 11}},
	}
	for _, candidate := range input {
		{
			err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Dimension() == 2)
	require.Equal(t, MetricL2, index.Metric())
	require.True(t, index.Len() == 4)
	require.True(t, index.NList() == 2)

	centroids := index.Centroids()
	require.Len(t, centroids, 2)

	firstList, ok := index.ListForKey(10)
	require.True(t, ok,
		"key 10 is not assigned")
	{
		list, _ := index.ListForKey(11)
		require.Equal(t, firstList, list)
	}

	secondList, ok := index.ListForKey(20)
	require.True(t, ok)
	require.NotEqual(t, firstList, secondList)
	{
		list, _ := index.ListForKey(21)
		require.Equal(t, secondList, list)
	}

	for _, list := range []int{firstList, secondList} {
		candidates, err := index.List(list)
		require.NoError(t, err)
		require.Len(t, candidates, 2)
		require.True(t, candidates[0].Key <= candidates[1].Key)
	}
	require.False(t, index.TrainingIterations() == 0)
	require.False(t, math.IsNaN(index.TrainingCost()))
	require.False(t, math.IsInf(index.TrainingCost(), 0))
}

func TestIVFBuildOwnsOriginalVectors(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 1
	builder, _ := NewIVFBuilder(2, options)
	input := []float32{1, 2}
	{
		err := builder.Add(context.Background(), 7, input)
		require.NoError(t, err)
	}

	input[0] = 99
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	vector, found := index.Vector(7)
	require.True(t, found)
	require.True(t, slices.Equal(vector, []float32{1, 2}))

	vector[0] = 88
	candidates, _ := index.List(0)
	candidates[0].Vector[0] = 77
	centroids := index.Centroids()
	centroids[0][0] = 66
	vector, _ = index.Vector(7)
	require.True(t, vector[0] == 1,
		"IVF accessors expose mutable index state")
	require.True(t, index.Centroids()[0][0] == 1,
		"IVF accessors expose mutable index state")
}

func TestIVFBuildDeterministicAcrossWorkers(t *testing.T) {
	t.Parallel()
	build := func(workers int) *IVFIndex {
		options := DefaultIVFBuildOptions(MetricL2)
		options.NList = 7
		options.NIterations = 8
		options.Seed = 123
		options.Workers = workers
		builder, err := NewIVFBuilder(3, options)
		require.NoError(t, err)

		for index := 0; index < 100; index++ {
			vector := []float32{float32(index % 11), float32(index%7) / 2, float32(index%5) - 2}
			{
				err := builder.Add(context.Background(), uint64(index+1), vector)
				require.NoError(t, err)
			}
		}
		built, err := builder.Build(context.Background())
		require.NoError(t, err)

		return built
	}
	one, many := build(1), build(8)
	require.Equal(t, many.Centroids(), one.Centroids(),
		"IVF training differs across worker counts")
	require.Equal(t, many.TrainingCost(), one.TrainingCost(),
		"IVF training differs across worker counts")

	for key := uint64(1); key <= 100; key++ {
		left, _ := one.ListForKey(key)
		right, _ := many.ListForKey(key)
		require.Equal(t, right, left)
	}
}

func TestIVFBuildEmptyAndClusterCap(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	builder, _ := NewIVFBuilder(3, options)
	empty, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, empty.Len() == 0)
	require.True(t, empty.NList() == 0)
	require.Len(t, empty.Centroids(), 0)
	{
		_, err := empty.List(0)
		require.ErrorIs(t, err, ErrInvalidIVFList)
	}

	options.NList = 100
	builder, _ = NewIVFBuilder(1, options)
	for key := uint64(1); key <= 3; key++ {
		{
			err := builder.Add(context.Background(), key, []float32{float32(key)})
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.NList() == 3)
}

func TestIVFBuilderLifecycleAndValidation(t *testing.T) {
	t.Parallel()
	valid := DefaultIVFBuildOptions(MetricL2)
	for _, options := range []IVFBuildOptions{
		{},
		func() IVFBuildOptions { value := valid; value.Metric = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.NList = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.NIterations = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.Tolerance = -1; return value }(),
		func() IVFBuildOptions { value := valid; value.Tolerance = math.NaN(); return value }(),
	} {
		{
			_, err := NewIVFBuilder(2, options)
			assert.ErrorIs(t, err, ErrInvalidIVFOptions)
		}
	}
	{
		_, err := NewIVFBuilder(0, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}

	builder, _ := NewIVFBuilder(2, valid)
	{
		err := builder.Add(nil, 1, []float32{1, 2})
		require.Error(t, err,
			"nil add context accepted")
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1})
		require.ErrorIs(t, err, mathutil.ErrDimensionMismatch)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1, float32(math.Inf(1))})
		require.ErrorIs(t, err, mathutil.ErrNonFiniteVector)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1, 2})
		require.NoError(t, err)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{3, 4})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Len() == 1)
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		err := builder.Add(context.Background(), 2, []float32{3, 4})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, found := index.Vector(99)
		require.False(t, found,
			"missing vector found")
	}
	{
		_, found := index.ListForKey(99)
		require.False(t, found,
			"missing list assignment found")
	}
	{
		_, err := index.List(-1)
		require.ErrorIs(t, err, ErrInvalidIVFList)
	}
}

func TestIVFBuilderCancellationCanRetry(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 2
	builder, _ := NewIVFBuilder(1, options)
	for key := uint64(1); key <= 4; key++ {
		{
			err := builder.Add(context.Background(), key, []float32{float32(key)})
			require.NoError(t, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := builder.Build(ctx)
		require.ErrorIs(t, err, context.Canceled)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Len() == 4)
}

func TestIVFBuildOptionsAreValueSemantic(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricIP)
	options.NList = 2
	builder, _ := NewIVFBuilder(1, options)
	for key, value := range map[uint64]float32{1: -2, 2: -1, 3: 1, 4: 2} {
		{
			err := builder.Add(context.Background(), key, []float32{value})
			require.NoError(t, err)
		}
	}
	options.NList = 99
	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.BuildOptions().NList == 2)
	require.Equal(t, MetricIP, index.Metric())
}

func TestIVFSearchProbesNearestLists(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{
		{Key: 1, Vector: []float32{0, 0}},
		{Key: 2, Vector: []float32{0, 1}},
		{Key: 3, Vector: []float32{10, 10}},
		{Key: 4, Vector: []float32{10, 11}},
	}, 2)
	lists, err := index.ProbedLists(context.Background(), []float32{0, 0}, 1)
	require.NoError(t, err)
	require.Len(t, lists, 1)

	nearList, _ := index.ListForKey(1)
	require.Equal(t, nearList, lists[0])

	results, err := index.SearchIVF(context.Background(), []float32{0, 0}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 4}, NProbe: 1,
	})
	require.NoError(t, err)
	require.True(t, slices.Equal(results, []Result{{Key: 1, Score: 0}, {Key: 2, Score: 1}}))

	allLists, err := index.ProbedLists(context.Background(), []float32{0, 0}, 99)
	require.NoError(t, err)
	require.Len(t, allLists, index.NList())
}

func TestIVFFullProbeMatchesFlat(t *testing.T) {
	t.Parallel()
	candidates := []Candidate{
		{Key: 9, Vector: []float32{-2, 1, 0}},
		{Key: 2, Vector: []float32{1, 0, 2}},
		{Key: 7, Vector: []float32{0, 3, -1}},
		{Key: 4, Vector: []float32{2, 2, 2}},
		{Key: 1, Vector: []float32{-1, -1, 1}},
		{Key: 8, Vector: []float32{4, 0, -2}},
	}
	query := []float32{1, 2, .5}
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		index := buildSearchIVF(t, metric, candidates, 3)
		flat, err := TopK(context.Background(), metric, query, candidates, len(candidates))
		require.NoError(t, err)

		got, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
			SearchOptions: SearchOptions{TopK: len(candidates)}, NProbe: index.NList(),
		})
		require.NoError(t, err)
		require.True(t, slices.Equal(got, flat))
	}
}

func TestIVFSearchFilterRadiusAndTies(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{
		{Key: 5, Vector: []float32{-1}},
		{Key: 2, Vector: []float32{1}},
		{Key: 9, Vector: []float32{2}},
	}, 2)
	results, err := index.SearchIVF(context.Background(), []float32{0}, IVFSearchOptions{
		SearchOptions: SearchOptions{
			TopK: 3, Radius: 1,
			Filter: func(key uint64) bool { return key != 5 },
		},
		NProbe: 2,
	})
	require.NoError(t, err)
	require.True(t, slices.Equal(results, []Result{{Key: 2, Score: 1}}))

	results, err = index.SearchIVF(context.Background(), []float32{0}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 3}, NProbe: 2,
	})
	require.NoError(t, err)
	require.True(t, len(results) >= 2)
	require.True(t, results[0].Key == 2)
	require.True(t, results[1].Key == 5)
}

func TestIVFSearchInnerProductRadius(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricIP, []Candidate{
		{Key: 1, Vector: []float32{-2}},
		{Key: 2, Vector: []float32{1}},
		{Key: 3, Vector: []float32{3}},
	}, 2)
	results, err := index.SearchIVF(context.Background(), []float32{1}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 3, Radius: 2}, NProbe: index.NList(),
	})
	require.NoError(t, err)
	require.True(t, slices.Equal(results, []Result{{Key: 3, Score: 3}}))
}

func TestIVFSearchDefaultAndEmpty(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{{Key: 1, Vector: []float32{1}}}, 1)
	results, err := index.Search(context.Background(), []float32{0}, 1)
	require.NoError(t, err)
	require.True(t, slices.Equal(results, []Result{{Key: 1, Score: 1}}))

	results, err = index.Search(context.Background(), []float32{0}, 0)
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Len(t, results, 0)

	options := DefaultIVFBuildOptions(MetricL2)
	builder, _ := NewIVFBuilder(1, options)
	empty, err := builder.Build(context.Background())
	require.NoError(t, err)

	results, err = empty.SearchIVF(context.Background(), []float32{0}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, NProbe: 1,
	})
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Len(t, results, 0)

	lists, err := empty.ProbedLists(context.Background(), []float32{0}, 1)
	require.NoError(t, err)
	require.NotNil(t, lists)
	require.Len(t, lists, 0)
}

func TestIVFSearchValidation(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{{Key: 1, Vector: []float32{1, 2}}}, 1)
	valid := IVFSearchOptions{SearchOptions: SearchOptions{TopK: 1}, NProbe: 1}
	{
		_, err := index.SearchIVF(nil, []float32{1, 2}, valid)
		require.Error(t, err,
			"nil context accepted")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := index.SearchIVF(ctx, []float32{1, 2}, valid)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := index.SearchIVF(context.Background(), []float32{1}, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
	{
		_, err := index.SearchIVF(context.Background(), []float32{1, float32(math.NaN())}, valid)
		require.ErrorIs(t, err, mathutil.ErrNonFiniteVector)
	}

	invalidProbe := valid
	invalidProbe.NProbe = 0
	{
		_, err := index.SearchIVF(context.Background(), []float32{1, 2}, invalidProbe)
		require.ErrorIs(t, err, ErrInvalidIVFNProbe)
	}

	invalidTopK := valid
	invalidTopK.TopK = 0
	{
		_, err := index.SearchIVF(context.Background(), []float32{1, 2}, invalidTopK)
		require.ErrorIs(t, err, ErrInvalidTopK)
	}
	{
		_, err := index.Search(context.Background(), []float32{1, 2}, -1)
		require.Error(t, err,
			"negative top-k accepted")
	}
	{
		_, err := index.ProbedLists(context.Background(), []float32{1, 2}, 0)
		require.ErrorIs(t, err, ErrInvalidIVFNProbe)
	}

	var nilIndex *IVFIndex
	{
		_, err := nilIndex.Search(context.Background(), []float32{1}, 1)
		require.Error(t, err,
			"nil index accepted")
	}
}

func buildSearchIVF(t *testing.T, metric Metric, candidates []Candidate, nlist int) *IVFIndex {
	t.Helper()
	options := DefaultIVFBuildOptions(metric)
	options.NList = nlist
	options.NIterations = 20
	options.Seed = 17
	builder, err := NewIVFBuilder(len(candidates[0].Vector), options)
	require.NoError(t, err)

	for _, candidate := range candidates {
		{
			err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

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
		require.ErrorIs(t, err, mathutil.ErrDimensionMismatch)
	}
	{
		err := index.Add(context.Background(), 1000, []float32{1, float32(math.NaN()), 3})
		require.ErrorIs(t, err, mathutil.ErrNonFiniteVector)
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

func TestIVFPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricCosine, 3)
	path := filepath.Join(t.TempDir(), "vectors.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())

	opened, err := OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameIVFIndex(t, opened, index)
	query := []float32{0.25, 0.5, 0.75}
	want, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: index.Len()},
		NProbe:        index.NList(),
	})
	require.NoError(t, err)

	got, err := opened.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: opened.Len()},
		NProbe:        opened.NList(),
	})
	require.NoError(t, err)
	require.Equal(t, want, got)

	replacement := persistedIVFIndex(t, MetricIP, 2)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err = OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameIVFIndex(t, opened, replacement)
}

func TestIVFPersistenceEmpty(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricMIPSL2)
	options.Seed = 17
	builder, err := NewIVFBuilder(7, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "empty.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameIVFIndex(t, opened, index)
}

func TestIVFPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricL2, 2)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	original, err := os.ReadFile(path)
	require.NoError(t, err)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Save(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"canceled replacement changed published IVF file")
	{
		_, err := OpenIVFIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenIVFIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
	{
		_, err := OpenIVFIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
	{
		_, err := OpenIVFIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	invalid := &IVFIndex{dimension: 3, options: DefaultIVFBuildOptions(MetricL2), keys: []uint64{1}}
	{
		err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.ivf"))
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
}

func TestIVFPersistenceDetectsTruncationAndCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeIVFIndex(context.Background(), persistedIVFIndex(t, MetricL2, 2))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, ivfHeaderSize - 1, ivfHeaderSize, len(valid) - 1} {
		{
			_, err := decodeIVFIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidIVFFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeIVFIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeIVFIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], ivfFileVersion+1)
	{
		_, err := decodeIVFIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedIVFVersion)
	}

	badHeaderCRC := slices.Clone(valid)
	badHeaderCRC[55] ^= 1
	{
		_, err := decodeIVFIndex(context.Background(), badHeaderCRC)
		require.ErrorIs(t, err, ErrIVFChecksumMismatch)
	}

	badPayloadCRC := slices.Clone(valid)
	badPayloadCRC[len(badPayloadCRC)-1] ^= 1
	{
		_, err := decodeIVFIndex(context.Background(), badPayloadCRC)
		require.ErrorIs(t, err, ErrIVFChecksumMismatch)
	}
}

func TestIVFPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeIVFIndex(context.Background(), persistedIVFIndex(t, MetricL2, 2))
	require.NoError(t, err)

	dimension := int(binary.LittleEndian.Uint32(valid[40:44]))
	nlist := int(binary.LittleEndian.Uint32(valid[44:48]))
	firstRecord := ivfHeaderSize + nlist*dimension*4
	recordSize := ivfRecordOverhead + dimension*4

	tests := []struct {
		name   string
		mutate func([]byte)
	}{
		{
			name: "duplicate key",
			mutate: func(data []byte) {
				copy(data[firstRecord+recordSize:firstRecord+recordSize+8], data[firstRecord:firstRecord+8])
			},
		},
		{
			name: "list out of range",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[firstRecord+8+dimension*4:firstRecord+recordSize], uint32(nlist))
			},
		},
		{
			name: "non-finite centroid",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[ivfHeaderSize:ivfHeaderSize+4], math.Float32bits(float32(math.NaN())))
			},
		},
		{
			name: "non-finite vector",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[firstRecord+8:firstRecord+12], math.Float32bits(float32(math.Inf(1))))
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			rechecksumIVF(encoded)
			{
				_, err := decodeIVFIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidIVFFile)
			}
		})
	}

	badOptions := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badOptions[52:56], 0)
	binary.LittleEndian.PutUint32(badOptions[108:112], hashutil.CRC32C(badOptions[:108]))
	{
		_, err := decodeIVFIndex(context.Background(), badOptions)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}

	badLength := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badLength[16:24], uint64(len(badLength)+1))
	binary.LittleEndian.PutUint32(badLength[108:112], hashutil.CRC32C(badLength[:108]))
	{
		_, err := decodeIVFIndex(context.Background(), badLength)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
}

func FuzzDecodeIVFIndex(f *testing.F) {
	index := persistedIVFIndex(f, MetricL2, 2)
	valid, err := encodeIVFIndex(context.Background(), index)
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("ZVECIVF"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeIVFIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateIVFIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func BenchmarkIVFBuild(b *testing.B) {
	const (
		count     = 4096
		dimension = 64
	)
	vectors := make([][]float32, count)
	for position := range vectors {
		vectors[position] = make([]float32, dimension)
		for coordinate := range vectors[position] {
			vectors[position][coordinate] = float32((position*17+coordinate*31)%997) / 997
		}
	}
	options := DefaultIVFBuildOptions(MetricCosine)
	options.NList = 64
	options.NIterations = 5
	options.Workers = 8

	b.ResetTimer()
	for range b.N {
		builder, err := NewIVFBuilder(dimension, options)
		if err != nil {
			b.Fatal(err)
		}
		for position, vector := range vectors {
			if err := builder.Add(context.Background(), uint64(position), vector); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := builder.Build(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIVFSearch(b *testing.B) {
	const (
		count     = 8192
		dimension = 128
	)
	options := DefaultIVFBuildOptions(MetricCosine)
	options.NList = 256
	options.NIterations = 2
	options.Workers = 8
	builder, err := NewIVFBuilder(dimension, options)
	if err != nil {
		b.Fatal(err)
	}
	for position := range count {
		vector := make([]float32, dimension)
		for coordinate := range vector {
			vector[coordinate] = float32((position*17+coordinate*31)%997) / 997
		}
		if err := builder.Add(context.Background(), uint64(position), vector); err != nil {
			b.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		b.Fatal(err)
	}
	query, _ := index.Vector(17)
	search := IVFSearchOptions{SearchOptions: SearchOptions{TopK: 100}, NProbe: 10}

	b.ResetTimer()
	for range b.N {
		if _, err := index.SearchIVF(context.Background(), query, search); err != nil {
			b.Fatal(err)
		}
	}
}

func persistedIVFIndex(t testing.TB, metric Metric, nlist int) *IVFIndex {
	t.Helper()
	options := DefaultIVFBuildOptions(metric)
	options.NList = nlist
	options.NIterations = 7
	options.Tolerance = 1e-8
	options.Workers = 2
	options.Seed = 0x123456789abcdef0
	builder, err := NewIVFBuilder(3, options)
	require.NoError(t, err)

	for _, candidate := range []Candidate{
		{Key: 41, Vector: []float32{1, 0, 0}},
		{Key: 7, Vector: []float32{0, 1, 0}},
		{Key: 99, Vector: []float32{0, 0, 1}},
		{Key: 5, Vector: []float32{1, 1, 0}},
		{Key: 123, Vector: []float32{0.5, 0.25, 0.75}},
	} {
		{
			err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

func assertSameIVFIndex(t testing.TB, got, want *IVFIndex) {
	t.Helper()
	require.Equal(t, want.Dimension(), got.Dimension())
	require.Equal(t, want.Metric(), got.Metric())
	require.Equal(t, want.Len(), got.Len())
	require.Equal(t, want.NList(), got.NList())
	require.Equal(t, want.BuildOptions(), got.BuildOptions())
	require.Equal(t, want.TrainingCost(), got.TrainingCost())
	require.Equal(t, want.TrainingIterations(), got.TrainingIterations())
	require.Equal(t, want.TrainingConverged(), got.TrainingConverged())
	require.Equal(t, want.Centroids(), got.Centroids())
	require.True(t, slices.Equal(got.keys, want.keys))

	for _, key := range want.keys {
		gotVector, gotOK := got.Vector(key)
		wantVector, wantOK := want.Vector(key)
		gotList, gotListOK := got.ListForKey(key)
		wantList, wantListOK := want.ListForKey(key)
		require.Equal(t, wantOK, gotOK)
		require.True(t, slices.Equal(gotVector, wantVector))
		require.Equal(t, wantListOK, gotListOK)
		require.Equal(t, wantList, gotList)
	}
}

func rechecksumIVF(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[96:100], hashutil.CRC32C(encoded[ivfHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], hashutil.CRC32C(encoded[:108]))
}

// TestIVFPartialProbeRecall locks the quality boundary that is not covered
// by IVF's full-probe exactness tests. The corpus, training seed, and query
// selection are deterministic so a graph/training change cannot silently buy
// speed by crossing the configured recall floor.
func TestIVFPartialProbeRecall(t *testing.T) {
	candidates := quantizedIndexCandidates(2048)
	index := buildDenseIVFFromCandidates(t, candidates, 32)

	var matched, total int
	for queryIndex := range 24 {
		query := candidates[(queryIndex*79+31)%len(candidates)].Vector
		got, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
			SearchOptions: SearchOptions{TopK: 10},
			NProbe:        8,
		})
		require.NoError(t, err)

		want, err := TopK(context.Background(), MetricL2, query, candidates, 10)
		require.NoError(t, err)

		matched += resultOverlap(got, want)
		total += len(want)
	}
	{
		recall := float64(matched) / float64(total)
		require.True(t, recall >= .90)
	}
}

// BenchmarkIVFSearchQualityComparison records latency, allocation, and recall
// on a common corpus. It complements the algorithm-specific build/search
// benchmarks and makes performance comparisons meaningful only at their
// reported quality point.
func BenchmarkIVFSearchQualityComparison(b *testing.B) {
	candidates := quantizedIndexCandidates(10_000)
	query := candidates[4321].Vector
	truth, err := TopK(context.Background(), MetricL2, query, candidates, 10)
	if err != nil {
		require.NoError(b, err)
	}

	flat, err := NewDenseFlatIndex(4, MetricL2)
	if err != nil {
		require.NoError(b, err)
	}

	for _, candidate := range candidates {
		{
			err := flat.Add(context.Background(), candidate.Key, candidate.Vector)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
	ivf := buildDenseIVFFromCandidates(b, candidates, 64)
	hnsw := buildDenseHNSWFromCandidates(b, candidates)
	quantizedHNSW, err := NewScalarQuantizedHNSWIndex(
		context.Background(), hnsw, QuantizationInt8, nil,
	)
	if err != nil {
		require.NoError(b, err)
	}

	benchmarks := []struct {
		name   string
		search func() ([]Result, error)
	}{
		{
			name: "Flat",
			search: func() ([]Result, error) {
				return flat.Search(context.Background(), query, 10)
			},
		},
		{
			name: "IVF/NProbe=8",
			search: func() ([]Result, error) {
				return ivf.SearchIVF(context.Background(), query, IVFSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, NProbe: 8,
				})
			},
		},
		{
			name: "HNSW/EF=100",
			search: func() ([]Result, error) {
				return hnsw.SearchHNSW(context.Background(), query, HNSWSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, EF: 100,
				})
			},
		},
		{
			name: "HNSW-INT8/EF=100",
			search: func() ([]Result, error) {
				return quantizedHNSW.SearchHNSW(context.Background(), query, HNSWSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, EF: 100,
				})
			},
		},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			got, err := benchmark.search()
			if err != nil {
				require.NoError(b, err)
			}

			recall := float64(resultOverlap(got, truth)) / float64(len(truth))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				{
					_, err := benchmark.search()
					if err != nil {
						require.NoError(b, err)
					}
				}
			}
			b.ReportMetric(recall, "recall@10")
		})
	}
}
