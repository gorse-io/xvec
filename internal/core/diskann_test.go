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
	"errors"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiskANNBuildSearchMetricsFilterRadiusAndRefiner(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		t.Run(diskANNMetricName(metric), func(t *testing.T) {
			candidates := diskANNIndexCandidates(72, 8)
			options := DefaultDiskANNBuildOptions(metric)
			options.MaxDegree, options.ListSize, options.PQChunks = 8, 24, 4
			options.CacheCapacity = len(candidates)
			index := buildDiskANNIndex(t, candidates, options)
			require.True(t, index.Dimension() == 8)
			require.Equal(t, metric, index.Metric())
			require.Len(t, candidates, index.Len())
			require.True(t, index.PQChunks() == 4)
			{
				got, ok := index.EntryPoint()
				require.True(t, ok)
				require.True(t, slices.ContainsFunc(candidates, func(candidate Candidate) bool { return candidate.Key == got }))
			}

			query := slices.Clone(candidates[17].Vector)
			linearOptions := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 12}, ListSize: len(candidates), Linear: true}
			want, err := index.SearchDiskANN(context.Background(), query, linearOptions)
			require.NoError(t, err)

			graphOptions := linearOptions
			graphOptions.Linear = false
			got, err := index.SearchDiskANN(context.Background(), query, graphOptions)
			require.NoError(t, err)
			require.Equal(t, want, got)
			require.Equal(t, candidates[17].Key, got[0].Key)

			filter := func(key uint64) bool { return key%2 == 0 }
			filteredOptions := graphOptions
			filteredOptions.Filter = filter
			filtered, err := index.SearchDiskANN(context.Background(), query, filteredOptions)
			require.NoError(t, err)

			for _, result := range filtered {
				require.True(t, filter(result.Key))
			}
			require.False(t, len(filtered) == 0,
				"filter removed every result")

			radiusOptions := graphOptions
			radiusOptions.Radius = want[min(5, len(want)-1)].Score
			if metric == MetricIP && radiusOptions.Radius <= 0 {
				radiusOptions.Radius = 0.0001
			}
			radius, err := index.SearchDiskANN(context.Background(), query, radiusOptions)
			require.NoError(t, err)

			for _, result := range radius {
				require.True(t, scoreWithinRadius(metric, result.Score, radiusOptions.Radius))
			}

			refiner, err := NewOriginalVectorRefiner(index, metric)
			require.NoError(t, err)

			refined, err := refiner.Refine(context.Background(), query, got, SearchOptions{TopK: 5})
			require.NoError(t, err)
			require.Equal(t, want[:5], refined)

			vector, found := index.Vector(candidates[9].Key)
			require.True(t, found,
				"original vector provider differs")
			require.True(t, slices.Equal(vector, candidates[9].Vector),
				"original vector provider differs")

			vector[0]++
			again, _ := index.Vector(candidates[9].Key)
			require.False(t, slices.Equal(vector, again),
				"vector provider returned an alias")
		})
	}
}

func TestDiskANNBuilderEmptyValidationCancellationAndClose(t *testing.T) {
	defaults := DefaultDiskANNBuildOptions(MetricL2)
	require.True(t, defaults.MaxDegree == 100)
	require.True(t, defaults.ListSize == 50)
	require.True(t, defaults.PQChunks == 0)
	require.True(t, defaults.CacheCapacity == 1024)

	invalid := []DiskANNBuildOptions{
		{},
		func() DiskANNBuildOptions { value := defaults; value.MaxDegree = 0; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.ListSize = 0; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.PQChunks = -1; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.Workers = -1; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.CacheCapacity = -1; return value }(),
	}
	for _, options := range invalid {
		{
			_, err := NewDiskANNBuilder(8, options)
			require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
		}
	}
	{
		_, err := NewDiskANNBuilder(0, defaults)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}

	tooManyChunks := defaults
	tooManyChunks.PQChunks = 9
	{
		_, err := NewDiskANNBuilder(8, tooManyChunks)
		require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
	}

	builder, err := NewDiskANNBuilder(4, defaults)
	require.NoError(t, err)
	{
		err := builder.Add(context.Background(), 7, []float32{1, 2, 3, 4})
		require.NoError(t, err)
	}
	{
		err := builder.Add(context.Background(), 7, []float32{4, 3, 2, 1})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		err := builder.Add(context.Background(), 8, []float32{1})
		require.Error(t, err,
			"invalid vector succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := builder.Add(canceled, 8, []float32{1, 2, 3, 4})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := builder.Build(canceled)
		require.ErrorIs(t, err, context.Canceled)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		err := builder.Add(context.Background(), 8, []float32{1, 2, 3, 4})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, err := index.SearchDiskANN(context.Background(), []float32{1, 2, 3, 4}, DiskANNSearchOptions{})
		require.ErrorIs(t, err, ErrInvalidDiskANNListSize)
	}
	{
		_, err := index.SearchWithOptions(context.Background(), []float32{1, 2, 3, 4}, SearchOptions{})
		require.ErrorIs(t, err, ErrInvalidTopK)
	}
	{
		_, err := index.SearchDiskANN(canceled, []float32{1, 2, 3, 4}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Close()
		require.NoError(t, err)
	}
	{
		err := index.Close()
		require.NoError(t, err)
	}
	{
		_, err := index.SearchDiskANN(context.Background(), []float32{1, 2, 3, 4}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1})
		require.ErrorIs(t, err, ErrDiskANNClosed)
	}
	{
		_, found := index.Vector(7)
		require.False(t, found,
			"closed provider returned a vector")
	}

	emptyBuilder, err := NewDiskANNBuilder(4, defaults)
	require.NoError(t, err)

	empty, err := emptyBuilder.Build(context.Background())
	require.NoError(t, err)

	results, err := empty.SearchDiskANN(context.Background(), []float32{0, 0, 0, 0}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1})
	require.NoError(t, err)
	require.Len(t, results, 0)
	require.True(t, empty.PQChunks() == 0)
}

func TestDiskANNWarmCacheBoundAndSearchOwnership(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 6, 16, 3, 7
	index := buildDiskANNIndex(t, diskANNIndexCandidates(40, 6), options)
	warmed, err := index.WarmCache(context.Background(), 20)
	require.NoError(t, err)
	require.True(t, warmed == 7)
	require.True(t, index.nodes.cache.Len() == 7)
	{
		_, err := index.WarmCache(context.Background(), -1)
		require.ErrorIs(t, err, ErrInvalidDiskANNOptions)
	}

	before := index.CacheStats()
	query := diskANNIndexCandidates(1, 6)[0].Vector
	first, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 5}, ListSize: 30})
	require.NoError(t, err)

	second, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 5}, ListSize: 30})
	require.NoError(t, err)
	require.Equal(t, second, first)
	require.True(t, index.CacheStats().Hits > before.Hits)
}

func TestDiskANNApproximateRecall(t *testing.T) {
	const count, dimension, queryCount, topK = 320, 16, 8, 10
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 12, 48, 8, 64
	candidates := diskANNIndexCandidates(count, dimension)
	index := buildDiskANNIndex(t, candidates, options)
	hits := 0
	for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
		query := candidates[(queryIndex*37+11)%count].Vector
		want, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
			SearchOptions: SearchOptions{TopK: topK}, ListSize: count, Linear: true,
		})
		require.NoError(t, err)

		got, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
			SearchOptions: SearchOptions{TopK: topK}, ListSize: 64,
		})
		require.NoError(t, err)

		truth := make(map[uint64]struct{}, len(want))
		for _, result := range want {
			truth[result.Key] = struct{}{}
		}
		for _, result := range got {
			if _, found := truth[result.Key]; found {
				hits++
			}
		}
	}
	recall := float64(hits) / (queryCount * topK)
	require.True(t, recall >= 0.80)
}

func TestDiskANNConcurrentSearchAndProviderReads(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricCosine)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 32
	candidates := diskANNIndexCandidates(96, 8)
	index := buildDiskANNIndex(t, candidates, options)
	query := candidates[31].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: 64}
	want, err := index.SearchDiskANN(context.Background(), query, search)
	require.NoError(t, err)

	var wait sync.WaitGroup
	errorsCh := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func(worker int) {
			defer wait.Done()
			for iteration := 0; iteration < 20; iteration++ {
				got, err := index.SearchDiskANN(context.Background(), query, search)
				if err != nil {
					errorsCh <- err
					return
				}
				if !assert.Equal(t, want, got) {
					errorsCh <- errors.New("concurrent DiskANN result differs")
					return
				}
				candidate := candidates[(worker+iteration)%len(candidates)]
				if vector, found := index.Vector(candidate.Key); !found || !slices.Equal(vector, candidate.Vector) {
					errorsCh <- errors.New("concurrent DiskANN provider read differs")
					return
				}
			}
		}(worker)
	}
	wait.Wait()
	close(errorsCh)
	for err := range errorsCh {
		require.NoError(t, err)
	}
}

func buildDiskANNIndex(t testing.TB, candidates []Candidate, options DiskANNBuildOptions) *DiskANNIndex {
	t.Helper()
	dimension := 8
	if len(candidates) != 0 {
		dimension = len(candidates[0].Vector)
	}
	builder, err := NewDiskANNBuilder(dimension, options)
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

func diskANNIndexCandidates(count, dimension int) []Candidate {
	result := make([]Candidate, count)
	for position := range result {
		vector := make([]float32, dimension)
		for component := range vector {
			angle := float64((position+3)*(component+5)) * 0.071
			vector[component] = float32(math.Sin(angle) + 0.45*math.Cos(angle*0.37) + float64(position%9)*0.025)
		}
		result[position] = Candidate{Key: uint64(1000 + position*7), Vector: vector}
	}
	return result
}

func diskANNMetricName(metric Metric) string {
	switch metric {
	case MetricL2:
		return "L2"
	case MetricIP:
		return "IP"
	case MetricCosine:
		return "Cosine"
	case MetricMIPSL2:
		return "MIPSL2"
	default:
		return "invalid"
	}
}
