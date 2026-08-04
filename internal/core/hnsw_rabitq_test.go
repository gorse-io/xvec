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

func TestHNSWRaBitQBuildOptionsAndDeterminism(t *testing.T) {
	defaults := DefaultHNSWRaBitQBuildOptions(MetricCosine)
	require.Equal(t, MetricCosine, defaults.Metric)
	require.True(t, defaults.TotalBits == 7)
	require.True(t, defaults.Clusters == 16)
	require.True(t, defaults.M == 50)
	require.True(t, defaults.EFConstruction == 500)

	invalid := []HNSWRaBitQBuildOptions{
		{},
		func() HNSWRaBitQBuildOptions { value := defaults; value.Metric = MetricMIPSL2; return value }(),
		func() HNSWRaBitQBuildOptions { value := defaults; value.TotalBits = 10; return value }(),
		func() HNSWRaBitQBuildOptions { value := defaults; value.M = 0; return value }(),
		func() HNSWRaBitQBuildOptions { value := defaults; value.EFConstruction = 1; return value }(),
	}
	for _, options := range invalid {
		{
			_, err := NewHNSWRaBitQBuilder(64, options)
			require.ErrorIs(t, err, ErrInvalidHNSWRaBitQOptions)
		}
	}
	{
		_, err := NewHNSWRaBitQBuilder(63, defaults)
		require.ErrorIs(t, err, ErrInvalidHNSWRaBitQOptions)
	}

	candidates := hnswRaBitQCandidates(120, 70)
	options := hnswRaBitQTestOptions(MetricL2)
	options.Workers = 1
	first := buildHNSWRaBitQ(t, candidates, options)
	options.Workers = 4
	second := buildHNSWRaBitQ(t, candidates, options)
	require.Equal(t, second.ModelState(), first.ModelState(),

		"HNSW-RaBitQ changed across worker counts")
	require.Equal(t, second.base.levels, first.base.levels,

		"HNSW-RaBitQ changed across worker counts")
	require.Equal(t, second.base.neighbors, first.base.neighbors,

		"HNSW-RaBitQ changed across worker counts")
	require.Equal(t, second.codes, first.codes,
		"HNSW-RaBitQ changed across worker counts")
	require.True(t, first.Dimension() == 70,
		"HNSW-RaBitQ metadata differs")
	require.Equal(t, MetricL2, first.Metric(),
		"HNSW-RaBitQ metadata differs")
	require.Len(t, candidates, first.Len(),
		"HNSW-RaBitQ metadata differs")
	require.Equal(t, first.base.maxLevel, first.MaxLevel(),
		"HNSW-RaBitQ metadata differs")

	entry, found := first.EntryPoint()
	require.True(t, found,
		"built graph has no entry point")
	{
		level, _ := first.Level(entry)
		require.Equal(t, first.MaxLevel(), level)
	}

	vector, found := first.Vector(candidates[0].Key)
	require.True(t, found,
		"first vector missing")

	candidates[0].Vector[0]++
	{
		again, _ := first.Vector(candidates[0].Key)
		require.Equal(t, vector[0], again[0],
			"builder did not own originals")
	}

	vector[0]++
	{
		again, _ := first.Vector(candidates[0].Key)
		require.NotEqual(t, vector[0], again[0],
			"Vector exposed mutable storage")
	}
}

func TestHNSWRaBitQSearchMetricsFilterRadiusAndRefine(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		candidates := hnswRaBitQCandidates(180, 64)
		index := buildHNSWRaBitQ(t, candidates, hnswRaBitQTestOptions(metric))
		query := slices.Clone(candidates[37].Vector)
		options := HNSWRaBitQSearchOptions{
			SearchOptions: SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }},
			EF:            180,
		}
		got, err := index.SearchHNSWRaBitQ(context.Background(), query, options)
		require.NoError(t, err)
		require.Len(t, got, options.TopK)

		for position, result := range got {
			require.False(t, result.Key%3 == 0)
			require.False(t, position > 0 && resultBetter(metric, result, got[position-1]))
		}

		options.Refine = true
		refined, err := index.SearchHNSWRaBitQ(context.Background(), query, options)
		require.NoError(t, err)

		want, err := topKCandidatesWithOptions(context.Background(), metric, query, options.SearchOptions, len(candidates), func(position int) Candidate {
			return candidates[position]
		}, true)
		require.NoError(t, err)
		require.Equal(t, want, refined)
	}

	candidates := hnswRaBitQCandidates(80, 64)
	index := buildHNSWRaBitQ(t, candidates, hnswRaBitQTestOptions(MetricL2))
	target := candidates[17]
	results, err := index.SearchHNSWRaBitQ(context.Background(), target.Vector, HNSWRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: 5, Radius: .2, Filter: func(key uint64) bool { return key == target.Key }},
		EF:            80, Refine: true,
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: target.Key, Score: 0}}, results)
}

func TestHNSWRaBitQLargeGraphRecall(t *testing.T) {
	candidates := hnswRaBitQCandidates(DefaultHNSWBruteForceThreshold+120, 64)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		options := hnswRaBitQTestOptions(metric)
		options.M = 12
		options.EFConstruction = 80
		index := buildHNSWRaBitQ(t, candidates, options)
		var approximateMatches, refinedMatches, total int
		for queryIndex := 0; queryIndex < 12; queryIndex++ {
			query := candidates[(queryIndex*73+19)%len(candidates)].Vector
			truth, err := TopK(context.Background(), metric, query, candidates, 10)
			require.NoError(t, err)

			approximate, err := index.SearchHNSWRaBitQ(context.Background(), query, HNSWRaBitQSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, EF: 100,
			})
			require.NoError(t, err)

			refined, err := index.SearchHNSWRaBitQ(context.Background(), query, HNSWRaBitQSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, EF: 100, Refine: true,
			})
			require.NoError(t, err)

			approximateMatches += resultOverlap(approximate, truth)
			refinedMatches += resultOverlap(refined, truth)
			total += len(truth)
		}
		{
			recall := float64(approximateMatches) / float64(total)
			require.True(t, recall >= .75)
		}
		{
			recall := float64(refinedMatches) / float64(total)
			require.True(t, recall >= .85)
		}
	}
}

func TestHNSWRaBitQEmptyIncrementalAndValidation(t *testing.T) {
	options := hnswRaBitQTestOptions(MetricL2)
	builder, err := NewHNSWRaBitQBuilder(64, options)
	require.NoError(t, err)

	empty, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, empty.Len() == 0)
	require.Equal(t, -1, empty.MaxLevel())
	{
		results, err := empty.SearchHNSWRaBitQ(context.Background(), make([]float32, 64), HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 1})
		require.NoError(t, err)
		require.Len(t, results, 0)
	}

	vector := hnswRaBitQCandidates(1, 64)[0].Vector
	{
		err := empty.Add(context.Background(), 99, vector)
		require.NoError(t, err)
	}
	require.True(t, empty.Len() == 1)
	{
		err := empty.Add(context.Background(), 99, vector)
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		err := empty.Add(context.Background(), 100, vector[:63])
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}

	valid := HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 1}
	{
		_, err := empty.SearchHNSWRaBitQ(nil, vector, valid)
		require.Error(t, err,
			"nil search context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := empty.SearchHNSWRaBitQ(canceled, vector, valid)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := empty.SearchHNSWRaBitQ(context.Background(), vector[:63], valid)
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}

	nonFinite := slices.Clone(vector)
	nonFinite[1] = float32(math.NaN())
	{
		_, err := empty.SearchHNSWRaBitQ(context.Background(), nonFinite, valid)
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}

	valid.EF = 0
	{
		_, err := empty.SearchHNSWRaBitQ(context.Background(), vector, valid)
		require.ErrorIs(t, err, ErrInvalidHNSWEF)
	}
}

func TestHNSWRaBitQIncrementalFailuresAreAtomic(t *testing.T) {
	index := buildHNSWRaBitQ(t, hnswRaBitQCandidates(120, 64), hnswRaBitQTestOptions(MetricL2))
	before, err := encodeHNSWRaBitQIndex(context.Background(), index)
	require.NoError(t, err)

	vector := hnswRaBitQCandidates(1, 64)[0].Vector
	{
		err := index.Add(nil, 999999, vector)
		require.Error(t, err,
			"nil Add context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Add(canceled, 999999, vector)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Add(context.Background(), 999999, vector[:63])
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}

	nonFinite := slices.Clone(vector)
	nonFinite[0] = float32(math.Inf(1))
	{
		err := index.Add(context.Background(), 999999, nonFinite)
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}

	midClone := newCancelAfterChecks(4)
	{
		err := index.Add(midClone, 999999, vector)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := encodeHNSWRaBitQIndex(context.Background(), index)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, before),
		"failed Add changed HNSW-RaBitQ generation")
	{
		err := index.Add(context.Background(), 999999, vector)
		require.NoError(t, err)
	}
	require.True(t, index.Len() == 121)
}

func TestHNSWRaBitQConcurrentAddSearchSaveAndOpen(t *testing.T) {
	builder, err := NewHNSWRaBitQBuilder(64, hnswRaBitQTestOptions(MetricCosine))
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	dir := t.TempDir()
	errCh := make(chan error, 32)
	var writers sync.WaitGroup
	for worker := 0; worker < 3; worker++ {
		writers.Add(1)
		go func(worker int) {
			defer writers.Done()
			for value := 0; value < 12; value++ {
				vector := hnswRaBitQCandidates(1, 64)[0].Vector
				vector[worker] += float32(value+1) / 7
				key := uint64(worker*100 + value + 1)
				if err := index.Add(context.Background(), key, vector); err != nil {
					errCh <- err
					return
				}
			}
		}(worker)
	}
	stop := make(chan struct{})
	var readers sync.WaitGroup
	for worker := 0; worker < 3; worker++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			query := make([]float32, 64)
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := index.Search(context.Background(), query, 5); err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for generation := 0; generation < 8; generation++ {
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.hrbtq", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenHNSWRaBitQIndex(context.Background(), path); err != nil {
				errCh <- err
				return
			}
		}
	}()
	writers.Wait()
	close(stop)
	readers.Wait()
	close(errCh)
	for err := range errCh {
		require.NoError(t, err)
	}
	require.True(t, index.Len() == 36)
}

func BenchmarkHNSWRaBitQSearch(b *testing.B) {
	candidates := hnswRaBitQCandidates(2_000, 64)
	options := hnswRaBitQTestOptions(MetricL2)
	options.M = 12
	options.EFConstruction = 80
	index := buildHNSWRaBitQ(b, candidates, options)
	query := candidates[713].Vector
	for _, refine := range []bool{false, true} {
		name := "Approximate"
		if refine {
			name = "Refined"
		}
		b.Run(name, func(b *testing.B) {
			search := HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: 100, Refine: refine}
			b.ReportAllocs()
			for b.Loop() {
				{
					_, err := index.SearchHNSWRaBitQ(context.Background(), query, search)
					if err != nil {
						require.NoError(b, err)
					}
				}
			}
		})
	}
}

func hnswRaBitQTestOptions(metric Metric) HNSWRaBitQBuildOptions {
	options := DefaultHNSWRaBitQBuildOptions(metric)
	options.TotalBits = 7
	options.Clusters = 8
	options.MaxIterations = 8
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x726162697471
	return options
}

func hnswRaBitQCandidates(count, dimension int) []Candidate {
	vectors := raBitQGaussianVectors(count, dimension, uint64(count*17+dimension))
	candidates := make([]Candidate, count)
	for index := range candidates {
		candidates[index] = Candidate{Key: uint64(index*13 + 7), Vector: vectors[index]}
	}
	return candidates
}

func buildHNSWRaBitQ(t testing.TB, candidates []Candidate, options HNSWRaBitQBuildOptions) *HNSWRaBitQIndex {
	t.Helper()
	dimension := 64
	if len(candidates) != 0 {
		dimension = len(candidates[0].Vector)
	}
	builder, err := NewHNSWRaBitQBuilder(dimension, options)
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
