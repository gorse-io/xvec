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
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestHNSWBuildOptionsAndValidation(t *testing.T) {
	t.Parallel()
	defaults := DefaultHNSWBuildOptions(MetricCosine)
	require.True(t, defaults.M == 50)
	require.True(t, defaults.EFConstruction == 500)
	require.Equal(t, MetricCosine, defaults.Metric)

	valid := DefaultHNSWBuildOptions(MetricL2)
	for _, options := range []HNSWBuildOptions{
		{},
		func() HNSWBuildOptions { value := valid; value.Metric = 0; return value }(),
		func() HNSWBuildOptions { value := valid; value.M = 0; return value }(),
		func() HNSWBuildOptions { value := valid; value.M = MaxHNSWM + 1; return value }(),
		func() HNSWBuildOptions { value := valid; value.EFConstruction = value.M - 1; return value }(),
	} {
		{
			_, err := NewHNSWBuilder(3, options)
			require.ErrorIs(t, err, ErrInvalidHNSWOptions)
		}
	}
	{
		_, err := NewHNSWBuilder(0, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
	{
		_, err := NewHNSWBuilder(MaxRotationDimension+1, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
}

func TestHNSWBuildGraphInvariants(t *testing.T) {
	t.Parallel()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		options := DefaultHNSWBuildOptions(metric)
		options.M = 4
		options.EFConstruction = 16
		options.Seed = 0x5eed
		builder, err := NewHNSWBuilder(3, options)
		require.NoError(t, err)

		inputs := hnswBuildInputs(80)
		for _, input := range inputs {
			{
				err := builder.Add(context.Background(), input.Key, input.Vector)
				require.NoError(t, err)
			}
		}
		index, err := builder.Build(context.Background())
		require.NoError(t, err)
		require.True(t, index.Dimension() == 3)
		require.Equal(t, metric, index.Metric())
		require.Len(t, inputs, index.Len())
		require.Equal(t, options, index.BuildOptions())

		entryKey, found := index.EntryPoint()
		require.True(t, found)

		entryLevel, _ := index.Level(entryKey)
		require.Equal(t, index.MaxLevel(), entryLevel)

		assertHNSWGraphInvariants(t, index)
	}
}

func TestHNSWBuildDeterministicAndOwned(t *testing.T) {
	t.Parallel()
	inputs := hnswBuildInputs(120)
	build := func() *HNSWIndex {
		options := DefaultHNSWBuildOptions(MetricL2)
		options.M = 3
		options.EFConstruction = 12
		options.Seed = 42
		builder, err := NewHNSWBuilder(3, options)
		require.NoError(t, err)

		for _, input := range inputs {
			{
				err := builder.Add(context.Background(), input.Key, input.Vector)
				require.NoError(t, err)
			}
		}
		index, err := builder.Build(context.Background())
		require.NoError(t, err)

		return index
	}
	first, second := build(), build()
	require.Equal(t, second.entryPoint, first.entryPoint,

		"fixed seed and insertion order produced different HNSW graphs")
	require.Equal(t, second.maxLevel, first.maxLevel,

		"fixed seed and insertion order produced different HNSW graphs")
	require.Equal(t, second.levelRNGState, first.levelRNGState,

		"fixed seed and insertion order produced different HNSW graphs")
	require.Equal(t, second.levels, first.levels,
		"fixed seed and insertion order produced different HNSW graphs")
	require.Equal(t, second.neighbors, first.neighbors,
		"fixed seed and insertion order produced different HNSW graphs")

	original, found := first.Vector(inputs[0].Key)
	require.True(t, found,
		"first input missing")

	inputs[0].Vector[0] = -999
	{
		got, _ := first.Vector(inputs[0].Key)
		require.Equal(t, original[0], got[0],
			"builder did not own input vector")
	}

	original[0] = -888
	{
		got, _ := first.Vector(inputs[0].Key)
		require.NotEqual(t, float32(-888), got[0],
			"Vector exposed mutable storage")
	}

	neighbors, err := first.Neighbors(first.keys[1], 0)
	require.NoError(t, err)

	if len(neighbors) != 0 {
		neighbors[0] = math.MaxUint64
		again, err := first.Neighbors(first.keys[1], 0)
		require.NoError(t, err)
		require.False(t, slices.Equal(neighbors, again),
			"Neighbors exposed mutable storage")
	}
}

func TestHNSWBuildWithWorkers(t *testing.T) {
	t.Parallel()
	inputs := hnswBuildInputs(1200)
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 42
	build := func(workers int) *HNSWIndex {
		builder, err := NewHNSWBuilder(3, options)
		require.NoError(t, err)
		for _, input := range inputs {
			require.NoError(t, builder.Add(context.Background(), input.Key, input.Vector))
		}
		index, err := builder.BuildWithWorkers(context.Background(), workers)
		require.NoError(t, err)
		return index
	}

	serial := build(1)
	parallel := build(4)
	require.Equal(t, serial.levels, parallel.levels)
	require.Equal(t, serial.levelRNGState, parallel.levelRNGState)
	assertHNSWGraphInvariants(t, parallel)
	results, err := parallel.SearchHNSW(context.Background(), inputs[711].Vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 10}, EF: 80,
	})
	require.NoError(t, err)
	require.NotEmpty(t, results)
	require.Equal(t, inputs[711].Key, results[0].Key)
}

func TestHNSWBuildWithWorkersValidationAndRetry(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 2
	options.EFConstruction = 8
	builder, err := NewHNSWBuilder(2, options)
	require.NoError(t, err)
	require.NoError(t, builder.Add(context.Background(), 1, []float32{1, 2}))

	_, err = builder.BuildWithWorkers(context.Background(), 0)
	require.ErrorIs(t, err, ErrInvalidHNSWWorkers)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = builder.BuildWithWorkers(canceled, 2)
	require.ErrorIs(t, err, context.Canceled)
	index, err := builder.BuildWithWorkers(context.Background(), 2)
	require.NoError(t, err)
	require.Equal(t, 1, index.Len())
}

func TestParallelHNSWBuildExecutesConcurrentInsertions(t *testing.T) {
	t.Parallel()
	levels := make([]int, 8)
	neighbors := make([][][]int, len(levels))
	for position := range neighbors {
		neighbors[position] = make([][]int, 1)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var active, peak atomic.Int32
	var releaseOnce sync.Once
	release := make(chan struct{})
	score := func(left, right int) (float32, error) {
		current := active.Add(1)
		defer active.Add(-1)
		for {
			observed := peak.Load()
			if current <= observed || peak.CompareAndSwap(observed, current) {
				break
			}
		}
		if current >= 2 {
			releaseOnce.Do(func() { close(release) })
		}
		select {
		case <-release:
		case <-ctx.Done():
			return 0, ctx.Err()
		}
		delta := left - right
		return float32(delta * delta), nil
	}
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 2
	options.EFConstruction = 8
	_, _, err := buildParallelHNSW(ctx, 4, options, levels, neighbors, score)
	require.NoError(t, err)
	require.GreaterOrEqual(t, peak.Load(), int32(2))
}

func TestHNSWBuildEmptySingleAndLevels(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 2
	options.EFConstruction = 4
	options.Seed = 9
	builder, _ := NewHNSWBuilder(2, options)
	empty, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, empty.Len() == 0)
	require.Equal(t, -1, empty.MaxLevel())
	{
		_, found := empty.EntryPoint()
		require.False(t, found,
			"empty graph has entry point")
	}
	{
		_, err := empty.Neighbors(1, 0)
		require.ErrorIs(t, err, ErrHNSWKeyNotFound)
	}

	builder, _ = NewHNSWBuilder(2, options)
	{
		err := builder.Add(context.Background(), 17, []float32{1, 2})
		require.NoError(t, err)
	}

	single, err := builder.Build(context.Background())
	require.NoError(t, err)

	entry, found := single.EntryPoint()
	require.True(t, found)
	require.True(t, entry == 17)

	level, _ := single.Level(17)
	for current := 0; current <= level; current++ {
		neighbors, err := single.Neighbors(17, current)
		require.NoError(t, err)
		require.Len(t, neighbors, 0)
	}
	{
		_, err := single.Neighbors(17, level+1)
		require.ErrorIs(t, err, ErrInvalidHNSWLevel)
	}
}

func TestHNSWBuilderLifecycleAndErrors(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 2
	options.EFConstruction = 8
	builder, _ := NewHNSWBuilder(2, options)
	{
		err := builder.Add(nil, 1, []float32{1, 2})
		require.Error(t, err,
			"nil add context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := builder.Add(canceled, 1, []float32{1, 2})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1})
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1, float32(math.NaN())})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1, 2})
		require.NoError(t, err)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{2, 3})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		_, err := builder.Build(canceled)
		require.ErrorIs(t, err, context.Canceled)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Len() == 1)
	{
		err := builder.Add(context.Background(), 2, []float32{2, 3})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}

	var nilBuilder *HNSWBuilder
	{
		err := nilBuilder.Add(context.Background(), 1, []float32{1})
		require.Error(t, err,
			"nil builder add succeeded")
	}
	{
		_, err := nilBuilder.Build(context.Background())
		require.Error(t, err,
			"nil builder build succeeded")
	}
}

func TestHNSWLevelSamplingDeterministicAndBounded(t *testing.T) {
	t.Parallel()
	first := splitMix64{state: 123}
	second := splitMix64{state: 123}
	levels := make([]int, 10000)
	upper := 0
	for index := range levels {
		levels[index] = sampleHNSWLevel(&first, 4)
		{
			got := sampleHNSWLevel(&second, 4)
			require.Equal(t, levels[index], got)
		}
		require.True(t, levels[index] >= 0)
		require.True(t, levels[index] <= MaxHNSWLevel)

		if levels[index] > 0 {
			upper++
		}
	}
	require.True(t, upper >= 2000)
	require.True(t, upper <= 3000)
}

func BenchmarkHNSWBuild(b *testing.B) {
	inputs := hnswBuildInputs(1000)
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 16
	options.EFConstruction = 100
	for _, workers := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			for b.Loop() {
				builder, err := NewHNSWBuilder(3, options)
				if err != nil {
					require.NoError(b, err)
				}

				for _, input := range inputs {
					if err := builder.Add(context.Background(), input.Key, input.Vector); err != nil {
						require.NoError(b, err)
					}
				}
				if _, err := builder.BuildWithWorkers(context.Background(), workers); err != nil {
					require.NoError(b, err)
				}
			}
		})
	}
}

func hnswBuildInputs(count int) []Candidate {
	inputs := make([]Candidate, count)
	for index := range inputs {
		value := float32(index + 1)
		inputs[index] = Candidate{
			Key: uint64(index*17 + 3),
			Vector: []float32{
				float32((index*7)%31) + 0.25,
				float32((index*13)%37) + 0.5,
				value/float32(count+1) + 0.75,
			},
		}
	}
	return inputs
}

func assertHNSWGraphInvariants(t testing.TB, index *HNSWIndex) {
	t.Helper()
	require.Len(t, index.keys, len(index.levels),
		"inconsistent HNSW top-level storage")
	require.Len(t, index.keys, len(index.neighbors),
		"inconsistent HNSW top-level storage")
	require.Len(t, index.positions, len(index.keys),
		"inconsistent HNSW top-level storage")

	maxLevel := -1
	for position, key := range index.keys {
		{
			mapped := index.positions[key]
			require.Equal(t, position, mapped)
		}

		level := index.levels[position]
		maxLevel = max(maxLevel, level)
		require.True(t, level >= 0)
		require.True(t, level <= MaxHNSWLevel)
		require.Len(t, index.neighbors[position], level+1)

		for currentLevel, neighbors := range index.neighbors[position] {
			limit := index.options.M
			if currentLevel == 0 {
				limit *= 2
			}
			require.True(t, len(neighbors) <= limit)

			seen := make(map[int]struct{}, len(neighbors))
			for _, neighbor := range neighbors {
				require.True(t, neighbor >= 0)
				require.True(t, neighbor < len(index.keys))
				require.NotEqual(t, position, neighbor)
				require.True(t, index.levels[neighbor] >= currentLevel)
				{
					_, duplicate := seen[neighbor]
					require.False(t, duplicate)
				}

				seen[neighbor] = struct{}{}
			}
		}
	}
	require.Equal(t, index.maxLevel, maxLevel)
	require.False(t, len(index.keys) != 0 && index.levels[index.entryPoint] != maxLevel)
}

func TestHNSWSearchSmallGraphMatchesFlat(t *testing.T) {
	t.Parallel()
	inputs := hnswBuildInputs(200)
	query := []float32{7.25, 13.5, 1.25}
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		index := buildSearchHNSW(t, metric, inputs, 8, 40)
		options := HNSWSearchOptions{
			SearchOptions: SearchOptions{
				TopK:   17,
				Radius: 100,
				Filter: func(key uint64) bool { return key%2 == 0 },
			},
			EF: 1,
		}
		if metric == MetricIP {
			options.Radius = 10
		}
		got, err := index.SearchHNSW(context.Background(), query, options)
		require.NoError(t, err)

		want, err := topKCandidatesWithOptions(context.Background(), metric, query, options.SearchOptions, len(inputs), func(position int) Candidate {
			return inputs[position]
		}, true)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
}

func TestHNSWSearchLargeGraphRecall(t *testing.T) {
	inputs := hnswBuildInputs(DefaultHNSWBruteForceThreshold + 250)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		index := buildSearchHNSW(t, metric, inputs, 16, 120)
		var matched, total int
		for queryIndex := 0; queryIndex < 20; queryIndex++ {
			query := inputs[(queryIndex*53+11)%len(inputs)].Vector
			got, err := index.SearchHNSW(context.Background(), query, HNSWSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, EF: 120,
			})
			require.NoError(t, err)

			want, err := TopK(context.Background(), metric, query, inputs, 10)
			require.NoError(t, err)

			truth := make(map[uint64]struct{}, len(want))
			for _, result := range want {
				truth[result.Key] = struct{}{}
			}
			for _, result := range got {
				if _, found := truth[result.Key]; found {
					matched++
				}
			}
			total += len(want)
		}
		recall := float64(matched) / float64(total)
		require.True(t, recall >= 0.80)
	}
}

func TestHNSWSearchParallelBuildRecall(t *testing.T) {
	inputs := hnswBuildInputs(DefaultHNSWBruteForceThreshold + 250)
	index := buildSearchHNSWWithWorkers(t, MetricL2, inputs, 16, 120, 4)
	var matched, total int
	for queryIndex := 0; queryIndex < 20; queryIndex++ {
		query := inputs[(queryIndex*53+11)%len(inputs)].Vector
		got, err := index.SearchHNSW(context.Background(), query, HNSWSearchOptions{
			SearchOptions: SearchOptions{TopK: 10}, EF: 120,
		})
		require.NoError(t, err)
		want, err := TopK(context.Background(), MetricL2, query, inputs, 10)
		require.NoError(t, err)
		truth := make(map[uint64]struct{}, len(want))
		for _, result := range want {
			truth[result.Key] = struct{}{}
		}
		for _, result := range got {
			if _, found := truth[result.Key]; found {
				matched++
			}
		}
		total += len(want)
	}
	require.GreaterOrEqual(t, float64(matched)/float64(total), 0.80)
}

func TestHNSWSearchLargeGraphFilterRadiusAndEF(t *testing.T) {
	inputs := hnswBuildInputs(DefaultHNSWBruteForceThreshold + 100)
	index := buildSearchHNSW(t, MetricL2, inputs, 16, 100)
	target := inputs[len(inputs)-37]
	results, err := index.SearchHNSW(context.Background(), target.Vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{
			TopK:   5,
			Radius: 0.0001,
			Filter: func(key uint64) bool { return key == target.Key },
		},
		EF: 1,
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: target.Key, Score: 0}}, results)

	results, err = index.SearchHNSW(context.Background(), target.Vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 2,
	})
	require.NoError(t, err)
	require.Len(t, results, 25)
}

func TestHNSWSearchStableTiesAndDefaults(t *testing.T) {
	t.Parallel()
	inputs := []Candidate{
		{Key: 50, Vector: []float32{1}},
		{Key: 2, Vector: []float32{-1}},
		{Key: 9, Vector: []float32{1}},
	}
	index := buildSearchHNSW(t, MetricL2, inputs, 2, 4)
	results, err := index.Search(context.Background(), []float32{0}, 3)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 2, Score: 1}, {Key: 9, Score: 1}, {Key: 50, Score: 1}}, results)

	results, err = index.Search(context.Background(), []float32{0}, 0)
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Len(t, results, 0)

	options := DefaultHNSWBuildOptions(MetricL2)
	builder, _ := NewHNSWBuilder(1, options)
	empty, _ := builder.Build(context.Background())
	results, err = empty.SearchHNSW(context.Background(), []float32{0}, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, EF: 1,
	})
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Len(t, results, 0)
}

func TestHNSWResultTieBreaksByKey(t *testing.T) {
	t.Parallel()
	index := &HNSWIndex{
		options: HNSWBuildOptions{Metric: MetricL2},
		keys:    []uint64{90, 3},
	}
	left := hnswScoredNode{position: 0, score: 1}
	right := hnswScoredNode{position: 1, score: 1}
	require.False(t, index.hnswResultNodeBetter(left, right),
		"equal HNSW result scores did not prefer the smaller document key")
	require.True(t, index.hnswResultNodeBetter(right, left),
		"equal HNSW result scores did not prefer the smaller document key")
}

func TestHNSWSearchTraversesEqualScoreCandidates(t *testing.T) {
	t.Parallel()
	const count = DefaultHNSWBruteForceThreshold + 1
	index := &HNSWIndex{
		dimension:  1,
		options:    HNSWBuildOptions{Metric: MetricIP, M: 1, EFConstruction: 1},
		keys:       make([]uint64, count),
		vectors:    make([]float32, count),
		levels:     make([]int, count),
		neighbors:  make([][][]int, count),
		entryPoint: 0,
		maxLevel:   0,
	}
	for position := range count {
		index.keys[position] = uint64(position + 100)
		index.vectors[position] = 1
		index.neighbors[position] = make([][]int, 1)
	}
	index.keys[0], index.keys[1], index.keys[2] = 100, 200, 1
	index.neighbors[0][0] = []int{1}
	index.neighbors[1][0] = []int{2}

	results, err := index.SearchHNSW(context.Background(), []float32{1}, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, EF: 1,
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 1, Score: 1}}, results)
}

func TestHNSWSearchValidation(t *testing.T) {
	t.Parallel()
	index := buildSearchHNSW(t, MetricL2, hnswBuildInputs(4), 2, 4)
	valid := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 2}
	{
		_, err := index.SearchHNSW(nil, []float32{1, 2, 3}, valid)
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := index.SearchHNSW(canceled, []float32{1, 2, 3}, valid)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := index.SearchHNSW(context.Background(), []float32{1, 2}, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
	{
		_, err := index.SearchHNSW(context.Background(), []float32{1, float32(math.NaN()), 3}, valid)
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}

	invalidEF := valid
	invalidEF.EF = 0
	{
		_, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidEF)
		require.ErrorIs(t, err, ErrInvalidHNSWEF)
	}

	invalidEF.EF = MaxHNSWEFSearch + 1
	{
		_, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidEF)
		require.ErrorIs(t, err, ErrInvalidHNSWEF)
	}

	valid.EF = MaxHNSWEFSearch
	{
		_, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, valid)
		require.NoError(t, err)
	}

	invalidTopK := valid
	invalidTopK.TopK = 0
	{
		_, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidTopK)
		require.ErrorIs(t, err, ErrInvalidTopK)
	}

	invalidRadius := valid
	invalidRadius.Radius = -1
	{
		_, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidRadius)
		require.ErrorIs(t, err, ErrInvalidRadius)
	}

	var nilIndex *HNSWIndex
	{
		_, err := nilIndex.SearchHNSW(context.Background(), []float32{1}, valid)
		require.Error(t, err,
			"nil index search succeeded")
	}
	{
		err := (HNSWSearchOptions{}).Validate()
		require.ErrorIs(t, err, ErrInvalidTopK)
	}
	{
		err := (HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}}).Validate()
		require.ErrorIs(t, err, ErrInvalidHNSWEF)
	}
	{
		err := (HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: MaxHNSWEFSearch + 1}).Validate()
		require.ErrorIs(t, err, ErrInvalidHNSWEF)
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	inputs := hnswBuildInputs(10000)
	index := buildSearchHNSW(b, MetricL2, inputs, 16, 120)
	query := inputs[4321].Vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: 100}
	b.ResetTimer()
	for b.Loop() {
		{
			_, err := index.SearchHNSW(context.Background(), query, options)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func buildSearchHNSW(t testing.TB, metric Metric, inputs []Candidate, m, efConstruction int) *HNSWIndex {
	return buildSearchHNSWWithWorkers(t, metric, inputs, m, efConstruction, 1)
}

func buildSearchHNSWWithWorkers(t testing.TB, metric Metric, inputs []Candidate, m, efConstruction, workers int) *HNSWIndex {
	t.Helper()
	options := DefaultHNSWBuildOptions(metric)
	options.M = m
	options.EFConstruction = efConstruction
	options.Seed = 0x123456789abcdef
	builder, err := NewHNSWBuilder(len(inputs[0].Vector), options)
	require.NoError(t, err)

	for _, input := range inputs {
		err := builder.Add(context.Background(), input.Key, input.Vector)
		require.NoError(t, err)
	}
	index, err := builder.BuildWithWorkers(context.Background(), workers)
	require.NoError(t, err)

	return index
}

func TestHNSWIncrementalMatchesOneShotBuild(t *testing.T) {
	t.Parallel()
	inputs := hnswBuildInputs(180)
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x3141592653589793

	streamed := buildHNSWPrefix(t, options, inputs[:120])
	for _, input := range inputs[120:] {
		{
			err := streamed.Add(context.Background(), input.Key, input.Vector)
			require.NoError(t, err)
		}
	}
	oneShot := buildHNSWPrefix(t, options, inputs)
	assertSameHNSWIndex(t, streamed, oneShot)

	path := filepath.Join(t.TempDir(), "streamed.hnsw")
	{
		err := streamed.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, reopened, oneShot)

	next := Candidate{Key: 999999, Vector: []float32{3.25, 7.5, 1.125}}
	{
		err := reopened.Add(context.Background(), next.Key, next.Vector)
		require.NoError(t, err)
	}

	all := append(slices.Clone(inputs), next)
	want := buildHNSWPrefix(t, options, all)
	assertSameHNSWIndex(t, reopened, want)
}

func TestHNSWIncrementalEmptyAndOwnership(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricCosine)
	options.M = 4
	options.EFConstruction = 16
	options.Seed = 7
	builder, err := NewHNSWBuilder(2, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	vector := []float32{1, 2}
	{
		err := index.Add(context.Background(), 17, vector)
		require.NoError(t, err)
	}

	vector[0] = -100
	stored, found := index.Vector(17)
	require.True(t, found)
	require.True(t, slices.Equal(stored, []float32{1, 2}))

	entry, found := index.EntryPoint()
	require.True(t, found)
	require.True(t, entry == 17)
	require.True(t, index.Len() == 1)

	results, err := index.Search(context.Background(), []float32{1, 2}, 1)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 17, Score: 0}}, results)
}

func TestHNSWIncrementalFailuresAreAtomic(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricL2, 300)
	before, err := encodeHNSWIndex(context.Background(), index)
	require.NoError(t, err)
	{
		err := index.Add(nil, 100000, []float32{1, 2, 3})
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Add(canceled, 100000, []float32{1, 2, 3})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Add(context.Background(), 100000, []float32{1, 2})
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		err := index.Add(context.Background(), 100000, []float32{1, float32(math.NaN()), 3})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}
	{
		err := index.Add(context.Background(), index.keys[0], []float32{1, 2, 3})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}

	// This context cancels after cloning has begun, exercising rollback after
	// private topology state has already been allocated and copied.
	midClone := newCancelAfterChecks(5)
	{
		err := index.Add(midClone, 100000, []float32{1, 2, 3})
		require.ErrorIs(t, err, context.Canceled)
	}

	midTraversal := newCancelAfterChecks(7)
	{
		err := index.Add(midTraversal, 100000, []float32{1, 2, 3})
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := encodeHNSWIndex(context.Background(), index)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, before),
		"failed incremental add changed graph generation")

	var nilIndex *HNSWIndex
	{
		err := nilIndex.Add(context.Background(), 1, []float32{1})
		require.Error(t, err,
			"nil index add succeeded")
	}

	// A failed add must not consume a sampled level. The next successful add
	// therefore still matches a one-shot build of the same sequence.
	next := Candidate{Key: 100000, Vector: []float32{1, 2, 3}}
	{
		err := index.Add(context.Background(), next.Key, next.Vector)
		require.NoError(t, err)
	}

	inputs := append(hnswBuildInputs(300), next)
	want := buildHNSWPrefix(t, index.options, inputs)
	assertSameHNSWIndex(t, index, want)
}

func TestHNSWConcurrentStreamingSearchAndSave(t *testing.T) {
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 6
	options.EFConstruction = 30
	options.Seed = 123
	builder, err := NewHNSWBuilder(3, options)
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
				vector := []float32{float32(worker + 1), float32(value + 1), float32(worker + value + 1)}
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
				if _, err := index.Search(context.Background(), []float32{1, 2, 3}, 10); err != nil {
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
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.hnsw", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenHNSWIndex(context.Background(), path); err != nil {
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

	path := filepath.Join(dir, "final.hnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	reopened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, reopened, index)
}

func buildHNSWPrefix(t testing.TB, options HNSWBuildOptions, inputs []Candidate) *HNSWIndex {
	t.Helper()
	dimension := 3
	if len(inputs) != 0 {
		dimension = len(inputs[0].Vector)
	}
	builder, err := NewHNSWBuilder(dimension, options)
	require.NoError(t, err)

	for _, input := range inputs {
		{
			err := builder.Add(context.Background(), input.Key, input.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

type cancelAfterChecks struct {
	limit int32
	calls atomic.Int32
	done  chan struct{}
	once  sync.Once
}

func newCancelAfterChecks(limit int32) *cancelAfterChecks {
	return &cancelAfterChecks{limit: limit, done: make(chan struct{})}
}

func (c *cancelAfterChecks) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAfterChecks) Done() <-chan struct{}       { return c.done }
func (c *cancelAfterChecks) Value(any) any               { return nil }
func (c *cancelAfterChecks) Err() error {
	if c.calls.Add(1) < c.limit {
		return nil
	}
	c.once.Do(func() { close(c.done) })
	return context.Canceled
}

func TestHNSWPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricCosine, 160)
	path := filepath.Join(t.TempDir(), "vectors.hnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())

	opened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, opened, index)
	query := []float32{7.25, 13.5, 1.25}
	want, err := index.SearchHNSW(context.Background(), query, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 80,
	})
	require.NoError(t, err)

	got, err := opened.SearchHNSW(context.Background(), query, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 80,
	})
	require.NoError(t, err)
	require.Equal(t, want, got)

	replacement := persistedHNSWIndex(t, MetricIP, 40)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err = OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, opened, replacement)
}

func TestHNSWPersistenceLargeGraphSearch(t *testing.T) {
	index := persistedHNSWIndex(t, MetricL2, DefaultHNSWBruteForceThreshold+100)
	path := filepath.Join(t.TempDir(), "large.hnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	query := hnswBuildInputs(index.Len())[713].Vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EF: 120}
	want, err := index.SearchHNSW(context.Background(), query, options)
	require.NoError(t, err)

	got, err := opened.SearchHNSW(context.Background(), query, options)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestHNSWPersistenceEmpty(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricMIPSL2)
	options.M = 4
	options.EFConstruction = 12
	options.Seed = 17
	builder, err := NewHNSWBuilder(7, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "empty.hnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, opened, index)
}

func TestHNSWPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricL2, 32)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.hnsw")
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
		"canceled replacement changed published HNSW file")
	{
		_, err := OpenHNSWIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := encodeHNSWIndex(canceled, index)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := decodeHNSWIndex(canceled, original)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenHNSWIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		_, err := encodeHNSWIndex(nil, index)
		require.Error(t, err,
			"nil encode context succeeded")
	}
	{
		_, err := decodeHNSWIndex(nil, original)
		require.Error(t, err,
			"nil decode context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}
	{
		_, err := OpenHNSWIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}
	{
		_, err := OpenHNSWIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	var nilIndex *HNSWIndex
	{
		err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil.hnsw"))
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}

	invalid := &HNSWIndex{
		dimension: 3,
		options:   DefaultHNSWBuildOptions(MetricL2),
		keys:      []uint64{1},
	}
	{
		err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.hnsw"))
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}
}

func TestHNSWPersistenceDetectsTruncationAndCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(t, MetricL2, 32))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, hnswHeaderSize - 1, hnswHeaderSize, len(valid) - 1} {
		{
			_, err := decodeHNSWIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidHNSWFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeHNSWIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeHNSWIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], hnswFileVersion+1)
	{
		_, err := decodeHNSWIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedHNSWVersion)
	}

	badHeaderCRC := slices.Clone(valid)
	badHeaderCRC[44] ^= 1
	{
		_, err := decodeHNSWIndex(context.Background(), badHeaderCRC)
		require.ErrorIs(t, err, ErrHNSWChecksumMismatch)
	}

	badPayloadCRC := slices.Clone(valid)
	badPayloadCRC[len(badPayloadCRC)-1] ^= 1
	{
		_, err := decodeHNSWIndex(context.Background(), badPayloadCRC)
		require.ErrorIs(t, err, ErrHNSWChecksumMismatch)
	}
}

func TestHNSWPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(t, MetricL2, 32))
	require.NoError(t, err)

	records := parseHNSWRecordOffsets(t, valid)
	require.True(t, len(records) >= 2,
		"persistence fixture lacks required graph edges")
	require.True(t, len(records[0].neighborOffsets) >= 2,
		"persistence fixture lacks required graph edges")

	upperNeighborOffset := -1
	for _, record := range records {
		if len(record.upperNeighborOffsets) != 0 {
			upperNeighborOffset = record.upperNeighborOffsets[0]
			break
		}
	}
	lowLevelPosition := -1
	for position, record := range records {
		if record.maxLevel == 0 {
			lowLevelPosition = position
			break
		}
	}
	require.True(t, upperNeighborOffset >= 0,
		"persistence fixture lacks an upper edge or level-zero target")
	require.True(t, lowLevelPosition >= 0,
		"persistence fixture lacks an upper edge or level-zero target")

	tests := []struct {
		name   string
		mutate func([]byte)
	}{
		{
			name: "duplicate key",
			mutate: func(data []byte) {
				copy(data[records[1].key:records[1].key+8], data[records[0].key:records[0].key+8])
			},
		},
		{
			name: "non-finite vector",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[records[0].vector:records[0].vector+4], math.Float32bits(float32(math.NaN())))
			},
		},
		{
			name: "invalid level",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[records[0].level:records[0].level+4], MaxHNSWLevel+1)
			},
		},
		{
			name: "neighbor out of range",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[records[0].neighborOffsets[0]:records[0].neighborOffsets[0]+4], uint32(len(records)))
			},
		},
		{
			name: "duplicate neighbor",
			mutate: func(data []byte) {
				copy(data[records[0].neighborOffsets[1]:records[0].neighborOffsets[1]+4], data[records[0].neighborOffsets[0]:records[0].neighborOffsets[0]+4])
			},
		},
		{
			name: "neighbor lacks level",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[upperNeighborOffset:upperNeighborOffset+4], uint32(lowLevelPosition))
			},
		},
		{
			name: "degree exceeds limit",
			mutate: func(data []byte) {
				m := binary.LittleEndian.Uint32(data[44:48])
				binary.LittleEndian.PutUint32(data[records[0].degree:records[0].degree+4], m*2+1)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			rechecksumHNSW(encoded)
			{
				_, err := decodeHNSWIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidHNSWFile)
			}
		})
	}

	headerTests := []struct {
		name   string
		mutate func([]byte)
	}{
		{"invalid options", func(data []byte) { binary.LittleEndian.PutUint32(data[44:48], 0) }},
		{"entry out of range", func(data []byte) { binary.LittleEndian.PutUint64(data[56:64], uint64(len(records))) }},
		{"maximum level mismatch", func(data []byte) { binary.LittleEndian.PutUint32(data[64:68], MaxHNSWLevel) }},
		{"reserved field", func(data []byte) { data[88] = 1 }},
		{"file length", func(data []byte) { binary.LittleEndian.PutUint64(data[16:24], uint64(len(data)+1)) }},
	}
	for _, test := range headerTests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
			{
				_, err := decodeHNSWIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidHNSWFile)
			}
		})
	}
}

func FuzzDecodeHNSWIndex(f *testing.F) {
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(f, MetricL2, 12))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("ZVECHNSW"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeHNSWIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateHNSWIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func persistedHNSWIndex(t testing.TB, metric Metric, count int) *HNSWIndex {
	t.Helper()
	options := DefaultHNSWBuildOptions(metric)
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x123456789abcdef0
	builder, err := NewHNSWBuilder(3, options)
	require.NoError(t, err)

	for _, input := range hnswBuildInputs(count) {
		{
			err := builder.Add(context.Background(), input.Key, input.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

func assertSameHNSWIndex(t testing.TB, got, want *HNSWIndex) {
	t.Helper()
	require.Equal(t, want.Dimension(), got.Dimension())
	require.Equal(t, want.Metric(), got.Metric())
	require.Equal(t, want.Len(), got.Len())
	require.Equal(t, want.BuildOptions(), got.BuildOptions())
	require.Equal(t, want.entryPoint, got.entryPoint)
	require.Equal(t, want.MaxLevel(), got.MaxLevel())
	require.Equal(t, want.levelRNGState, got.levelRNGState)
	require.True(t, slices.Equal(got.keys, want.keys))
	require.True(t, slices.Equal(got.levels, want.levels))
	require.Equal(t, want.neighbors, got.neighbors)

	for _, key := range want.keys {
		gotVector, gotOK := got.Vector(key)
		wantVector, wantOK := want.Vector(key)
		require.Equal(t, wantOK, gotOK)
		require.True(t, slices.Equal(gotVector, wantVector))
	}
}

type hnswRecordOffset struct {
	key                  int
	level                int
	maxLevel             int
	vector               int
	degree               int
	neighborOffsets      []int
	upperNeighborOffsets []int
}

func parseHNSWRecordOffsets(t testing.TB, encoded []byte) []hnswRecordOffset {
	t.Helper()
	count := int(binary.LittleEndian.Uint64(encoded[32:40]))
	dimension := int(binary.LittleEndian.Uint32(encoded[40:44]))
	offset := hnswHeaderSize
	records := make([]hnswRecordOffset, count)
	for position := range records {
		records[position].key = offset
		offset += 8
		records[position].level = offset
		level := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		records[position].maxLevel = level
		offset += 4
		records[position].vector = offset
		offset += dimension * 4
		for currentLevel := 0; currentLevel <= level; currentLevel++ {
			degreeOffset := offset
			degree := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
			offset += 4
			if currentLevel == 0 {
				records[position].degree = degreeOffset
				for neighborIndex := 0; neighborIndex < degree; neighborIndex++ {
					records[position].neighborOffsets = append(records[position].neighborOffsets, offset)
					offset += 4
				}
			} else {
				for neighborIndex := 0; neighborIndex < degree; neighborIndex++ {
					records[position].upperNeighborOffsets = append(records[position].upperNeighborOffsets, offset)
					offset += 4
				}
			}
		}
	}
	require.Len(t, encoded, offset)

	return records
}

func rechecksumHNSW(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[84:88], ailego.CRC32C(encoded[hnswHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
