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
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestSparseHNSWBuildOptionsAndValidation(t *testing.T) {
	t.Parallel()
	defaults := DefaultSparseHNSWBuildOptions()
	require.Equal(t, MetricIP, defaults.Metric)
	require.Equal(t, DefaultHNSWM, defaults.M)
	require.Equal(t, DefaultHNSWEFConstruction, defaults.EFConstruction)

	for _, options := range []HNSWBuildOptions{
		{},
		DefaultHNSWBuildOptions(MetricL2),
		func() HNSWBuildOptions { value := defaults; value.M = 0; return value }(),
		func() HNSWBuildOptions { value := defaults; value.EFConstruction = value.M - 1; return value }(),
	} {
		{
			_, err := NewSparseHNSWBuilder(options)
			require.ErrorIs(t, err, ErrInvalidHNSWOptions)
		}
	}
}

func TestSparseHNSWBuildGraphInvariants(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 5
	options.EFConstruction = 24
	options.Seed = 0x5eed
	inputs := sparseHNSWBuildInputs(120)
	index := buildSparseHNSW(t, options, inputs)
	require.Equal(t, MetricIP, index.Metric())
	require.Len(t, inputs, index.Len())
	require.Equal(t, options, index.BuildOptions())

	entryKey, found := index.EntryPoint()
	require.True(t, found,
		"graph has no entry point")

	entryLevel, _ := index.Level(entryKey)
	require.Equal(t, index.MaxLevel(), entryLevel)

	assertSparseHNSWGraphInvariants(t, index)
}

func TestSparseHNSWBuildDeterministicAndOwned(t *testing.T) {
	t.Parallel()
	inputs := sparseHNSWBuildInputs(140)
	options := DefaultSparseHNSWBuildOptions()
	options.M = 4
	options.EFConstruction = 20
	options.Seed = 42
	first := buildSparseHNSW(t, options, inputs)
	second := buildSparseHNSW(t, options, inputs)
	require.Equal(t, second.entryPoint, first.entryPoint,

		"fixed seed and insertion order produced different sparse HNSW graphs")
	require.Equal(t, second.maxLevel, first.maxLevel,

		"fixed seed and insertion order produced different sparse HNSW graphs")
	require.Equal(t, second.levelRNGState, first.levelRNGState,

		"fixed seed and insertion order produced different sparse HNSW graphs")
	require.True(t, slices.Equal(first.levels, second.levels),
		"fixed seed and insertion order produced different sparse HNSW graphs")
	require.Equal(t, second.neighbors, first.neighbors,
		"fixed seed and insertion order produced different sparse HNSW graphs")

	original, found := first.SparseVector(inputs[0].key)
	require.True(t, found,
		"first input missing")

	inputs[0].vector.Indices[0] = math.MaxUint32
	inputs[0].vector.Values[0] = -999
	{
		got, _ := first.SparseVector(inputs[0].key)
		require.Equal(t, original, got,
			"builder did not own sparse input")
	}

	original.Indices[0] = math.MaxUint32
	original.Values[0] = -888
	{
		got, _ := first.SparseVector(inputs[0].key)
		require.NotEqual(t, uint32(math.MaxUint32), got.Indices[0],
			"SparseVector exposed mutable storage")
		require.NotEqual(t, float32(-888), got.Values[0],
			"SparseVector exposed mutable storage")
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

func TestSparseHNSWBuildWithWorkers(t *testing.T) {
	t.Parallel()
	inputs := sparseHNSWBuildInputs(1200)
	options := DefaultSparseHNSWBuildOptions()
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 42
	build := func(workers int) *SparseHNSWIndex {
		builder, err := NewSparseHNSWBuilder(options)
		require.NoError(t, err)
		for _, input := range inputs {
			require.NoError(t, builder.AddSparse(context.Background(), input.key, input.vector))
		}
		index, err := builder.BuildWithWorkers(context.Background(), workers)
		require.NoError(t, err)
		return index
	}

	serial := build(1)
	parallel := build(4)
	require.Equal(t, serial.levels, parallel.levels)
	require.Equal(t, serial.levelRNGState, parallel.levelRNGState)
	assertSparseHNSWGraphInvariants(t, parallel)
	results, err := parallel.SearchSparseHNSW(context.Background(), inputs[711].vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 10}, EF: 80,
	})
	require.NoError(t, err)
	require.NotEmpty(t, results)
}

func TestSparseHNSWBuildWithWorkersValidationAndRetry(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 2
	options.EFConstruction = 8
	builder, err := NewSparseHNSWBuilder(options)
	require.NoError(t, err)
	require.NoError(t, builder.AddSparse(context.Background(), 1, SparseVector{Indices: []uint32{1}, Values: []float32{2}}))

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

func TestSparseHNSWBuildEmptySingleAndLevels(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 2
	options.EFConstruction = 8
	options.Seed = 9
	builder, _ := NewSparseHNSWBuilder(options)
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

	builder, _ = NewSparseHNSWBuilder(options)
	{
		err := builder.AddSparse(context.Background(), 17, SparseVector{})
		require.NoError(t, err)
	}

	single, err := builder.Build(context.Background())
	require.NoError(t, err)

	entry, found := single.EntryPoint()
	require.True(t, found)
	require.True(t, entry == 17)

	vector, found := single.SparseVector(17)
	require.True(t, found)
	require.Len(t, vector.Indices, 0)
	require.Len(t, vector.Values, 0)

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

func TestSparseHNSWBuilderLifecycleAndErrors(t *testing.T) {
	t.Parallel()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 3
	options.EFConstruction = 12
	builder, _ := NewSparseHNSWBuilder(options)
	valid := SparseVector{Indices: []uint32{1, 9}, Values: []float32{2, 3}}
	{
		err := builder.AddSparse(nil, 1, valid)
		require.Error(t, err,
			"nil add context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := builder.AddSparse(canceled, 1, valid)
		require.ErrorIs(t, err, context.Canceled)
	}

	invalid := []struct {
		vector SparseVector
		want   error
	}{
		{SparseVector{Indices: []uint32{1}, Values: nil}, ailego.ErrDimensionMismatch},
		{SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{SparseVector{Indices: []uint32{1, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.NaN())}}, ailego.ErrNonFiniteVector},
	}
	for _, test := range invalid {
		{
			err := builder.AddSparse(context.Background(), 1, test.vector)
			require.ErrorIs(t, err, test.want)
		}
	}
	{
		err := builder.AddSparse(context.Background(), 1, valid)
		require.NoError(t, err)
	}
	{
		err := builder.AddSparse(context.Background(), 1, valid)
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
		err := builder.AddSparse(context.Background(), 2, SparseVector{})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}

	var nilBuilder *SparseHNSWBuilder
	{
		err := nilBuilder.AddSparse(context.Background(), 1, SparseVector{})
		require.Error(t, err,
			"nil builder add succeeded")
	}
	{
		_, err := nilBuilder.Build(context.Background())
		require.Error(t, err,
			"nil builder build succeeded")
	}
}

func BenchmarkSparseHNSWBuild(b *testing.B) {
	inputs := sparseHNSWBuildInputs(1000)
	options := DefaultSparseHNSWBuildOptions()
	options.M = 16
	options.EFConstruction = 100
	for b.Loop() {
		builder, err := NewSparseHNSWBuilder(options)
		if err != nil {
			require.NoError(b, err)
		}

		for _, input := range inputs {
			{
				err := builder.AddSparse(context.Background(), input.key, input.vector)
				if err != nil {
					require.NoError(b, err)
				}
			}
		}
		{
			_, err := builder.Build(context.Background())
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

type sparseHNSWInput struct {
	key    uint64
	vector SparseVector
}

func sparseHNSWBuildInputs(count int) []sparseHNSWInput {
	inputs := make([]sparseHNSWInput, count)
	for position := range inputs {
		inputs[position] = sparseHNSWInput{
			key: uint64(position*19 + 5),
			vector: SparseVector{
				Indices: []uint32{uint32(position % 31), uint32(100 + position%37), uint32(200 + position%43)},
				Values: []float32{
					float32(position%7) + 0.25,
					float32(position%11) + 0.5,
					float32(position%13) + 0.75,
				},
			},
		}
	}
	return inputs
}

func buildSparseHNSW(t testing.TB, options HNSWBuildOptions, inputs []sparseHNSWInput) *SparseHNSWIndex {
	t.Helper()
	builder, err := NewSparseHNSWBuilder(options)
	require.NoError(t, err)

	for _, input := range inputs {
		{
			err := builder.AddSparse(context.Background(), input.key, input.vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

func assertSparseHNSWGraphInvariants(t testing.TB, index *SparseHNSWIndex) {
	t.Helper()
	count := len(index.keys)
	require.Len(t, index.offsets, count+1,

		"inconsistent sparse HNSW top-level storage")
	require.Len(t, index.indices, len(index.values),

		"inconsistent sparse HNSW top-level storage")
	require.Len(t, index.positions, count,
		"inconsistent sparse HNSW top-level storage")
	require.Len(t, index.levels, count,
		"inconsistent sparse HNSW top-level storage")
	require.Len(t, index.neighbors, count,
		"inconsistent sparse HNSW top-level storage")
	require.True(t, index.offsets[0] == 0,
		"inconsistent sparse HNSW offsets")
	require.Len(t, index.indices, index.offsets[count],
		"inconsistent sparse HNSW offsets")

	derivedMax := -1
	for position, key := range index.keys {
		{
			mapped := index.positions[key]
			require.Equal(t, position, mapped)
		}
		require.True(t, index.offsets[position] <= index.offsets[position+1])

		vector := index.sparseVectorAt(position)
		{
			_, err := ailego.SparseInnerProduct(vector.Indices, vector.Values, nil, nil)
			require.NoError(t, err)
		}

		level := index.levels[position]
		derivedMax = max(derivedMax, level)
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
				require.True(t, neighbor < count)
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
	require.Equal(t, index.maxLevel, derivedMax)
	require.False(t, count != 0 && index.levels[index.entryPoint] != derivedMax)
}

func TestSparseHNSWSearchSmallGraphMatchesFlat(t *testing.T) {
	t.Parallel()
	inputs := sparseHNSWBuildInputs(240)
	index := buildSearchSparseHNSW(t, inputs, 8, 48)
	flat, err := NewSparseFlatIndex(MetricIP)
	require.NoError(t, err)

	for _, input := range inputs {
		{
			err := flat.AddSparse(context.Background(), input.key, input.vector)
			require.NoError(t, err)
		}
	}
	query := SparseVector{Indices: []uint32{3, 107, 211}, Values: []float32{1, 2, 3}}
	options := HNSWSearchOptions{
		SearchOptions: SearchOptions{
			TopK:   17,
			Radius: 2,
			Filter: func(key uint64) bool { return key%2 == 1 },
		},
		EF: 1,
	}
	got, err := index.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)

	want, err := flat.SearchSparseWithOptions(context.Background(), query, options.SearchOptions)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestSparseHNSWSearchLargeGraphRecall(t *testing.T) {
	inputs := sparseHNSWBuildInputs(DefaultHNSWBruteForceThreshold + 300)
	index := buildSearchSparseHNSW(t, inputs, 16, 120)
	var matched, total int
	for queryIndex := 0; queryIndex < 30; queryIndex++ {
		query := inputs[(queryIndex*41+17)%len(inputs)].vector
		got, err := index.SearchSparseHNSW(context.Background(), query, HNSWSearchOptions{
			SearchOptions: SearchOptions{TopK: 10}, EF: 120,
		})
		require.NoError(t, err)

		want, err := exactSparseResults(context.Background(), query, inputs, 10)
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

func TestSparseHNSWSearchParallelBuildRecall(t *testing.T) {
	inputs := sparseHNSWBuildInputs(DefaultHNSWBruteForceThreshold + 300)
	index := buildSearchSparseHNSWWithWorkers(t, inputs, 16, 120, 4)
	var matched, total int
	for queryIndex := 0; queryIndex < 30; queryIndex++ {
		query := inputs[(queryIndex*41+17)%len(inputs)].vector
		got, err := index.SearchSparseHNSW(context.Background(), query, HNSWSearchOptions{
			SearchOptions: SearchOptions{TopK: 10}, EF: 120,
		})
		require.NoError(t, err)
		want, err := exactSparseResults(context.Background(), query, inputs, 10)
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

func TestSparseHNSWSearchLargeGraphFilterRadiusAndEF(t *testing.T) {
	inputs := sparseHNSWBuildInputs(DefaultHNSWBruteForceThreshold + 100)
	index := buildSearchSparseHNSW(t, inputs, 16, 100)
	target := inputs[len(inputs)-37]
	targetScore, err := sparseHNSWScore(target.vector, target.vector)
	require.NoError(t, err)

	results, err := index.SearchSparseHNSW(context.Background(), target.vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{
			TopK:   5,
			Radius: targetScore,
			Filter: func(key uint64) bool { return key == target.key },
		},
		EF: 1,
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: target.key, Score: targetScore}}, results)

	results, err = index.SearchSparseHNSW(context.Background(), target.vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 2,
	})
	require.NoError(t, err)
	require.Len(t, results, 25)
}

func TestSparseHNSWSearchStableTiesAndDefaults(t *testing.T) {
	t.Parallel()
	inputs := []sparseHNSWInput{
		{key: 50, vector: SparseVector{}},
		{key: 2, vector: SparseVector{}},
		{key: 9, vector: SparseVector{}},
	}
	index := buildSearchSparseHNSW(t, inputs, 2, 4)
	results, err := index.SearchSparse(context.Background(), SparseVector{}, 3)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 2}, {Key: 9}, {Key: 50}}, results)

	results, err = index.SearchSparse(context.Background(), SparseVector{}, 0)
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Len(t, results, 0)

	builder, _ := NewSparseHNSWBuilder(DefaultSparseHNSWBuildOptions())
	empty, _ := builder.Build(context.Background())
	results, err = empty.SearchSparseHNSW(context.Background(), SparseVector{}, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, EF: 1,
	})
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Len(t, results, 0)
}

func TestSparseHNSWResultTieBreaksByKey(t *testing.T) {
	t.Parallel()
	index := &SparseHNSWIndex{keys: []uint64{90, 3}}
	left := hnswScoredNode{position: 0, score: 1}
	right := hnswScoredNode{position: 1, score: 1}
	require.False(t, index.resultNodeBetter(left, right),
		"equal sparse HNSW result scores did not prefer the smaller key")
	require.True(t, index.resultNodeBetter(right, left),
		"equal sparse HNSW result scores did not prefer the smaller key")
}

func TestSparseHNSWSearchValidation(t *testing.T) {
	t.Parallel()
	index := buildSearchSparseHNSW(t, sparseHNSWBuildInputs(4), 2, 4)
	query := SparseVector{Indices: []uint32{1}, Values: []float32{2}}
	valid := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 2}
	{
		_, err := index.SearchSparseHNSW(nil, query, valid)
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := index.SearchSparseHNSW(canceled, query, valid)
		require.ErrorIs(t, err, context.Canceled)
	}

	badQueries := []struct {
		query SparseVector
		want  error
	}{
		{SparseVector{Indices: []uint32{1}, Values: nil}, ailego.ErrDimensionMismatch},
		{SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.Inf(1))}}, ailego.ErrNonFiniteVector},
	}
	for _, test := range badQueries {
		{
			_, err := index.SearchSparseHNSW(context.Background(), test.query, valid)
			require.ErrorIs(t, err, test.want)
		}
	}
	invalidEF := valid
	invalidEF.EF = 0
	{
		_, err := index.SearchSparseHNSW(context.Background(), query, invalidEF)
		require.ErrorIs(t, err, ErrInvalidHNSWEF)
	}

	invalidEF.EF = MaxHNSWEFSearch + 1
	{
		_, err := index.SearchSparseHNSW(context.Background(), query, invalidEF)
		require.ErrorIs(t, err, ErrInvalidHNSWEF)
	}

	invalidTopK := valid
	invalidTopK.TopK = 0
	{
		_, err := index.SearchSparseHNSW(context.Background(), query, invalidTopK)
		require.ErrorIs(t, err, ErrInvalidTopK)
	}

	invalidRadius := valid
	invalidRadius.Radius = -1
	{
		_, err := index.SearchSparseHNSW(context.Background(), query, invalidRadius)
		require.ErrorIs(t, err, ErrInvalidRadius)
	}

	var nilIndex *SparseHNSWIndex
	{
		_, err := nilIndex.SearchSparseHNSW(context.Background(), query, valid)
		require.Error(t, err,
			"nil index search succeeded")
	}

	overflow := buildSearchSparseHNSW(t, []sparseHNSWInput{{
		key: 1, vector: SparseVector{Indices: []uint32{1}, Values: []float32{math.MaxFloat32}},
	}}, 2, 4)
	{
		_, err := overflow.SearchSparse(context.Background(), SparseVector{
			Indices: []uint32{1}, Values: []float32{math.MaxFloat32},
		}, 1)
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}
}

func BenchmarkSparseHNSWSearch(b *testing.B) {
	inputs := sparseHNSWBuildInputs(10000)
	index := buildSearchSparseHNSW(b, inputs, 16, 120)
	query := inputs[4321].vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: 100}
	b.ResetTimer()
	for b.Loop() {
		{
			_, err := index.SearchSparseHNSW(context.Background(), query, options)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func buildSearchSparseHNSW(t testing.TB, inputs []sparseHNSWInput, m, efConstruction int) *SparseHNSWIndex {
	return buildSearchSparseHNSWWithWorkers(t, inputs, m, efConstruction, 1)
}

func buildSearchSparseHNSWWithWorkers(t testing.TB, inputs []sparseHNSWInput, m, efConstruction, workers int) *SparseHNSWIndex {
	t.Helper()
	options := DefaultSparseHNSWBuildOptions()
	options.M = m
	options.EFConstruction = efConstruction
	options.Seed = 0x123456789abcdef
	builder, err := NewSparseHNSWBuilder(options)
	require.NoError(t, err)
	for _, input := range inputs {
		require.NoError(t, builder.AddSparse(context.Background(), input.key, input.vector))
	}
	index, err := builder.BuildWithWorkers(context.Background(), workers)
	require.NoError(t, err)
	return index
}

func exactSparseResults(ctx context.Context, query SparseVector, inputs []sparseHNSWInput, k int) ([]Result, error) {
	keys := make([]uint64, len(inputs))
	offsets := make([]int, len(inputs)+1)
	var indices []uint32
	var values []float32
	for position, input := range inputs {
		keys[position] = input.key
		indices = append(indices, input.vector.Indices...)
		values = append(values, input.vector.Values...)
		offsets[position+1] = len(indices)
	}
	return topKSparseCandidatesWithOptions(ctx, query, SearchOptions{TopK: k}, keys, offsets, indices, values, false)
}

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

func TestSparseHNSWPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedSparseHNSWIndex(t, 160)
	path := filepath.Join(t.TempDir(), "vectors.shnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, opened, index)
	query := SparseVector{Indices: []uint32{3, 107, 211}, Values: []float32{1, 2, 3}}
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 25}, EF: 80}
	want, err := index.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)

	got, err := opened.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)
	require.Equal(t, want, got)

	replacement := persistedSparseHNSWIndex(t, 40)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err = OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, opened, replacement)
}

func TestSparseHNSWPersistenceLargeGraphSearch(t *testing.T) {
	index := persistedSparseHNSWIndex(t, DefaultHNSWBruteForceThreshold+100)
	path := filepath.Join(t.TempDir(), "large.shnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	query := sparseHNSWBuildInputs(index.Len())[713].vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EF: 120}
	want, err := index.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)

	got, err := opened.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestSparseHNSWPersistenceEmpty(t *testing.T) {
	t.Parallel()
	builder, err := NewSparseHNSWBuilder(DefaultSparseHNSWBuildOptions())
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "empty.shnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, opened, index)
}

func TestSparseHNSWPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedSparseHNSWIndex(t, 32)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.shnsw")
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

	after, _ := os.ReadFile(path)
	require.True(t, slices.Equal(after, original),
		"canceled replacement changed published file")
	{
		_, err := OpenSparseHNSWIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenSparseHNSWIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}
	{
		_, err := OpenSparseHNSWIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}
	{
		_, err := OpenSparseHNSWIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	var nilIndex *SparseHNSWIndex
	{
		err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil.shnsw"))
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}

	invalid, err := cloneSparseHNSWIndex(context.Background(), index)
	require.NoError(t, err)

	invalid.offsets[1] = len(invalid.indices) + 1
	{
		err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.shnsw"))
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}
}

func TestSparseHNSWPersistenceDetectsCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(t, 32))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, sparseHNSWHeaderSize - 1, sparseHNSWHeaderSize, len(valid) - 1} {
		{
			_, err := decodeSparseHNSWIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeSparseHNSWIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], sparseHNSWFileVersion+1)
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedSparseHNSWVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[48] ^= 1
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badHeader)
		require.ErrorIs(t, err, ErrSparseHNSWChecksumMismatch)
	}

	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badPayload)
		require.ErrorIs(t, err, ErrSparseHNSWChecksumMismatch)
	}
}

func TestSparseHNSWPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(t, 32))
	require.NoError(t, err)

	records := parseSparseHNSWRecordOffsets(t, valid)
	require.True(t, len(records) >= 2,
		"fixture lacks sparse elements or graph edges")
	require.True(t, len(records[0].coordinates) >= 2,
		"fixture lacks sparse elements or graph edges")
	require.True(t, records[0].neighbor >= 0,
		"fixture lacks sparse elements or graph edges")

	tests := []struct {
		name   string
		mutate func([]byte)
	}{
		{"duplicate key", func(data []byte) { copy(data[records[1].key:records[1].key+8], data[records[0].key:records[0].key+8]) }},
		{"coordinate order", func(data []byte) {
			copy(data[records[0].coordinates[1]:records[0].coordinates[1]+4], data[records[0].coordinates[0]:records[0].coordinates[0]+4])
		}},
		{"non-finite value", func(data []byte) {
			binary.LittleEndian.PutUint32(data[records[0].coordinates[0]+4:records[0].coordinates[0]+8], math.Float32bits(float32(math.NaN())))
		}},
		{"invalid level", func(data []byte) {
			binary.LittleEndian.PutUint32(data[records[0].level:records[0].level+4], MaxHNSWLevel+1)
		}},
		{"neighbor out of range", func(data []byte) {
			binary.LittleEndian.PutUint32(data[records[0].neighbor:records[0].neighbor+4], uint32(len(records)))
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			rechecksumSparseHNSW(encoded)
			{
				_, err := decodeSparseHNSWIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
			}
		})
	}
	headerTests := []func([]byte){
		func(data []byte) { binary.LittleEndian.PutUint32(data[48:52], 0) },
		func(data []byte) { binary.LittleEndian.PutUint64(data[60:68], uint64(len(records))) },
		func(data []byte) { data[92] = 1 },
		func(data []byte) { binary.LittleEndian.PutUint64(data[16:24], uint64(len(data)+1)) },
	}
	for _, mutate := range headerTests {
		encoded := slices.Clone(valid)
		mutate(encoded)
		binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
		{
			_, err := decodeSparseHNSWIndex(context.Background(), encoded)
			require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
		}
	}
}

func FuzzDecodeSparseHNSWIndex(f *testing.F) {
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(f, 12))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("ZVSPHNSW"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeSparseHNSWIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateSparseHNSWIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func persistedSparseHNSWIndex(t testing.TB, count int) *SparseHNSWIndex {
	t.Helper()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x123456789abcdef0
	return buildSparseHNSW(t, options, sparseHNSWBuildInputs(count))
}

func assertSameSparseHNSWIndex(t testing.TB, got, want *SparseHNSWIndex) {
	t.Helper()
	require.Equal(t, want.Metric(), got.Metric(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.Len(), got.Len(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.BuildOptions(), got.BuildOptions(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.entryPoint, got.entryPoint,

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.MaxLevel(), got.MaxLevel(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.levelRNGState, got.levelRNGState,

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.keys, want.keys),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.offsets, want.offsets),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.indices, want.indices),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.values, want.values),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.levels, want.levels),
		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.neighbors, got.neighbors,
		"reopened sparse HNSW metadata differs")
}

type sparseHNSWRecordOffset struct {
	key         int
	level       int
	coordinates []int
	neighbor    int
}

func parseSparseHNSWRecordOffsets(t testing.TB, encoded []byte) []sparseHNSWRecordOffset {
	t.Helper()
	count := int(binary.LittleEndian.Uint64(encoded[32:40]))
	offset := sparseHNSWHeaderSize
	records := make([]sparseHNSWRecordOffset, count)
	for position := range records {
		records[position].neighbor = -1
		records[position].key = offset
		offset += 8
		records[position].level = offset
		level := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		offset += 4
		nonzero := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		offset += 4
		for range nonzero {
			records[position].coordinates = append(records[position].coordinates, offset)
			offset += sparseHNSWElementBytes
		}
		for currentLevel := 0; currentLevel <= level; currentLevel++ {
			degree := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
			offset += 4
			if currentLevel == 0 && degree != 0 {
				records[position].neighbor = offset
			}
			offset += degree * 4
		}
	}
	require.Len(t, encoded, offset)

	return records
}

func rechecksumSparseHNSW(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[88:92], ailego.CRC32C(encoded[sparseHNSWHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
