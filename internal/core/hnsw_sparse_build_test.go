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
	"math"
	"slices"
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
