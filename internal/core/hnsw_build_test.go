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
	for b.Loop() {
		builder, err := NewHNSWBuilder(3, options)
		if err != nil {
			require.NoError(b, err)
		}

		for _, input := range inputs {
			{
				err := builder.Add(context.Background(), input.Key, input.Vector)
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
