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

func TestVamanaBuildOptionsGraphDeterminismAndOwnership(t *testing.T) {
	defaults := DefaultVamanaBuildOptions(MetricCosine)
	require.True(t, defaults.MaxDegree == 64)
	require.True(t, defaults.SearchListSize == 100)
	require.True(t, defaults.Alpha == 1.2)
	require.True(t, defaults.MaxOcclusionSize == 750)
	require.Equal(t, MetricCosine, defaults.Metric)

	valid := DefaultVamanaBuildOptions(MetricL2)
	invalid := []VamanaBuildOptions{
		{},
		func() VamanaBuildOptions { value := valid; value.MaxDegree = 0; return value }(),
		func() VamanaBuildOptions { value := valid; value.MaxDegree = MaxVamanaDegree + 1; return value }(),
		func() VamanaBuildOptions { value := valid; value.SearchListSize = value.MaxDegree - 1; return value }(),
		func() VamanaBuildOptions { value := valid; value.Alpha = float32(math.NaN()); return value }(),
		func() VamanaBuildOptions { value := valid; value.MaxOcclusionSize = 0; return value }(),
	}
	for _, options := range invalid {
		{
			_, err := NewVamanaBuilder(3, options)
			require.ErrorIs(t, err, ErrInvalidVamanaOptions)
		}
	}
	{
		_, err := NewVamanaBuilder(0, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}

	inputs := hnswBuildInputs(160)
	build := func(saturate bool) *VamanaIndex {
		options := DefaultVamanaBuildOptions(MetricL2)
		options.MaxDegree = 8
		options.SearchListSize = 40
		options.MaxOcclusionSize = 60
		options.SaturateGraph = saturate
		return buildVamana(t, inputs, options)
	}
	first, second := build(false), build(false)
	require.Equal(t, second.neighbors, first.neighbors,

		"Vamana build is not deterministic")
	require.Equal(t, second.entryPoint, first.entryPoint,

		"Vamana build is not deterministic")
	require.Equal(t, second.neighborDistances, first.neighborDistances,
		"Vamana build is not deterministic")
	require.True(t, first.Dimension() == 3,
		"Vamana metadata differs")
	require.Equal(t, MetricL2, first.Metric(),
		"Vamana metadata differs")
	require.Len(t, inputs, first.Len(),
		"Vamana metadata differs")
	require.True(t, first.BuildOptions().MaxDegree == 8,
		"Vamana metadata differs")

	entry, found := first.EntryPoint()
	require.True(t, found,
		"Vamana entry point missing")
	require.Equal(t, first.keys[first.entryPoint], entry,
		"Vamana entry point missing")

	assertVamanaGraphInvariants(t, first)
	for _, adjacent := range build(true).neighbors {
		require.False(t, len(adjacent) == 0,
			"saturated graph retained an empty neighbor list")
	}
	original, _ := first.Vector(inputs[0].Key)
	inputs[0].Vector[0]++
	again, _ := first.Vector(inputs[0].Key)
	require.Equal(t, again[0], original[0],
		"builder did not own input vector")

	original[0]++
	again, _ = first.Vector(inputs[0].Key)
	require.NotEqual(t, again[0], original[0],
		"Vector exposed mutable storage")
}

func TestVamanaSearchMetricsFilterRadiusAndRecall(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		inputs := hnswBuildInputs(180)
		options := DefaultVamanaBuildOptions(metric)
		options.MaxDegree = 8
		options.SearchListSize = 40
		options.MaxOcclusionSize = 60
		index := buildVamana(t, inputs, options)
		query := slices.Clone(inputs[77].Vector)
		search := VamanaSearchOptions{
			SearchOptions: SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }},
			EFSearch:      80,
		}
		got, err := index.SearchVamana(context.Background(), query, search)
		require.NoError(t, err)

		want, err := topKCandidatesWithOptions(context.Background(), metric, query, search.SearchOptions, len(inputs), func(position int) Candidate {
			return inputs[position]
		}, true)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
	inputs := hnswBuildInputs(80)
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 8, 32
	index := buildVamana(t, inputs, options)
	target := inputs[17]
	bounded, err := index.SearchVamana(context.Background(), target.Vector, VamanaSearchOptions{
		SearchOptions: SearchOptions{TopK: 5, Radius: .01, Filter: func(key uint64) bool { return key == target.Key }},
		EFSearch:      40,
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: target.Key, Score: 0}}, bounded)

	inputs = hnswRaBitQCandidates(DefaultVamanaBruteForceThreshold+120, 64)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		options := DefaultVamanaBuildOptions(metric)
		options.MaxDegree = 16
		options.SearchListSize = 80
		options.MaxOcclusionSize = 160
		index := buildVamana(t, inputs, options)
		var matches, total int
		for queryIndex := 0; queryIndex < 10; queryIndex++ {
			query := inputs[(queryIndex*79+17)%len(inputs)].Vector
			truth, err := TopK(context.Background(), metric, query, inputs, 10)
			require.NoError(t, err)

			got, err := index.SearchVamana(context.Background(), query, VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EFSearch: 100})
			require.NoError(t, err)

			if metric == MetricL2 && queryIndex == 0 {
				prefetched, err := index.SearchVamana(context.Background(), query, VamanaSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, EFSearch: 100,
					PrefetchOffset: math.MaxUint32, PrefetchLines: math.MaxUint32,
				})
				require.NoError(t, err)
				require.Equal(t, got, prefetched)
			}
			matches += resultOverlap(got, truth)
			total += len(truth)
		}
		{
			recall := float64(matches) / float64(total)
			require.True(t, recall >= .80)
		}
	}
}

func TestVamanaRobustPrunePinnedGeometry(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize = 3, 4, 4
	index := &VamanaIndex{
		dimension: 2, options: options,
		keys: []uint64{1, 2, 3, 4}, vectors: []float32{0, 0, 1, 0, 1.1, 0, 0, 2},
	}
	candidates := []vamanaDistanceNode{{position: 1, distance: 1}, {position: 2, distance: 1.21}, {position: 3, distance: 4}}
	selected, err := index.robustPrune(context.Background(), 0, candidates)
	require.NoError(t, err)
	require.Equal(t, []vamanaDistanceNode{{position: 1, distance: 1}, {position: 3, distance: 4}}, selected)

	index.options.SaturateGraph = true
	selected, err = index.robustPrune(context.Background(), 0, candidates)
	require.NoError(t, err)
	require.Equal(t, []vamanaDistanceNode{{position: 1, distance: 1}, {position: 3, distance: 4}, {position: 2, distance: 1.21}}, selected)
}

func TestVamanaEmptyIncrementalAndValidation(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree = 4
	options.SearchListSize = 12
	builder, err := NewVamanaBuilder(3, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Len() == 0,
		"empty Vamana index is not empty")
	{
		_, found := index.EntryPoint()
		require.False(t, found,
			"empty Vamana index has entry point")
	}

	validSearch := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EFSearch: 10}
	{
		got, err := index.SearchVamana(context.Background(), []float32{0, 0, 0}, validSearch)
		require.NoError(t, err)
		require.Len(t, got, 0)
	}

	vector := []float32{1, 2, 3}
	{
		err := index.Add(context.Background(), 7, vector)
		require.NoError(t, err)
	}
	require.True(t, index.Len() == 1)
	{
		err := index.Add(context.Background(), 7, vector)
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		err := index.Add(context.Background(), 8, vector[:2])
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		_, err := index.SearchVamana(nil, vector, validSearch)
		require.Error(t, err,
			"nil search context succeeded")
	}

	validSearch.EFSearch = 0
	{
		_, err := index.SearchVamana(context.Background(), vector, validSearch)
		require.ErrorIs(t, err, ErrInvalidVamanaEF)
	}
}

func buildVamana(t testing.TB, inputs []Candidate, options VamanaBuildOptions) *VamanaIndex {
	t.Helper()
	dimension := 3
	if len(inputs) != 0 {
		dimension = len(inputs[0].Vector)
	}
	builder, err := NewVamanaBuilder(dimension, options)
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

func assertVamanaGraphInvariants(t testing.TB, index *VamanaIndex) {
	t.Helper()
	{
		err := validateVamanaIndex(context.Background(), index)
		require.NoError(t, err)
	}

	for position, adjacent := range index.neighbors {
		seen := make(map[int]struct{}, len(adjacent))
		for _, neighbor := range adjacent {
			require.NotEqual(t, position, neighbor)
			{
				_, found := seen[neighbor]
				require.False(t, found)
			}

			seen[neighbor] = struct{}{}
		}
	}
}
