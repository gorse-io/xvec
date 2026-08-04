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
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
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
