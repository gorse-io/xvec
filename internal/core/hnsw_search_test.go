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
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

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
	t.Helper()
	options := DefaultHNSWBuildOptions(metric)
	options.M = m
	options.EFConstruction = efConstruction
	options.Seed = 0x123456789abcdef
	builder, err := NewHNSWBuilder(len(inputs[0].Vector), options)
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
