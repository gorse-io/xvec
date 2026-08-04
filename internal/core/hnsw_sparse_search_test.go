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
	t.Helper()
	options := DefaultSparseHNSWBuildOptions()
	options.M = m
	options.EFConstruction = efConstruction
	options.Seed = 0x123456789abcdef
	return buildSparseHNSW(t, options, inputs)
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
