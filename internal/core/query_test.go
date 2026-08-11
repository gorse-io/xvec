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
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSearchOptionsValidation(t *testing.T) {
	{
		err := (SearchOptions{TopK: 1}).Validate()
		require.NoError(t, err)
	}
	{
		err := (SearchOptions{TopK: 0}).Validate()
		require.ErrorIs(t, err, ErrInvalidTopK)
	}

	for _, radius := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		{
			err := (SearchOptions{TopK: 1, Radius: radius}).Validate()
			require.ErrorIs(t, err, ErrInvalidRadius)
		}
	}
}

func TestDenseSearchRadiusAndCandidateFilter(t *testing.T) {
	tests := []struct {
		name   string
		metric Metric
		radius float32
		want   []Result
	}{
		{
			name: "L2 maximum distance", metric: MetricL2, radius: 1,
			want: []Result{{Key: 30, Score: 0}, {Key: 5, Score: 1}, {Key: 10, Score: 1}},
		},
		{
			name: "IP minimum similarity", metric: MetricIP, radius: 2,
			want: []Result{{Key: 5, Score: 2}, {Key: 10, Score: 2}},
		},
		{
			name: "cosine maximum distance", metric: MetricCosine, radius: 0.5,
			want: []Result{{Key: 5, Score: 0}, {Key: 10, Score: 0}, {Key: 30, Score: 0}},
		},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			index := denseIndexFromCandidates(t, testCase.metric, exactCandidates)
			results, err := index.SearchWithOptions(context.Background(), []float32{1, 0}, SearchOptions{TopK: 10, Radius: testCase.radius})
			require.NoError(t, err)
			require.Equal(t, testCase.want, results)
		})
	}

	index := denseIndexFromCandidates(t, MetricIP, exactCandidates)
	results, err := index.SearchWithOptions(context.Background(), []float32{1, 0}, SearchOptions{
		TopK: 2,
		Filter: func(key uint64) bool {
			return key != 5 && key != 10 && key != 30
		},
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 20, Score: 0}, {Key: 40, Score: -1}}, results)

	results, err = index.SearchWithOptions(context.Background(), []float32{1, 0}, SearchOptions{TopK: 5})
	require.NoError(t, err)
	require.Len(t, results, 5)
	{
		_, err := index.SearchWithOptions(context.Background(), []float32{1, 0}, SearchOptions{})
		require.ErrorIs(t, err, ErrInvalidTopK)
	}
}

func TestSparseSearchRadiusAndCandidateFilter(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	require.NoError(t, err)

	for key, value := range []float32{1, 2, 3, 4} {
		{
			err := index.AddSparse(context.Background(), uint64(key), SparseVector{Indices: []uint32{1}, Values: []float32{value}})
			require.NoError(t, err)
		}
	}
	results, err := index.SearchSparseWithOptions(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, SearchOptions{
		TopK: 10, Radius: 2,
		Filter: func(key uint64) bool { return key != 3 },
	})
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 2, Score: 3}, {Key: 1, Score: 2}}, results)
	{
		_, err := index.SearchSparseWithOptions(context.Background(), SparseVector{}, SearchOptions{TopK: 1, Radius: -1})
		require.ErrorIs(t, err, ErrInvalidRadius)
	}
}

func TestQueryDenseMergesSegmentsDeterministically(t *testing.T) {
	first := denseIndexFromCandidates(t, MetricIP, []Candidate{
		{Key: 10, Vector: []float32{2, 0}},
		{Key: 30, Vector: []float32{1, 0}},
	})
	second := denseIndexFromCandidates(t, MetricIP, []Candidate{
		{Key: 5, Vector: []float32{2, 0}},
		{Key: 20, Vector: []float32{3, 0}},
	})
	results, err := QueryDense(context.Background(), MetricIP, []DenseQuerySearcher{first, second}, []float32{1, 0}, SearchOptions{TopK: 3}, 2)
	require.NoError(t, err)

	want := []Result{{Key: 20, Score: 3}, {Key: 5, Score: 2}, {Key: 10, Score: 2}}
	require.Equal(t, want, results)

	reversed, err := QueryDense(context.Background(), MetricIP, []DenseQuerySearcher{second, first}, []float32{1, 0}, SearchOptions{TopK: 3}, 1)
	require.NoError(t, err)
	require.Equal(t, want, reversed)

	empty, err := QueryDense(context.Background(), MetricIP, nil, []float32{1, 0}, SearchOptions{TopK: 3}, 0)
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
}

func TestQuerySparseMergesSegments(t *testing.T) {
	first, _ := NewSparseFlatIndex(MetricIP)
	second, _ := NewSparseFlatIndex(MetricIP)
	_ = first.AddSparse(context.Background(), 1, SparseVector{Indices: []uint32{1}, Values: []float32{1}})
	_ = first.AddSparse(context.Background(), 4, SparseVector{Indices: []uint32{1}, Values: []float32{4}})
	_ = second.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{2}})
	_ = second.AddSparse(context.Background(), 3, SparseVector{Indices: []uint32{1}, Values: []float32{3}})
	results, err := QuerySparse(context.Background(), []SparseQuerySearcher{first, second}, SparseVector{Indices: []uint32{1}, Values: []float32{1}}, SearchOptions{TopK: 3}, 2)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 4, Score: 4}, {Key: 3, Score: 3}, {Key: 2, Score: 2}}, results)
}

func TestSegmentQueryValidationAndCancellation(t *testing.T) {
	index := denseIndexFromCandidates(t, MetricL2, exactCandidates[:1])
	{
		_, err := QueryDense(context.Background(), MetricIP, []DenseQuerySearcher{index}, []float32{1, 0}, SearchOptions{TopK: 1}, 1)
		require.Error(t, err,
			"mismatched dense metric succeeded")
	}
	{
		_, err := QueryDense(context.Background(), Metric(99), nil, []float32{1}, SearchOptions{TopK: 1}, 1)
		require.Error(t, err,
			"invalid query metric succeeded")
	}
	{
		_, err := QueryDense(context.Background(), MetricL2, []DenseQuerySearcher{nil}, []float32{1, 0}, SearchOptions{TopK: 1}, 1)
		require.Error(t, err,
			"nil dense searcher succeeded")
	}
	{
		_, err := QuerySparse(context.Background(), []SparseQuerySearcher{nil}, SparseVector{}, SearchOptions{TopK: 1}, 1)
		require.Error(t, err,
			"nil sparse searcher succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := QueryDense(canceled, MetricL2, []DenseQuerySearcher{index}, []float32{1, 0}, SearchOptions{TopK: 1}, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := QuerySparse(canceled, nil, SparseVector{}, SearchOptions{TopK: 1}, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestMergeSearchResults(t *testing.T) {
	batches := [][]Result{
		{{Key: 3, Score: 1}, {Key: 1, Score: 2}},
		{{Key: 2, Score: 2}, {Key: 4, Score: 0}},
	}
	{
		got := MergeSearchResults(MetricIP, 3, batches...)
		require.Equal(t, []Result{{Key: 1, Score: 2}, {Key: 2, Score: 2}, {Key: 3, Score: 1}}, got)
	}
	{
		got := MergeSearchResults(MetricL2, 2, batches...)
		require.Equal(t, []Result{{Key: 4, Score: 0}, {Key: 3, Score: 1}}, got)
	}
	{
		got := MergeSearchResults(MetricL2, 0, batches...)
		require.NotNil(t, got)
		require.Len(t, got, 0)
	}
}

func denseIndexFromCandidates(t *testing.T, metric Metric, candidates []Candidate) *DenseFlatIndex {
	t.Helper()
	index, err := NewDenseFlatIndex(2, metric)
	require.NoError(t, err)

	for _, candidate := range candidates {
		{
			err := index.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	return index
}
