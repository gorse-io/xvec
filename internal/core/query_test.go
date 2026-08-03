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
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestSearchOptionsValidation(t *testing.T) {
	if err := (SearchOptions{TopK: 1}).Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (SearchOptions{TopK: 0}).Validate(); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("zero top-k error = %v", err)
	}
	for _, radius := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		if err := (SearchOptions{TopK: 1, Radius: radius}).Validate(); !errors.Is(err, ErrInvalidRadius) {
			t.Fatalf("radius %v error = %v", radius, err)
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
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(results, testCase.want) {
				t.Fatalf("results = %#v, want %#v", results, testCase.want)
			}
		})
	}

	index := denseIndexFromCandidates(t, MetricIP, exactCandidates)
	results, err := index.SearchWithOptions(context.Background(), []float32{1, 0}, SearchOptions{
		TopK: 2,
		Filter: func(key uint64) bool {
			return key != 5 && key != 10 && key != 30
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: 20, Score: 0}, {Key: 40, Score: -1}}) {
		t.Fatalf("filtered results = %#v", results)
	}
	results, err = index.SearchWithOptions(context.Background(), []float32{1, 0}, SearchOptions{TopK: 5})
	if err != nil || len(results) != 5 {
		t.Fatalf("zero radius should be disabled: %#v, %v", results, err)
	}
	if _, err := index.SearchWithOptions(context.Background(), []float32{1, 0}, SearchOptions{}); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("invalid options error = %v", err)
	}
}

func TestSparseSearchRadiusAndCandidateFilter(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	for key, value := range []float32{1, 2, 3, 4} {
		if err := index.AddSparse(context.Background(), uint64(key), SparseVector{Indices: []uint32{1}, Values: []float32{value}}); err != nil {
			t.Fatal(err)
		}
	}
	results, err := index.SearchSparseWithOptions(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, SearchOptions{
		TopK: 10, Radius: 2,
		Filter: func(key uint64) bool { return key != 3 },
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: 2, Score: 3}, {Key: 1, Score: 2}}) {
		t.Fatalf("sparse filtered radius results = %#v", results)
	}
	if _, err := index.SearchSparseWithOptions(context.Background(), SparseVector{}, SearchOptions{TopK: 1, Radius: -1}); !errors.Is(err, ErrInvalidRadius) {
		t.Fatalf("invalid sparse radius = %v", err)
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
	if err != nil {
		t.Fatal(err)
	}
	want := []Result{{Key: 20, Score: 3}, {Key: 5, Score: 2}, {Key: 10, Score: 2}}
	if !reflect.DeepEqual(results, want) {
		t.Fatalf("merged = %#v, want %#v", results, want)
	}
	reversed, err := QueryDense(context.Background(), MetricIP, []DenseQuerySearcher{second, first}, []float32{1, 0}, SearchOptions{TopK: 3}, 1)
	if err != nil || !reflect.DeepEqual(reversed, want) {
		t.Fatalf("reversed = %#v, %v", reversed, err)
	}
	empty, err := QueryDense(context.Background(), MetricIP, nil, []float32{1, 0}, SearchOptions{TopK: 3}, 0)
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("empty query = %#v, %v", empty, err)
	}
}

func TestQuerySparseMergesSegments(t *testing.T) {
	first, _ := NewSparseFlatIndex(MetricIP)
	second, _ := NewSparseFlatIndex(MetricIP)
	_ = first.AddSparse(context.Background(), 1, SparseVector{Indices: []uint32{1}, Values: []float32{1}})
	_ = first.AddSparse(context.Background(), 4, SparseVector{Indices: []uint32{1}, Values: []float32{4}})
	_ = second.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{2}})
	_ = second.AddSparse(context.Background(), 3, SparseVector{Indices: []uint32{1}, Values: []float32{3}})
	results, err := QuerySparse(context.Background(), []SparseQuerySearcher{first, second}, SparseVector{Indices: []uint32{1}, Values: []float32{1}}, SearchOptions{TopK: 3}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: 4, Score: 4}, {Key: 3, Score: 3}, {Key: 2, Score: 2}}) {
		t.Fatalf("merged sparse = %#v", results)
	}
}

func TestSegmentQueryValidationAndCancellation(t *testing.T) {
	index := denseIndexFromCandidates(t, MetricL2, exactCandidates[:1])
	if _, err := QueryDense(context.Background(), MetricIP, []DenseQuerySearcher{index}, []float32{1, 0}, SearchOptions{TopK: 1}, 1); err == nil {
		t.Fatal("mismatched dense metric succeeded")
	}
	if _, err := QueryDense(context.Background(), Metric(99), nil, []float32{1}, SearchOptions{TopK: 1}, 1); err == nil {
		t.Fatal("invalid query metric succeeded")
	}
	if _, err := QueryDense(context.Background(), MetricL2, []DenseQuerySearcher{nil}, []float32{1, 0}, SearchOptions{TopK: 1}, 1); err == nil {
		t.Fatal("nil dense searcher succeeded")
	}
	if _, err := QuerySparse(context.Background(), []SparseQuerySearcher{nil}, SparseVector{}, SearchOptions{TopK: 1}, 1); err == nil {
		t.Fatal("nil sparse searcher succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := QueryDense(canceled, MetricL2, []DenseQuerySearcher{index}, []float32{1, 0}, SearchOptions{TopK: 1}, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled dense query = %v", err)
	}
	if _, err := QuerySparse(canceled, nil, SparseVector{}, SearchOptions{TopK: 1}, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled sparse query = %v", err)
	}
}

func TestMergeSearchResults(t *testing.T) {
	batches := [][]Result{
		{{Key: 3, Score: 1}, {Key: 1, Score: 2}},
		{{Key: 2, Score: 2}, {Key: 4, Score: 0}},
	}
	if got := MergeSearchResults(MetricIP, 3, batches...); !reflect.DeepEqual(got, []Result{{Key: 1, Score: 2}, {Key: 2, Score: 2}, {Key: 3, Score: 1}}) {
		t.Fatalf("IP merge = %#v", got)
	}
	if got := MergeSearchResults(MetricL2, 2, batches...); !reflect.DeepEqual(got, []Result{{Key: 4, Score: 0}, {Key: 3, Score: 1}}) {
		t.Fatalf("L2 merge = %#v", got)
	}
	if got := MergeSearchResults(MetricL2, 0, batches...); got == nil || len(got) != 0 {
		t.Fatalf("zero merge = %#v", got)
	}
}

func denseIndexFromCandidates(t *testing.T, metric Metric, candidates []Candidate) *DenseFlatIndex {
	t.Helper()
	index, err := NewDenseFlatIndex(2, metric)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range candidates {
		if err := index.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			t.Fatal(err)
		}
	}
	return index
}
