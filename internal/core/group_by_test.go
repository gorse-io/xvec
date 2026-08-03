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
	"strconv"
	"testing"
)

func TestGroupByOptionsValidation(t *testing.T) {
	resolver := func(uint64) (string, bool) { return "group", true }
	if err := (GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: resolver}).Validate(); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		options GroupByOptions
		want    error
	}{
		{name: "group count", options: GroupByOptions{TopKPerGroup: 1, Resolve: resolver}, want: ErrInvalidGroupCount},
		{name: "per-group top-k", options: GroupByOptions{GroupCount: 1, Resolve: resolver}, want: ErrInvalidGroupTopK},
		{name: "negative radius", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Radius: -1, Resolve: resolver}, want: ErrInvalidRadius},
		{name: "NaN radius", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Radius: float32(math.NaN()), Resolve: resolver}, want: ErrInvalidRadius},
		{name: "infinite radius", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Radius: float32(math.Inf(1)), Resolve: resolver}, want: ErrInvalidRadius},
		{name: "resolver", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1}, want: ErrNilGroupResolver},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			if err := testCase.options.Validate(); !errors.Is(err, testCase.want) {
				t.Fatalf("Validate() = %v, want %v", err, testCase.want)
			}
		})
	}
}

func TestDenseFlatSearchGroups(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	for key := uint64(0); key < 12; key++ {
		if err := index.Add(context.Background(), key, []float32{float32(key), 1}); err != nil {
			t.Fatal(err)
		}
	}
	options := GroupByOptions{
		GroupCount:   3,
		TopKPerGroup: 2,
		Resolve: func(key uint64) (string, bool) {
			return strconv.FormatUint(key%3, 10), true
		},
	}
	got, err := index.SearchGroups(context.Background(), []float32{1, 0}, options)
	if err != nil {
		t.Fatal(err)
	}
	want := []GroupResult{
		{Value: "2", Results: []Result{{Key: 11, Score: 11}, {Key: 8, Score: 8}}},
		{Value: "1", Results: []Result{{Key: 10, Score: 10}, {Key: 7, Score: 7}}},
		{Value: "0", Results: []Result{{Key: 9, Score: 9}, {Key: 6, Score: 6}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("SearchGroups() = %#v, want %#v", got, want)
	}
}

func TestDenseFlatSearchGroupsFilterRadiusAndMissingValue(t *testing.T) {
	index, err := NewDenseFlatIndex(1, MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	for key := uint64(0); key < 12; key++ {
		if err := index.Add(context.Background(), key, []float32{float32(key)}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := index.SearchGroups(context.Background(), []float32{1}, GroupByOptions{
		GroupCount:   3,
		TopKPerGroup: 3,
		Radius:       8,
		Filter:       func(key uint64) bool { return key != 10 },
		Resolve: func(key uint64) (string, bool) {
			if key == 9 {
				return "", false
			}
			if key%3 == 2 {
				return "", true
			}
			return strconv.FormatUint(key%3, 10), true
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []GroupResult{{Value: "", Results: []Result{{Key: 11, Score: 11}, {Key: 8, Score: 8}}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("filtered SearchGroups() = %#v, want %#v", got, want)
	}
}

func TestDenseFlatSearchGroupsUsesMetricOrdering(t *testing.T) {
	index, err := NewDenseFlatIndex(1, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	for key, value := range []float32{0, 1, 2, 3} {
		if err := index.Add(context.Background(), uint64(key), []float32{value}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := index.SearchGroups(context.Background(), []float32{0}, GroupByOptions{
		GroupCount:   2,
		TopKPerGroup: 2,
		Resolve: func(key uint64) (string, bool) {
			if key%2 == 0 {
				return "even", true
			}
			return "odd", true
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []GroupResult{
		{Value: "even", Results: []Result{{Key: 0, Score: 0}, {Key: 2, Score: 4}}},
		{Value: "odd", Results: []Result{{Key: 1, Score: 1}, {Key: 3, Score: 9}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("L2 groups = %#v, want %#v", got, want)
	}
}

func TestSparseFlatSearchGroups(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	for key := uint64(0); key < 6; key++ {
		vector := SparseVector{Indices: []uint32{1}, Values: []float32{float32(key)}}
		if err := index.AddSparse(context.Background(), key, vector); err != nil {
			t.Fatal(err)
		}
	}
	got, err := index.SearchSparseGroups(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, GroupByOptions{
		GroupCount:   2,
		TopKPerGroup: 2,
		Resolve: func(key uint64) (string, bool) {
			return strconv.FormatUint(key%2, 10), true
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []GroupResult{
		{Value: "1", Results: []Result{{Key: 5, Score: 5}, {Key: 3, Score: 3}}},
		{Value: "0", Results: []Result{{Key: 4, Score: 4}, {Key: 2, Score: 2}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sparse groups = %#v, want %#v", got, want)
	}
}

func TestQueryDenseGroupsMergesBeforeTruncating(t *testing.T) {
	first, err := NewDenseFlatIndex(1, MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	second, _ := NewDenseFlatIndex(1, MetricIP)
	for key, value := range []float32{0, 1} {
		_ = first.Add(context.Background(), uint64(key), []float32{value})
	}
	for key, value := range []float32{10, 11} {
		_ = second.Add(context.Background(), uint64(key+2), []float32{value})
	}
	resolver := func(key uint64) (string, bool) {
		if key < 2 {
			return "low", true
		}
		return "high", true
	}
	options := GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: resolver}
	want := []GroupResult{{Value: "high", Results: []Result{{Key: 3, Score: 11}}}}
	got, err := QueryDenseGroups(context.Background(), MetricIP, []DenseGroupSearcher{first, second}, []float32{1}, options, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merged groups = %#v, want %#v", got, want)
	}
	reversed, err := QueryDenseGroups(context.Background(), MetricIP, []DenseGroupSearcher{second, first}, []float32{1}, options, 1)
	if err != nil || !reflect.DeepEqual(reversed, want) {
		t.Fatalf("reversed groups = %#v, %v", reversed, err)
	}
}

func TestQuerySparseGroupsAndEmptySegments(t *testing.T) {
	first, _ := NewSparseFlatIndex(MetricIP)
	second, _ := NewSparseFlatIndex(MetricIP)
	_ = first.AddSparse(context.Background(), 1, SparseVector{Indices: []uint32{1}, Values: []float32{1}})
	_ = second.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{2}})
	resolver := func(key uint64) (string, bool) { return strconv.FormatUint(key, 10), true }
	options := GroupByOptions{GroupCount: 2, TopKPerGroup: 1, Resolve: resolver}
	got, err := QuerySparseGroups(context.Background(), []SparseGroupSearcher{first, second}, SparseVector{Indices: []uint32{1}, Values: []float32{1}}, options, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []GroupResult{
		{Value: "2", Results: []Result{{Key: 2, Score: 2}}},
		{Value: "1", Results: []Result{{Key: 1, Score: 1}}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merged sparse groups = %#v, want %#v", got, want)
	}
	empty, err := QuerySparseGroups(context.Background(), nil, SparseVector{}, options, 0)
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("empty sparse groups = %#v, %v", empty, err)
	}
}

func TestGroupByQueryValidationAndCancellation(t *testing.T) {
	index, err := NewDenseFlatIndex(1, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	_ = index.Add(context.Background(), 1, []float32{1})
	resolver := func(uint64) (string, bool) { return "group", true }
	options := GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: resolver}
	if _, err := QueryDenseGroups(context.Background(), MetricIP, []DenseGroupSearcher{index}, []float32{1}, options, 1); err == nil {
		t.Fatal("mismatched dense group metric succeeded")
	}
	if _, err := QueryDenseGroups(context.Background(), Metric(99), nil, []float32{1}, options, 1); err == nil {
		t.Fatal("invalid dense group metric succeeded")
	}
	if _, err := QueryDenseGroups(context.Background(), MetricL2, []DenseGroupSearcher{nil}, []float32{1}, options, 1); err == nil {
		t.Fatal("nil dense group searcher succeeded")
	}
	if _, err := QuerySparseGroups(context.Background(), []SparseGroupSearcher{nil}, SparseVector{}, options, 1); err == nil {
		t.Fatal("nil sparse group searcher succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := index.SearchGroups(canceled, []float32{1}, options); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled local dense group query = %v", err)
	}
	if _, err := QueryDenseGroups(canceled, MetricL2, []DenseGroupSearcher{index}, []float32{1}, options, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled dense group query = %v", err)
	}
}

func TestMergeGroupResultsRebuildsPerGroupTopK(t *testing.T) {
	batches := [][]GroupResult{
		{{Value: "a", Results: []Result{{Key: 3, Score: 3}, {Key: 1, Score: 1}}}},
		{{Value: "b", Results: []Result{{Key: 4, Score: 4}}}, {Value: "a", Results: []Result{{Key: 5, Score: 5}}}},
	}
	want := []GroupResult{
		{Value: "a", Results: []Result{{Key: 5, Score: 5}, {Key: 3, Score: 3}}},
		{Value: "b", Results: []Result{{Key: 4, Score: 4}}},
	}
	if got := MergeGroupResults(MetricIP, 2, 2, batches...); !reflect.DeepEqual(got, want) {
		t.Fatalf("MergeGroupResults() = %#v, want %#v", got, want)
	}
	if got := MergeGroupResults(MetricIP, 0, 2, batches...); got == nil || len(got) != 0 {
		t.Fatalf("zero group count merge = %#v", got)
	}
}
