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
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestSparseFlatExactSearch(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	vectors := []struct {
		key    uint64
		vector SparseVector
	}{
		{30, SparseVector{Indices: []uint32{1, 4}, Values: []float32{1, 1}}},
		{10, SparseVector{Indices: []uint32{1, 3}, Values: []float32{2, 3}}},
		{5, SparseVector{Indices: []uint32{1, 9}, Values: []float32{2, 8}}},
		{20, SparseVector{Indices: []uint32{2}, Values: []float32{7}}},
	}
	for _, item := range vectors {
		if err := index.AddSparse(context.Background(), item.key, item.vector); err != nil {
			t.Fatal(err)
		}
	}
	results, err := index.SearchSparse(context.Background(), SparseVector{
		Indices: []uint32{1, 3}, Values: []float32{1, 1},
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []Result{{Key: 10, Score: 5}, {Key: 5, Score: 2}, {Key: 30, Score: 1}}
	if !reflect.DeepEqual(results, want) {
		t.Fatalf("results = %#v, want %#v", results, want)
	}
	emptyQuery, err := index.SearchSparse(context.Background(), SparseVector{}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(emptyQuery, []Result{{Key: 5}, {Key: 10}, {Key: 20}, {Key: 30}}) {
		t.Fatalf("empty query results = %#v", emptyQuery)
	}
}

func TestSparseFlatProviderOwnsInput(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	vector := SparseVector{Indices: []uint32{1, 3}, Values: []float32{2, 4}}
	if err := index.AddSparse(context.Background(), 7, vector); err != nil {
		t.Fatal(err)
	}
	vector.Indices[0], vector.Values[0] = 99, 99
	stored, found := index.SparseVector(7)
	if !found || !reflect.DeepEqual(stored, SparseVector{Indices: []uint32{1, 3}, Values: []float32{2, 4}}) {
		t.Fatalf("stored = %#v, %v", stored, found)
	}
	stored.Indices[0], stored.Values[0] = 88, 88
	again, found := index.SparseVector(7)
	if !found || again.Indices[0] != 1 || again.Values[0] != 2 {
		t.Fatalf("provider shares storage: %#v, %v", again, found)
	}
	if _, found := index.SparseVector(8); found {
		t.Fatal("missing sparse vector was found")
	}
}

func TestSparseFlatValidation(t *testing.T) {
	if _, err := NewSparseFlatIndex(MetricL2); err == nil {
		t.Fatal("sparse L2 succeeded")
	}
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		vector SparseVector
		want   error
	}{
		{"length", SparseVector{Indices: []uint32{1}, Values: nil}, ailego.ErrDimensionMismatch},
		{"order", SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{"duplicate", SparseVector{Indices: []uint32{1, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{"non-finite", SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.Inf(1))}}, ailego.ErrNonFiniteVector},
	}
	for _, testCase := range tests {
		if err := index.AddSparse(context.Background(), 1, testCase.vector); !errors.Is(err, testCase.want) {
			t.Fatalf("%s error = %v", testCase.name, err)
		}
	}
	if err := index.AddSparse(context.Background(), 1, SparseVector{}); err != nil {
		t.Fatal(err)
	}
	if err := index.AddSparse(context.Background(), 1, SparseVector{}); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate key error = %v", err)
	}
	if err := index.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{math.MaxFloat32}}); err != nil {
		t.Fatalf("finite vector with overflowing self product was rejected: %v", err)
	}
	if _, err := index.SearchSparse(context.Background(), SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 1}}, 1); !errors.Is(err, ailego.ErrInvalidSparseOrder) {
		t.Fatalf("query order error = %v", err)
	}
	if _, err := index.SearchSparse(context.Background(), SparseVector{}, -1); err == nil {
		t.Fatal("negative top-k succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := index.AddSparse(canceled, 2, SparseVector{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled add = %v", err)
	}
	if _, err := index.SearchSparse(canceled, SparseVector{}, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled search = %v", err)
	}
	var nilIndex *SparseFlatIndex
	if err := nilIndex.AddSparse(context.Background(), 1, SparseVector{}); err == nil {
		t.Fatal("nil add succeeded")
	}
	if _, err := nilIndex.SearchSparse(context.Background(), SparseVector{}, 1); err == nil {
		t.Fatal("nil search succeeded")
	}
}

func TestSparseFlatBuilderAndStreaming(t *testing.T) {
	builder, err := NewSparseFlatBuilder(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.AddSparse(context.Background(), 1, SparseVector{Indices: []uint32{1}, Values: []float32{1}}); err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.AddSparse(context.Background(), 2, SparseVector{}); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("closed builder add = %v", err)
	}
	if _, err := builder.Build(context.Background()); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("second build = %v", err)
	}
	if err := index.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{2}}); err != nil {
		t.Fatal(err)
	}
	results, err := index.SearchSparse(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, 2)
	if err != nil || !reflect.DeepEqual(results, []Result{{Key: 2, Score: 2}, {Key: 1, Score: 1}}) {
		t.Fatalf("built search = %#v, %v", results, err)
	}
	var nilBuilder *SparseFlatIndexBuilder
	if err := nilBuilder.AddSparse(context.Background(), 1, SparseVector{}); err == nil {
		t.Fatal("nil builder add succeeded")
	}
	if _, err := nilBuilder.Build(context.Background()); err == nil {
		t.Fatal("nil builder build succeeded")
	}
}

func TestSparseFlatConcurrentStreamingAndSearch(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	const count = 128
	var wait sync.WaitGroup
	for key := range count {
		wait.Add(1)
		go func() {
			defer wait.Done()
			vector := SparseVector{Indices: []uint32{uint32(key)}, Values: []float32{float32(key)}}
			if err := index.AddSparse(context.Background(), uint64(key), vector); err != nil {
				t.Errorf("add %d: %v", key, err)
			}
			if _, err := index.SearchSparse(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, 10); err != nil {
				t.Errorf("search: %v", err)
			}
		}()
	}
	wait.Wait()
	if index.Len() != count {
		t.Fatalf("len = %d, want %d", index.Len(), count)
	}
}

func BenchmarkSparseFlatSearch(b *testing.B) {
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		b.Fatal(err)
	}
	for key := range 10_000 {
		vector := SparseVector{
			Indices: []uint32{uint32(key % 100), uint32(100 + key%100)},
			Values:  []float32{1, 2},
		}
		if err := index.AddSparse(context.Background(), uint64(key), vector); err != nil {
			b.Fatal(err)
		}
	}
	query := SparseVector{Indices: []uint32{1, 101}, Values: []float32{1, 1}}
	b.ResetTimer()
	for range b.N {
		if _, err := index.SearchSparse(context.Background(), query, 10); err != nil {
			b.Fatal(err)
		}
	}
}
