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

func TestDenseFlatSearchMetrics(t *testing.T) {
	t.Parallel()
	tests := []struct {
		metric Metric
		want   []Result
	}{
		{MetricL2, []Result{{Key: 30, Score: 0}, {Key: 5, Score: 1}, {Key: 10, Score: 1}}},
		{MetricIP, []Result{{Key: 5, Score: 2}, {Key: 10, Score: 2}, {Key: 30, Score: 1}}},
		{MetricCosine, []Result{{Key: 5, Score: 0}, {Key: 10, Score: 0}, {Key: 30, Score: 0}}},
		{MetricMIPSL2, []Result{{Key: 30, Score: 0}, {Key: 5, Score: 1}, {Key: 10, Score: 1}}},
	}
	for _, testCase := range tests {
		index, err := NewDenseFlatIndex(2, testCase.metric)
		if err != nil {
			t.Fatal(err)
		}
		for _, candidate := range exactCandidates {
			if err := index.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
				t.Fatal(err)
			}
		}
		got, err := index.Search(context.Background(), []float32{1, 0}, 3)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, testCase.want) {
			t.Fatalf("metric %d results = %#v, want %#v", testCase.metric, got, testCase.want)
		}
	}
}

func TestDenseFlatProviderAndInputOwnership(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	vector := []float32{1, 2}
	if err := index.Add(context.Background(), 42, vector); err != nil {
		t.Fatal(err)
	}
	vector[0] = 99
	stored, found := index.Vector(42)
	if !found || !reflect.DeepEqual(stored, []float32{1, 2}) {
		t.Fatalf("stored vector = %#v, %v", stored, found)
	}
	stored[0] = 88
	again, found := index.Vector(42)
	if !found || !reflect.DeepEqual(again, []float32{1, 2}) {
		t.Fatalf("provider shares storage: %#v, %v", again, found)
	}
	if _, found := index.Vector(7); found {
		t.Fatal("missing key was found")
	}
	if index.Dimension() != 2 || index.Metric() != MetricIP || index.Len() != 1 {
		t.Fatalf("metadata = dimension %d, metric %d, len %d", index.Dimension(), index.Metric(), index.Len())
	}
}

func TestDenseFlatValidation(t *testing.T) {
	if _, err := NewDenseFlatIndex(0, MetricL2); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err := NewDenseFlatIndex(2, Metric(99)); err == nil {
		t.Fatal("invalid metric succeeded")
	}
	index, err := NewDenseFlatIndex(2, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	if err := index.Add(context.Background(), 1, []float32{1}); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("add dimension error = %v", err)
	}
	if err := index.Add(context.Background(), 1, []float32{1, float32(math.NaN())}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("non-finite error = %v", err)
	}
	if err := index.Add(context.Background(), 1, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := index.Add(context.Background(), 1, []float32{3, 4}); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	if _, err := index.Search(context.Background(), []float32{1}, 1); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("query dimension error = %v", err)
	}
	if _, err := index.Search(context.Background(), []float32{1, 2}, -1); err == nil {
		t.Fatal("negative top-k succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := index.Add(canceled, 2, []float32{1, 2}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled add = %v", err)
	}
	if _, err := index.Search(canceled, []float32{1, 2}, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled search = %v", err)
	}
	var nilIndex *DenseFlatIndex
	if err := nilIndex.Add(context.Background(), 1, []float32{1}); err == nil {
		t.Fatal("nil index add succeeded")
	}
	if _, err := nilIndex.Search(context.Background(), []float32{1}, 1); err == nil {
		t.Fatal("nil index search succeeded")
	}
}

func TestDenseFlatBuilderLifecycle(t *testing.T) {
	builder, err := NewDenseFlatBuilder(2, MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.Add(context.Background(), 2, []float32{2, 0}); err != nil {
		t.Fatal(err)
	}
	if err := builder.Add(context.Background(), 1, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.Add(context.Background(), 3, []float32{3, 0}); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("add after build = %v", err)
	}
	if _, err := builder.Build(context.Background()); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("second build = %v", err)
	}
	if err := index.Add(context.Background(), 3, []float32{3, 0}); err != nil {
		t.Fatalf("stream after build: %v", err)
	}
	results, err := index.Search(context.Background(), []float32{1, 0}, 3)
	if err != nil || !reflect.DeepEqual(results, []Result{{Key: 3, Score: 3}, {Key: 2, Score: 2}, {Key: 1, Score: 1}}) {
		t.Fatalf("built search = %#v, %v", results, err)
	}
	var nilBuilder *DenseFlatIndexBuilder
	if err := nilBuilder.Add(context.Background(), 1, []float32{1}); err == nil {
		t.Fatal("nil builder add succeeded")
	}
	if _, err := nilBuilder.Build(context.Background()); err == nil {
		t.Fatal("nil builder build succeeded")
	}
}

func TestDenseFlatConcurrentStreamingAndSearch(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	const count = 128
	var wait sync.WaitGroup
	for key := range count {
		wait.Add(1)
		go func() {
			defer wait.Done()
			value := float32(key)
			if err := index.Add(context.Background(), uint64(key), []float32{value, value}); err != nil {
				t.Errorf("add %d: %v", key, err)
			}
			if _, err := index.Search(context.Background(), []float32{0, 0}, 10); err != nil {
				t.Errorf("search: %v", err)
			}
		}()
	}
	wait.Wait()
	if index.Len() != count {
		t.Fatalf("len = %d, want %d", index.Len(), count)
	}
	results, err := index.Search(context.Background(), []float32{0, 0}, 3)
	if err != nil || !reflect.DeepEqual(results, []Result{{Key: 0, Score: 0}, {Key: 1, Score: 2}, {Key: 2, Score: 8}}) {
		t.Fatalf("final results = %#v, %v", results, err)
	}
}

func BenchmarkDenseFlatSearch(b *testing.B) {
	index, err := NewDenseFlatIndex(32, MetricL2)
	if err != nil {
		b.Fatal(err)
	}
	for key := range 10_000 {
		vector := make([]float32, 32)
		for dimension := range vector {
			vector[dimension] = float32(key+dimension) / 10_000
		}
		if err := index.Add(context.Background(), uint64(key), vector); err != nil {
			b.Fatal(err)
		}
	}
	query := make([]float32, 32)
	b.ResetTimer()
	for range b.N {
		if _, err := index.Search(context.Background(), query, 10); err != nil {
			b.Fatal(err)
		}
	}
}
