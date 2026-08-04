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
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestOriginalSparseVectorRefiner(t *testing.T) {
	t.Parallel()
	provider, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	for key, vector := range map[uint64]SparseVector{
		10: {Indices: []uint32{0, 3}, Values: []float32{1, 4}},
		20: {Indices: []uint32{0, 2}, Values: []float32{2, 5}},
		30: {Indices: []uint32{1, 3}, Values: []float32{9, 1}},
	} {
		if err := provider.AddSparse(context.Background(), key, vector); err != nil {
			t.Fatal(err)
		}
	}
	refiner, err := NewOriginalSparseVectorRefiner(provider)
	if err != nil {
		t.Fatal(err)
	}
	query := SparseVector{Indices: []uint32{0, 3}, Values: []float32{2, 1}}
	results, err := refiner.RefineSparse(context.Background(), query,
		[]Result{{Key: 30, Score: 99}, {Key: 10}, {Key: 20}, {Key: 10, Score: -1}},
		SearchOptions{TopK: 3, Radius: 4, Filter: func(key uint64) bool { return key != 30 }},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []Result{{Key: 10, Score: 6}, {Key: 20, Score: 4}}) {
		t.Fatalf("results = %#v", results)
	}
}

func TestOriginalSparseVectorRefinerValidation(t *testing.T) {
	t.Parallel()
	if _, err := NewOriginalSparseVectorRefiner(nil); err == nil {
		t.Fatal("nil provider accepted")
	}
	provider, _ := NewSparseFlatIndex(MetricIP)
	refiner, _ := NewOriginalSparseVectorRefiner(provider)
	valid := SparseVector{Indices: []uint32{1}, Values: []float32{1}}
	if _, err := refiner.RefineSparse(nil, valid, nil, SearchOptions{TopK: 1}); err == nil {
		t.Fatal("nil context accepted")
	}
	invalid := SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.NaN())}}
	if _, err := refiner.RefineSparse(context.Background(), invalid, nil, SearchOptions{TopK: 1}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("invalid query = %v", err)
	}
	if _, err := refiner.RefineSparse(context.Background(), valid, []Result{{Key: 99}}, SearchOptions{TopK: 1}); !errors.Is(err, ErrMissingRefineVector) {
		t.Fatalf("missing vector = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := refiner.RefineSparse(canceled, valid, nil, SearchOptions{TopK: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled refine = %v", err)
	}
}

func TestRefinedSparseSearch(t *testing.T) {
	t.Parallel()
	provider, _ := NewSparseFlatIndex(MetricIP)
	for key, value := range map[uint64]float32{1: 1, 2: 2, 3: 3, 4: 4} {
		if err := provider.AddSparse(context.Background(), key, SparseVector{Indices: []uint32{0}, Values: []float32{value}}); err != nil {
			t.Fatal(err)
		}
	}
	base := &recordingSparseSearcher{
		metric:  MetricIP,
		results: []Result{{Key: 1, Score: 99}, {Key: 2, Score: 3}, {Key: 3, Score: 2}, {Key: 4, Score: 1}},
	}
	refiner, _ := NewOriginalSparseVectorRefiner(provider)
	results, err := RefinedSparseSearch(context.Background(), base, refiner,
		SparseVector{Indices: []uint32{0}, Values: []float32{1}},
		SearchOptions{TopK: 2, Radius: 2}, 2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []Result{{Key: 4, Score: 4}, {Key: 3, Score: 3}}) {
		t.Fatalf("results = %#v", results)
	}
	if base.options.TopK != 4 || base.options.Radius != 0 {
		t.Fatalf("base options = %#v", base.options)
	}
	if _, err := RefinedSparseSearch(context.Background(), nil, refiner, SparseVector{}, SearchOptions{TopK: 1}, 1); err == nil {
		t.Fatal("nil base accepted")
	}
}

type recordingSparseSearcher struct {
	metric  Metric
	results []Result
	options SearchOptions
}

func (s *recordingSparseSearcher) Metric() Metric { return s.metric }
func (s *recordingSparseSearcher) SearchSparseWithOptions(_ context.Context, _ SparseVector, options SearchOptions) ([]Result, error) {
	s.options = options
	return slices.Clone(s.results[:min(options.TopK, len(s.results))]), nil
}
