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

func TestOriginalVectorRefiner(t *testing.T) {
	t.Parallel()
	provider, err := NewDenseFlatIndex(2, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range []Candidate{
		{Key: 10, Vector: []float32{3, 0}},
		{Key: 20, Vector: []float32{1, 0}},
		{Key: 30, Vector: []float32{2, 0}},
	} {
		if err := provider.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			t.Fatal(err)
		}
	}
	refiner, err := NewOriginalVectorRefiner(provider, MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	approximate := []Result{{Key: 10, Score: 0}, {Key: 30, Score: 1}, {Key: 20, Score: 99}, {Key: 20, Score: -1}}
	results, err := refiner.Refine(context.Background(), []float32{0, 0}, approximate, SearchOptions{TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	want := []Result{{Key: 20, Score: 1}, {Key: 30, Score: 4}}
	if !slices.Equal(results, want) {
		t.Fatalf("results = %#v, want %#v", results, want)
	}
}

func TestOriginalVectorRefinerFilterAndRadius(t *testing.T) {
	t.Parallel()
	provider, _ := NewDenseFlatIndex(1, MetricIP)
	for key, value := range map[uint64]float32{1: 1, 2: 2, 3: 3} {
		if err := provider.Add(context.Background(), key, []float32{value}); err != nil {
			t.Fatal(err)
		}
	}
	refiner, _ := NewOriginalVectorRefiner(provider, MetricIP)
	results, err := refiner.Refine(context.Background(), []float32{1}, []Result{{Key: 1}, {Key: 2}, {Key: 3}}, SearchOptions{
		TopK: 3, Radius: 2, Filter: func(key uint64) bool { return key != 3 },
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []Result{{Key: 2, Score: 2}}) {
		t.Fatalf("results = %#v", results)
	}
}

func TestOriginalVectorRefinerValidation(t *testing.T) {
	t.Parallel()
	if _, err := NewOriginalVectorRefiner(nil, MetricL2); err == nil {
		t.Fatal("nil provider accepted")
	}
	provider, _ := NewDenseFlatIndex(2, MetricL2)
	refiner, _ := NewOriginalVectorRefiner(provider, MetricL2)
	if _, err := refiner.Refine(nil, []float32{0, 0}, nil, SearchOptions{TopK: 1}); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := refiner.Refine(context.Background(), []float32{0}, nil, SearchOptions{TopK: 1}); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err := refiner.Refine(context.Background(), []float32{0, float32(math.NaN())}, nil, SearchOptions{TopK: 1}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("query error = %v", err)
	}
	if _, err := refiner.Refine(context.Background(), []float32{0, 0}, []Result{{Key: 99}}, SearchOptions{TopK: 1}); !errors.Is(err, ErrMissingRefineVector) {
		t.Fatalf("missing vector error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := refiner.Refine(ctx, []float32{0, 0}, nil, SearchOptions{TopK: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
}

func TestRefinementCandidateCount(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		topK  int
		scale float32
		want  int
	}{
		{10, 10, 100},
		{3, 1.9, 5},
		{1, .1, 1},
	} {
		got, err := RefinementCandidateCount(test.topK, test.scale)
		if err != nil || got != test.want {
			t.Fatalf("count(%d, %g) = %d, %v, want %d", test.topK, test.scale, got, err, test.want)
		}
	}
	if _, err := RefinementCandidateCount(0, 1); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("top-k error = %v", err)
	}
	for _, scale := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32} {
		if _, err := RefinementCandidateCount(2, scale); !errors.Is(err, ErrInvalidRefinerScale) {
			t.Fatalf("scale %g error = %v", scale, err)
		}
	}
}

func TestRefinedSearch(t *testing.T) {
	t.Parallel()
	provider, _ := NewDenseFlatIndex(1, MetricL2)
	for key, value := range map[uint64]float32{1: 1, 2: 2, 3: 3, 4: 4} {
		if err := provider.Add(context.Background(), key, []float32{value}); err != nil {
			t.Fatal(err)
		}
	}
	base := &recordingDenseSearcher{
		metric:  MetricL2,
		results: []Result{{Key: 4, Score: 0}, {Key: 2, Score: 1}, {Key: 1, Score: 2}, {Key: 3, Score: 3}},
	}
	refiner, _ := NewOriginalVectorRefiner(provider, MetricL2)
	results, err := RefinedSearch(context.Background(), base, refiner, []float32{0}, SearchOptions{TopK: 2, Radius: 5}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []Result{{Key: 1, Score: 1}, {Key: 2, Score: 4}}) {
		t.Fatalf("results = %#v", results)
	}
	if base.options.TopK != 4 || base.options.Radius != 0 {
		t.Fatalf("base options = %#v", base.options)
	}
}

func TestRefinedSearchValidation(t *testing.T) {
	t.Parallel()
	provider, _ := NewDenseFlatIndex(1, MetricL2)
	refiner, _ := NewOriginalVectorRefiner(provider, MetricL2)
	base := &recordingDenseSearcher{metric: MetricIP}
	if _, err := RefinedSearch(context.Background(), base, refiner, []float32{0}, SearchOptions{TopK: 1}, 2); err == nil {
		t.Fatal("metric mismatch accepted")
	}
	if _, err := RefinedSearch(context.Background(), nil, refiner, []float32{0}, SearchOptions{TopK: 1}, 2); err == nil {
		t.Fatal("nil base accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := RefinedSearch(ctx, base, refiner, []float32{0}, SearchOptions{TopK: 1}, 2); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
}

type recordingDenseSearcher struct {
	metric  Metric
	results []Result
	options SearchOptions
}

func (s *recordingDenseSearcher) Metric() Metric { return s.metric }
func (s *recordingDenseSearcher) SearchWithOptions(_ context.Context, _ []float32, options SearchOptions) ([]Result, error) {
	s.options = options
	count := min(options.TopK, len(s.results))
	return slices.Clone(s.results[:count]), nil
}
