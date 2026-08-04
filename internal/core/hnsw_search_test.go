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

	"github.com/gorse-io/zvec/internal/ailego"
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
		if err != nil {
			t.Fatal(err)
		}
		want, err := topKCandidatesWithOptions(context.Background(), metric, query, options.SearchOptions, len(inputs), func(position int) Candidate {
			return inputs[position]
		}, true)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("metric %d HNSW = %#v, Flat = %#v", metric, got, want)
		}
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
			if err != nil {
				t.Fatalf("metric %d: %v", metric, err)
			}
			want, err := TopK(context.Background(), metric, query, inputs, 10)
			if err != nil {
				t.Fatal(err)
			}
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
		if recall < 0.80 {
			t.Fatalf("metric %d recall@10 = %.3f, want >= 0.80", metric, recall)
		}
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
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: target.Key, Score: 0}}) {
		t.Fatalf("filtered radius results = %#v", results)
	}
	results, err = index.SearchHNSW(context.Background(), target.Vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 25 {
		t.Fatalf("EF below TopK returned %d results, want 25", len(results))
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: 2, Score: 1}, {Key: 9, Score: 1}, {Key: 50, Score: 1}}) {
		t.Fatalf("ties = %#v", results)
	}
	results, err = index.Search(context.Background(), []float32{0}, 0)
	if err != nil || results == nil || len(results) != 0 {
		t.Fatalf("zero top-k = %#v, %v", results, err)
	}
	options := DefaultHNSWBuildOptions(MetricL2)
	builder, _ := NewHNSWBuilder(1, options)
	empty, _ := builder.Build(context.Background())
	results, err = empty.SearchHNSW(context.Background(), []float32{0}, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, EF: 1,
	})
	if err != nil || results == nil || len(results) != 0 {
		t.Fatalf("empty search = %#v, %v", results, err)
	}
}

func TestHNSWResultTieBreaksByKey(t *testing.T) {
	t.Parallel()
	index := &HNSWIndex{
		options: HNSWBuildOptions{Metric: MetricL2},
		keys:    []uint64{90, 3},
	}
	left := hnswScoredNode{position: 0, score: 1}
	right := hnswScoredNode{position: 1, score: 1}
	if index.hnswResultNodeBetter(left, right) || !index.hnswResultNodeBetter(right, left) {
		t.Fatal("equal HNSW result scores did not prefer the smaller document key")
	}
}

func TestHNSWSearchValidation(t *testing.T) {
	t.Parallel()
	index := buildSearchHNSW(t, MetricL2, hnswBuildInputs(4), 2, 4)
	valid := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 2}
	if _, err := index.SearchHNSW(nil, []float32{1, 2, 3}, valid); err == nil {
		t.Fatal("nil context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := index.SearchHNSW(canceled, []float32{1, 2, 3}, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	if _, err := index.SearchHNSW(context.Background(), []float32{1, 2}, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err := index.SearchHNSW(context.Background(), []float32{1, float32(math.NaN()), 3}, valid); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("finite error = %v", err)
	}
	invalidEF := valid
	invalidEF.EF = 0
	if _, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidEF); !errors.Is(err, ErrInvalidHNSWEF) {
		t.Fatalf("EF error = %v", err)
	}
	invalidEF.EF = MaxHNSWEFSearch + 1
	if _, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidEF); !errors.Is(err, ErrInvalidHNSWEF) {
		t.Fatalf("large EF error = %v", err)
	}
	valid.EF = MaxHNSWEFSearch
	if _, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, valid); err != nil {
		t.Fatalf("maximum EF error = %v", err)
	}
	invalidTopK := valid
	invalidTopK.TopK = 0
	if _, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidTopK); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("top-k error = %v", err)
	}
	invalidRadius := valid
	invalidRadius.Radius = -1
	if _, err := index.SearchHNSW(context.Background(), []float32{1, 2, 3}, invalidRadius); !errors.Is(err, ErrInvalidRadius) {
		t.Fatalf("radius error = %v", err)
	}
	var nilIndex *HNSWIndex
	if _, err := nilIndex.SearchHNSW(context.Background(), []float32{1}, valid); err == nil {
		t.Fatal("nil index search succeeded")
	}
	if err := (HNSWSearchOptions{}).Validate(); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("options top-k error = %v", err)
	}
	if err := (HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}}).Validate(); !errors.Is(err, ErrInvalidHNSWEF) {
		t.Fatalf("options EF error = %v", err)
	}
	if err := (HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: MaxHNSWEFSearch + 1}).Validate(); !errors.Is(err, ErrInvalidHNSWEF) {
		t.Fatalf("options large EF error = %v", err)
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	inputs := hnswBuildInputs(10000)
	index := buildSearchHNSW(b, MetricL2, inputs, 16, 120)
	query := inputs[4321].Vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: 100}
	b.ResetTimer()
	for b.Loop() {
		if _, err := index.SearchHNSW(context.Background(), query, options); err != nil {
			b.Fatal(err)
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
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range inputs {
		if err := builder.Add(context.Background(), input.Key, input.Vector); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return index
}
