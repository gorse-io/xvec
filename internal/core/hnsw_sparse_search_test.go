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

func TestSparseHNSWSearchSmallGraphMatchesFlat(t *testing.T) {
	t.Parallel()
	inputs := sparseHNSWBuildInputs(240)
	index := buildSearchSparseHNSW(t, inputs, 8, 48)
	flat, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range inputs {
		if err := flat.AddSparse(context.Background(), input.key, input.vector); err != nil {
			t.Fatal(err)
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
	if err != nil {
		t.Fatal(err)
	}
	want, err := flat.SearchSparseWithOptions(context.Background(), query, options.SearchOptions)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sparse HNSW = %#v, Flat = %#v", got, want)
	}
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
		if err != nil {
			t.Fatal(err)
		}
		want, err := exactSparseResults(context.Background(), query, inputs, 10)
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
		t.Fatalf("sparse recall@10 = %.3f, want >= 0.80", recall)
	}
}

func TestSparseHNSWSearchLargeGraphFilterRadiusAndEF(t *testing.T) {
	inputs := sparseHNSWBuildInputs(DefaultHNSWBruteForceThreshold + 100)
	index := buildSearchSparseHNSW(t, inputs, 16, 100)
	target := inputs[len(inputs)-37]
	targetScore, err := sparseHNSWScore(target.vector, target.vector)
	if err != nil {
		t.Fatal(err)
	}
	results, err := index.SearchSparseHNSW(context.Background(), target.vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{
			TopK:   5,
			Radius: targetScore,
			Filter: func(key uint64) bool { return key == target.key },
		},
		EF: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: target.key, Score: targetScore}}) {
		t.Fatalf("filtered radius results = %#v", results)
	}
	results, err = index.SearchSparseHNSW(context.Background(), target.vector, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 25 {
		t.Fatalf("EF below TopK returned %d results, want 25", len(results))
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: 2}, {Key: 9}, {Key: 50}}) {
		t.Fatalf("ties = %#v", results)
	}
	results, err = index.SearchSparse(context.Background(), SparseVector{}, 0)
	if err != nil || results == nil || len(results) != 0 {
		t.Fatalf("zero top-k = %#v, %v", results, err)
	}
	builder, _ := NewSparseHNSWBuilder(DefaultSparseHNSWBuildOptions())
	empty, _ := builder.Build(context.Background())
	results, err = empty.SearchSparseHNSW(context.Background(), SparseVector{}, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, EF: 1,
	})
	if err != nil || results == nil || len(results) != 0 {
		t.Fatalf("empty search = %#v, %v", results, err)
	}
}

func TestSparseHNSWResultTieBreaksByKey(t *testing.T) {
	t.Parallel()
	index := &SparseHNSWIndex{keys: []uint64{90, 3}}
	left := hnswScoredNode{position: 0, score: 1}
	right := hnswScoredNode{position: 1, score: 1}
	if index.resultNodeBetter(left, right) || !index.resultNodeBetter(right, left) {
		t.Fatal("equal sparse HNSW result scores did not prefer the smaller key")
	}
}

func TestSparseHNSWSearchValidation(t *testing.T) {
	t.Parallel()
	index := buildSearchSparseHNSW(t, sparseHNSWBuildInputs(4), 2, 4)
	query := SparseVector{Indices: []uint32{1}, Values: []float32{2}}
	valid := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 2}
	if _, err := index.SearchSparseHNSW(nil, query, valid); err == nil {
		t.Fatal("nil context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := index.SearchSparseHNSW(canceled, query, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
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
		if _, err := index.SearchSparseHNSW(context.Background(), test.query, valid); !errors.Is(err, test.want) {
			t.Fatalf("query %#v error = %v", test.query, err)
		}
	}
	invalidEF := valid
	invalidEF.EF = 0
	if _, err := index.SearchSparseHNSW(context.Background(), query, invalidEF); !errors.Is(err, ErrInvalidHNSWEF) {
		t.Fatalf("EF error = %v", err)
	}
	invalidEF.EF = MaxHNSWEFSearch + 1
	if _, err := index.SearchSparseHNSW(context.Background(), query, invalidEF); !errors.Is(err, ErrInvalidHNSWEF) {
		t.Fatalf("large EF error = %v", err)
	}
	invalidTopK := valid
	invalidTopK.TopK = 0
	if _, err := index.SearchSparseHNSW(context.Background(), query, invalidTopK); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("top-k error = %v", err)
	}
	invalidRadius := valid
	invalidRadius.Radius = -1
	if _, err := index.SearchSparseHNSW(context.Background(), query, invalidRadius); !errors.Is(err, ErrInvalidRadius) {
		t.Fatalf("radius error = %v", err)
	}
	var nilIndex *SparseHNSWIndex
	if _, err := nilIndex.SearchSparseHNSW(context.Background(), query, valid); err == nil {
		t.Fatal("nil index search succeeded")
	}

	overflow := buildSearchSparseHNSW(t, []sparseHNSWInput{{
		key: 1, vector: SparseVector{Indices: []uint32{1}, Values: []float32{math.MaxFloat32}},
	}}, 2, 4)
	if _, err := overflow.SearchSparse(context.Background(), SparseVector{
		Indices: []uint32{1}, Values: []float32{math.MaxFloat32},
	}, 1); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("overflow score error = %v", err)
	}
}

func BenchmarkSparseHNSWSearch(b *testing.B) {
	inputs := sparseHNSWBuildInputs(10000)
	index := buildSearchSparseHNSW(b, inputs, 16, 120)
	query := inputs[4321].vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: 100}
	b.ResetTimer()
	for b.Loop() {
		if _, err := index.SearchSparseHNSW(context.Background(), query, options); err != nil {
			b.Fatal(err)
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
