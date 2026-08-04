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
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestVamanaBuildOptionsGraphDeterminismAndOwnership(t *testing.T) {
	defaults := DefaultVamanaBuildOptions(MetricCosine)
	if defaults.MaxDegree != 64 || defaults.SearchListSize != 100 || defaults.Alpha != 1.2 ||
		defaults.MaxOcclusionSize != 750 || defaults.Metric != MetricCosine {
		t.Fatalf("defaults = %#v", defaults)
	}
	valid := DefaultVamanaBuildOptions(MetricL2)
	invalid := []VamanaBuildOptions{
		{},
		func() VamanaBuildOptions { value := valid; value.MaxDegree = 0; return value }(),
		func() VamanaBuildOptions { value := valid; value.MaxDegree = MaxVamanaDegree + 1; return value }(),
		func() VamanaBuildOptions { value := valid; value.SearchListSize = value.MaxDegree - 1; return value }(),
		func() VamanaBuildOptions { value := valid; value.Alpha = float32(math.NaN()); return value }(),
		func() VamanaBuildOptions { value := valid; value.MaxOcclusionSize = 0; return value }(),
	}
	for _, options := range invalid {
		if _, err := NewVamanaBuilder(3, options); !errors.Is(err, ErrInvalidVamanaOptions) {
			t.Fatalf("invalid options %#v: %v", options, err)
		}
	}
	if _, err := NewVamanaBuilder(0, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dimension error = %v", err)
	}

	inputs := hnswBuildInputs(160)
	build := func(saturate bool) *VamanaIndex {
		options := DefaultVamanaBuildOptions(MetricL2)
		options.MaxDegree = 8
		options.SearchListSize = 40
		options.MaxOcclusionSize = 60
		options.SaturateGraph = saturate
		return buildVamana(t, inputs, options)
	}
	first, second := build(false), build(false)
	if !reflect.DeepEqual(first.neighbors, second.neighbors) || first.entryPoint != second.entryPoint ||
		!reflect.DeepEqual(first.neighborDistances, second.neighborDistances) {
		t.Fatal("Vamana build is not deterministic")
	}
	if first.Dimension() != 3 || first.Metric() != MetricL2 || first.Len() != len(inputs) || first.BuildOptions().MaxDegree != 8 {
		t.Fatal("Vamana metadata differs")
	}
	entry, found := first.EntryPoint()
	if !found || entry != first.keys[first.entryPoint] {
		t.Fatal("Vamana entry point missing")
	}
	assertVamanaGraphInvariants(t, first)
	for _, adjacent := range build(true).neighbors {
		if len(adjacent) == 0 {
			t.Fatal("saturated graph retained an empty neighbor list")
		}
	}
	original, _ := first.Vector(inputs[0].Key)
	inputs[0].Vector[0]++
	again, _ := first.Vector(inputs[0].Key)
	if original[0] != again[0] {
		t.Fatal("builder did not own input vector")
	}
	original[0]++
	again, _ = first.Vector(inputs[0].Key)
	if original[0] == again[0] {
		t.Fatal("Vector exposed mutable storage")
	}
}

func TestVamanaSearchMetricsFilterRadiusAndRecall(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		inputs := hnswBuildInputs(180)
		options := DefaultVamanaBuildOptions(metric)
		options.MaxDegree = 8
		options.SearchListSize = 40
		options.MaxOcclusionSize = 60
		index := buildVamana(t, inputs, options)
		query := slices.Clone(inputs[77].Vector)
		search := VamanaSearchOptions{
			SearchOptions: SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }},
			EFSearch:      80,
		}
		got, err := index.SearchVamana(context.Background(), query, search)
		if err != nil {
			t.Fatalf("metric %d: %v", metric, err)
		}
		want, err := topKCandidatesWithOptions(context.Background(), metric, query, search.SearchOptions, len(inputs), func(position int) Candidate {
			return inputs[position]
		}, true)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("small metric %d search differs: %#v vs %#v", metric, got, want)
		}
	}
	inputs := hnswBuildInputs(80)
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 8, 32
	index := buildVamana(t, inputs, options)
	target := inputs[17]
	bounded, err := index.SearchVamana(context.Background(), target.Vector, VamanaSearchOptions{
		SearchOptions: SearchOptions{TopK: 5, Radius: .01, Filter: func(key uint64) bool { return key == target.Key }},
		EFSearch:      40,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(bounded, []Result{{Key: target.Key, Score: 0}}) {
		t.Fatalf("filtered radius result = %#v", bounded)
	}

	inputs = hnswRaBitQCandidates(DefaultVamanaBruteForceThreshold+120, 64)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		options := DefaultVamanaBuildOptions(metric)
		options.MaxDegree = 16
		options.SearchListSize = 80
		options.MaxOcclusionSize = 160
		index := buildVamana(t, inputs, options)
		var matches, total int
		for queryIndex := 0; queryIndex < 10; queryIndex++ {
			query := inputs[(queryIndex*79+17)%len(inputs)].Vector
			truth, err := TopK(context.Background(), metric, query, inputs, 10)
			if err != nil {
				t.Fatal(err)
			}
			got, err := index.SearchVamana(context.Background(), query, VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EFSearch: 100})
			if err != nil {
				t.Fatal(err)
			}
			if metric == MetricL2 && queryIndex == 0 {
				prefetched, err := index.SearchVamana(context.Background(), query, VamanaSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, EFSearch: 100,
					PrefetchOffset: math.MaxUint32, PrefetchLines: math.MaxUint32,
				})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(prefetched, got) {
					t.Fatalf("prefetch controls changed results: %#v vs %#v", prefetched, got)
				}
			}
			matches += resultOverlap(got, truth)
			total += len(truth)
		}
		if recall := float64(matches) / float64(total); recall < .80 {
			t.Fatalf("metric %d recall@10 = %.3f", metric, recall)
		}
	}
}

func TestVamanaRobustPrunePinnedGeometry(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize = 3, 4, 4
	index := &VamanaIndex{
		dimension: 2, options: options,
		keys: []uint64{1, 2, 3, 4}, vectors: []float32{0, 0, 1, 0, 1.1, 0, 0, 2},
	}
	candidates := []vamanaDistanceNode{{position: 1, distance: 1}, {position: 2, distance: 1.21}, {position: 3, distance: 4}}
	selected, err := index.robustPrune(context.Background(), 0, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(selected, []vamanaDistanceNode{{position: 1, distance: 1}, {position: 3, distance: 4}}) {
		t.Fatalf("pruned geometry = %#v", selected)
	}
	index.options.SaturateGraph = true
	selected, err = index.robustPrune(context.Background(), 0, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(selected, []vamanaDistanceNode{{position: 1, distance: 1}, {position: 3, distance: 4}, {position: 2, distance: 1.21}}) {
		t.Fatalf("saturated geometry = %#v", selected)
	}
}

func TestVamanaEmptyIncrementalAndValidation(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree = 4
	options.SearchListSize = 12
	builder, err := NewVamanaBuilder(3, options)
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if index.Len() != 0 {
		t.Fatal("empty Vamana index is not empty")
	}
	if _, found := index.EntryPoint(); found {
		t.Fatal("empty Vamana index has entry point")
	}
	validSearch := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EFSearch: 10}
	if got, err := index.SearchVamana(context.Background(), []float32{0, 0, 0}, validSearch); err != nil || len(got) != 0 {
		t.Fatalf("empty search = %#v, %v", got, err)
	}
	vector := []float32{1, 2, 3}
	if err := index.Add(context.Background(), 7, vector); err != nil {
		t.Fatal(err)
	}
	if index.Len() != 1 {
		t.Fatalf("incremental length = %d", index.Len())
	}
	if err := index.Add(context.Background(), 7, vector); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	if err := index.Add(context.Background(), 8, vector[:2]); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err := index.SearchVamana(nil, vector, validSearch); err == nil {
		t.Fatal("nil search context succeeded")
	}
	validSearch.EFSearch = 0
	if _, err := index.SearchVamana(context.Background(), vector, validSearch); !errors.Is(err, ErrInvalidVamanaEF) {
		t.Fatalf("EF error = %v", err)
	}
}

func buildVamana(t testing.TB, inputs []Candidate, options VamanaBuildOptions) *VamanaIndex {
	t.Helper()
	dimension := 3
	if len(inputs) != 0 {
		dimension = len(inputs[0].Vector)
	}
	builder, err := NewVamanaBuilder(dimension, options)
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

func assertVamanaGraphInvariants(t testing.TB, index *VamanaIndex) {
	t.Helper()
	if err := validateVamanaIndex(context.Background(), index); err != nil {
		t.Fatal(err)
	}
	for position, adjacent := range index.neighbors {
		seen := make(map[int]struct{}, len(adjacent))
		for _, neighbor := range adjacent {
			if neighbor == position {
				t.Fatalf("node %d has self-loop", position)
			}
			if _, found := seen[neighbor]; found {
				t.Fatalf("node %d has duplicate neighbor %d", position, neighbor)
			}
			seen[neighbor] = struct{}{}
		}
	}
}
