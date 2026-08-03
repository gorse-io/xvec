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

func TestIVFSearchProbesNearestLists(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{
		{Key: 1, Vector: []float32{0, 0}},
		{Key: 2, Vector: []float32{0, 1}},
		{Key: 3, Vector: []float32{10, 10}},
		{Key: 4, Vector: []float32{10, 11}},
	}, 2)
	lists, err := index.ProbedLists(context.Background(), []float32{0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(lists) != 1 {
		t.Fatalf("probed lists = %v", lists)
	}
	nearList, _ := index.ListForKey(1)
	if lists[0] != nearList {
		t.Fatalf("probed list = %d, want %d", lists[0], nearList)
	}
	results, err := index.SearchIVF(context.Background(), []float32{0, 0}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 4}, NProbe: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []Result{{Key: 1, Score: 0}, {Key: 2, Score: 1}}) {
		t.Fatalf("results = %#v", results)
	}
	allLists, err := index.ProbedLists(context.Background(), []float32{0, 0}, 99)
	if err != nil || len(allLists) != index.NList() {
		t.Fatalf("all lists = %v, %v", allLists, err)
	}
}

func TestIVFFullProbeMatchesFlat(t *testing.T) {
	t.Parallel()
	candidates := []Candidate{
		{Key: 9, Vector: []float32{-2, 1, 0}},
		{Key: 2, Vector: []float32{1, 0, 2}},
		{Key: 7, Vector: []float32{0, 3, -1}},
		{Key: 4, Vector: []float32{2, 2, 2}},
		{Key: 1, Vector: []float32{-1, -1, 1}},
		{Key: 8, Vector: []float32{4, 0, -2}},
	}
	query := []float32{1, 2, .5}
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		index := buildSearchIVF(t, metric, candidates, 3)
		flat, err := TopK(context.Background(), metric, query, candidates, len(candidates))
		if err != nil {
			t.Fatal(err)
		}
		got, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
			SearchOptions: SearchOptions{TopK: len(candidates)}, NProbe: index.NList(),
		})
		if err != nil {
			t.Fatalf("metric %d: %v", metric, err)
		}
		if !slices.Equal(got, flat) {
			t.Fatalf("metric %d IVF = %#v, Flat = %#v", metric, got, flat)
		}
	}
}

func TestIVFSearchFilterRadiusAndTies(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{
		{Key: 5, Vector: []float32{-1}},
		{Key: 2, Vector: []float32{1}},
		{Key: 9, Vector: []float32{2}},
	}, 2)
	results, err := index.SearchIVF(context.Background(), []float32{0}, IVFSearchOptions{
		SearchOptions: SearchOptions{
			TopK: 3, Radius: 1,
			Filter: func(key uint64) bool { return key != 5 },
		},
		NProbe: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []Result{{Key: 2, Score: 1}}) {
		t.Fatalf("filtered results = %#v", results)
	}
	results, err = index.SearchIVF(context.Background(), []float32{0}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 3}, NProbe: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) < 2 || results[0].Key != 2 || results[1].Key != 5 {
		t.Fatalf("tie order = %#v", results)
	}
}

func TestIVFSearchInnerProductRadius(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricIP, []Candidate{
		{Key: 1, Vector: []float32{-2}},
		{Key: 2, Vector: []float32{1}},
		{Key: 3, Vector: []float32{3}},
	}, 2)
	results, err := index.SearchIVF(context.Background(), []float32{1}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 3, Radius: 2}, NProbe: index.NList(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []Result{{Key: 3, Score: 3}}) {
		t.Fatalf("IP radius results = %#v", results)
	}
}

func TestIVFSearchDefaultAndEmpty(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{{Key: 1, Vector: []float32{1}}}, 1)
	results, err := index.Search(context.Background(), []float32{0}, 1)
	if err != nil || !slices.Equal(results, []Result{{Key: 1, Score: 1}}) {
		t.Fatalf("default search = %#v, %v", results, err)
	}
	results, err = index.Search(context.Background(), []float32{0}, 0)
	if err != nil || results == nil || len(results) != 0 {
		t.Fatalf("zero top-k = %#v, %v", results, err)
	}

	options := DefaultIVFBuildOptions(MetricL2)
	builder, _ := NewIVFBuilder(1, options)
	empty, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	results, err = empty.SearchIVF(context.Background(), []float32{0}, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: 1}, NProbe: 1,
	})
	if err != nil || results == nil || len(results) != 0 {
		t.Fatalf("empty search = %#v, %v", results, err)
	}
	lists, err := empty.ProbedLists(context.Background(), []float32{0}, 1)
	if err != nil || lists == nil || len(lists) != 0 {
		t.Fatalf("empty probes = %v, %v", lists, err)
	}
}

func TestIVFSearchValidation(t *testing.T) {
	t.Parallel()
	index := buildSearchIVF(t, MetricL2, []Candidate{{Key: 1, Vector: []float32{1, 2}}}, 1)
	valid := IVFSearchOptions{SearchOptions: SearchOptions{TopK: 1}, NProbe: 1}
	if _, err := index.SearchIVF(nil, []float32{1, 2}, valid); err == nil {
		t.Fatal("nil context accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := index.SearchIVF(ctx, []float32{1, 2}, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	if _, err := index.SearchIVF(context.Background(), []float32{1}, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err := index.SearchIVF(context.Background(), []float32{1, float32(math.NaN())}, valid); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("non-finite error = %v", err)
	}
	invalidProbe := valid
	invalidProbe.NProbe = 0
	if _, err := index.SearchIVF(context.Background(), []float32{1, 2}, invalidProbe); !errors.Is(err, ErrInvalidIVFNProbe) {
		t.Fatalf("probe error = %v", err)
	}
	invalidTopK := valid
	invalidTopK.TopK = 0
	if _, err := index.SearchIVF(context.Background(), []float32{1, 2}, invalidTopK); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("top-k error = %v", err)
	}
	if _, err := index.Search(context.Background(), []float32{1, 2}, -1); err == nil {
		t.Fatal("negative top-k accepted")
	}
	if _, err := index.ProbedLists(context.Background(), []float32{1, 2}, 0); !errors.Is(err, ErrInvalidIVFNProbe) {
		t.Fatalf("probe validation error = %v", err)
	}
	var nilIndex *IVFIndex
	if _, err := nilIndex.Search(context.Background(), []float32{1}, 1); err == nil {
		t.Fatal("nil index accepted")
	}
}

func buildSearchIVF(t *testing.T, metric Metric, candidates []Candidate, nlist int) *IVFIndex {
	t.Helper()
	options := DefaultIVFBuildOptions(metric)
	options.NList = nlist
	options.NIterations = 20
	options.Seed = 17
	builder, err := NewIVFBuilder(len(candidates[0].Vector), options)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range candidates {
		if err := builder.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return index
}
