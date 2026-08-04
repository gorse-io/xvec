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
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestHNSWRaBitQBuildOptionsAndDeterminism(t *testing.T) {
	defaults := DefaultHNSWRaBitQBuildOptions(MetricCosine)
	if defaults.Metric != MetricCosine || defaults.TotalBits != 7 || defaults.Clusters != 16 ||
		defaults.M != 50 || defaults.EFConstruction != 500 {
		t.Fatalf("defaults = %#v", defaults)
	}
	invalid := []HNSWRaBitQBuildOptions{
		{},
		func() HNSWRaBitQBuildOptions { value := defaults; value.Metric = MetricMIPSL2; return value }(),
		func() HNSWRaBitQBuildOptions { value := defaults; value.TotalBits = 10; return value }(),
		func() HNSWRaBitQBuildOptions { value := defaults; value.M = 0; return value }(),
		func() HNSWRaBitQBuildOptions { value := defaults; value.EFConstruction = 1; return value }(),
	}
	for _, options := range invalid {
		if _, err := NewHNSWRaBitQBuilder(64, options); !errors.Is(err, ErrInvalidHNSWRaBitQOptions) {
			t.Fatalf("invalid options %#v: %v", options, err)
		}
	}
	if _, err := NewHNSWRaBitQBuilder(63, defaults); !errors.Is(err, ErrInvalidHNSWRaBitQOptions) {
		t.Fatalf("small dimension error = %v", err)
	}

	candidates := hnswRaBitQCandidates(120, 70)
	options := hnswRaBitQTestOptions(MetricL2)
	options.Workers = 1
	first := buildHNSWRaBitQ(t, candidates, options)
	options.Workers = 4
	second := buildHNSWRaBitQ(t, candidates, options)
	if !reflect.DeepEqual(first.ModelState(), second.ModelState()) ||
		!reflect.DeepEqual(first.base.levels, second.base.levels) ||
		!reflect.DeepEqual(first.base.neighbors, second.base.neighbors) ||
		!reflect.DeepEqual(first.codes, second.codes) {
		t.Fatal("HNSW-RaBitQ changed across worker counts")
	}
	if first.Dimension() != 70 || first.Metric() != MetricL2 || first.Len() != len(candidates) || first.MaxLevel() != first.base.maxLevel {
		t.Fatal("HNSW-RaBitQ metadata differs")
	}
	entry, found := first.EntryPoint()
	if !found {
		t.Fatal("built graph has no entry point")
	}
	if level, _ := first.Level(entry); level != first.MaxLevel() {
		t.Fatalf("entry level %d differs from max %d", level, first.MaxLevel())
	}
	vector, found := first.Vector(candidates[0].Key)
	if !found {
		t.Fatal("first vector missing")
	}
	candidates[0].Vector[0]++
	if again, _ := first.Vector(candidates[0].Key); again[0] != vector[0] {
		t.Fatal("builder did not own originals")
	}
	vector[0]++
	if again, _ := first.Vector(candidates[0].Key); again[0] == vector[0] {
		t.Fatal("Vector exposed mutable storage")
	}
}

func TestHNSWRaBitQSearchMetricsFilterRadiusAndRefine(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		candidates := hnswRaBitQCandidates(180, 64)
		index := buildHNSWRaBitQ(t, candidates, hnswRaBitQTestOptions(metric))
		query := slices.Clone(candidates[37].Vector)
		options := HNSWRaBitQSearchOptions{
			SearchOptions: SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }},
			EF:            180,
		}
		got, err := index.SearchHNSWRaBitQ(context.Background(), query, options)
		if err != nil {
			t.Fatalf("metric %d approximate: %v", metric, err)
		}
		if len(got) != options.TopK {
			t.Fatalf("metric %d returned %d results", metric, len(got))
		}
		for position, result := range got {
			if result.Key%3 == 0 {
				t.Fatalf("metric %d returned filtered key %d", metric, result.Key)
			}
			if position > 0 && resultBetter(metric, result, got[position-1]) {
				t.Fatalf("metric %d results are not ordered", metric)
			}
		}

		options.Refine = true
		refined, err := index.SearchHNSWRaBitQ(context.Background(), query, options)
		if err != nil {
			t.Fatalf("metric %d refine: %v", metric, err)
		}
		want, err := topKCandidatesWithOptions(context.Background(), metric, query, options.SearchOptions, len(candidates), func(position int) Candidate {
			return candidates[position]
		}, true)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(refined, want) {
			t.Fatalf("metric %d refined = %#v, want %#v", metric, refined, want)
		}
	}

	candidates := hnswRaBitQCandidates(80, 64)
	index := buildHNSWRaBitQ(t, candidates, hnswRaBitQTestOptions(MetricL2))
	target := candidates[17]
	results, err := index.SearchHNSWRaBitQ(context.Background(), target.Vector, HNSWRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: 5, Radius: .2, Filter: func(key uint64) bool { return key == target.Key }},
		EF:            80, Refine: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(results, []Result{{Key: target.Key, Score: 0}}) {
		t.Fatalf("filtered radius result = %#v", results)
	}
}

func TestHNSWRaBitQLargeGraphRecall(t *testing.T) {
	candidates := hnswRaBitQCandidates(DefaultHNSWBruteForceThreshold+120, 64)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		options := hnswRaBitQTestOptions(metric)
		options.M = 12
		options.EFConstruction = 80
		index := buildHNSWRaBitQ(t, candidates, options)
		var approximateMatches, refinedMatches, total int
		for queryIndex := 0; queryIndex < 12; queryIndex++ {
			query := candidates[(queryIndex*73+19)%len(candidates)].Vector
			truth, err := TopK(context.Background(), metric, query, candidates, 10)
			if err != nil {
				t.Fatal(err)
			}
			approximate, err := index.SearchHNSWRaBitQ(context.Background(), query, HNSWRaBitQSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, EF: 100,
			})
			if err != nil {
				t.Fatal(err)
			}
			refined, err := index.SearchHNSWRaBitQ(context.Background(), query, HNSWRaBitQSearchOptions{
				SearchOptions: SearchOptions{TopK: 10}, EF: 100, Refine: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			approximateMatches += resultOverlap(approximate, truth)
			refinedMatches += resultOverlap(refined, truth)
			total += len(truth)
		}
		if recall := float64(approximateMatches) / float64(total); recall < .75 {
			t.Fatalf("metric %d approximate recall@10 = %.3f", metric, recall)
		}
		if recall := float64(refinedMatches) / float64(total); recall < .85 {
			t.Fatalf("metric %d refined recall@10 = %.3f", metric, recall)
		}
	}
}

func TestHNSWRaBitQEmptyIncrementalAndValidation(t *testing.T) {
	options := hnswRaBitQTestOptions(MetricL2)
	builder, err := NewHNSWRaBitQBuilder(64, options)
	if err != nil {
		t.Fatal(err)
	}
	empty, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 0 || empty.MaxLevel() != -1 {
		t.Fatalf("empty metadata = %d/%d", empty.Len(), empty.MaxLevel())
	}
	if results, err := empty.SearchHNSWRaBitQ(context.Background(), make([]float32, 64), HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 1}); err != nil || len(results) != 0 {
		t.Fatalf("empty search = %#v, %v", results, err)
	}
	vector := hnswRaBitQCandidates(1, 64)[0].Vector
	if err := empty.Add(context.Background(), 99, vector); err != nil {
		t.Fatal(err)
	}
	if empty.Len() != 1 {
		t.Fatalf("incremental length = %d", empty.Len())
	}
	if err := empty.Add(context.Background(), 99, vector); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	if err := empty.Add(context.Background(), 100, vector[:63]); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}

	valid := HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 1}, EF: 1}
	if _, err := empty.SearchHNSWRaBitQ(nil, vector, valid); err == nil {
		t.Fatal("nil search context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := empty.SearchHNSWRaBitQ(canceled, vector, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled search error = %v", err)
	}
	if _, err := empty.SearchHNSWRaBitQ(context.Background(), vector[:63], valid); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("query dimension error = %v", err)
	}
	nonFinite := slices.Clone(vector)
	nonFinite[1] = float32(math.NaN())
	if _, err := empty.SearchHNSWRaBitQ(context.Background(), nonFinite, valid); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("non-finite error = %v", err)
	}
	valid.EF = 0
	if _, err := empty.SearchHNSWRaBitQ(context.Background(), vector, valid); !errors.Is(err, ErrInvalidHNSWEF) {
		t.Fatalf("EF error = %v", err)
	}
}

func TestHNSWRaBitQIncrementalFailuresAreAtomic(t *testing.T) {
	index := buildHNSWRaBitQ(t, hnswRaBitQCandidates(120, 64), hnswRaBitQTestOptions(MetricL2))
	before, err := encodeHNSWRaBitQIndex(context.Background(), index)
	if err != nil {
		t.Fatal(err)
	}
	vector := hnswRaBitQCandidates(1, 64)[0].Vector
	if err := index.Add(nil, 999999, vector); err == nil {
		t.Fatal("nil Add context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := index.Add(canceled, 999999, vector); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled Add error = %v", err)
	}
	if err := index.Add(context.Background(), 999999, vector[:63]); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension Add error = %v", err)
	}
	nonFinite := slices.Clone(vector)
	nonFinite[0] = float32(math.Inf(1))
	if err := index.Add(context.Background(), 999999, nonFinite); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("non-finite Add error = %v", err)
	}
	midClone := newCancelAfterChecks(4)
	if err := index.Add(midClone, 999999, vector); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-generation Add error = %v", err)
	}
	after, err := encodeHNSWRaBitQIndex(context.Background(), index)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after, before) {
		t.Fatal("failed Add changed HNSW-RaBitQ generation")
	}
	if err := index.Add(context.Background(), 999999, vector); err != nil {
		t.Fatal(err)
	}
	if index.Len() != 121 {
		t.Fatalf("successful Add length = %d", index.Len())
	}
}

func TestHNSWRaBitQConcurrentAddSearchSaveAndOpen(t *testing.T) {
	builder, err := NewHNSWRaBitQBuilder(64, hnswRaBitQTestOptions(MetricCosine))
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	errCh := make(chan error, 32)
	var writers sync.WaitGroup
	for worker := 0; worker < 3; worker++ {
		writers.Add(1)
		go func(worker int) {
			defer writers.Done()
			for value := 0; value < 12; value++ {
				vector := hnswRaBitQCandidates(1, 64)[0].Vector
				vector[worker] += float32(value+1) / 7
				key := uint64(worker*100 + value + 1)
				if err := index.Add(context.Background(), key, vector); err != nil {
					errCh <- err
					return
				}
			}
		}(worker)
	}
	stop := make(chan struct{})
	var readers sync.WaitGroup
	for worker := 0; worker < 3; worker++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			query := make([]float32, 64)
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := index.Search(context.Background(), query, 5); err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for generation := 0; generation < 8; generation++ {
			path := filepath.Join(dir, fmt.Sprintf("snapshot-%02d.hrbtq", generation))
			if err := index.Save(context.Background(), path); err != nil {
				errCh <- err
				return
			}
			if _, err := OpenHNSWRaBitQIndex(context.Background(), path); err != nil {
				errCh <- err
				return
			}
		}
	}()
	writers.Wait()
	close(stop)
	readers.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	if index.Len() != 36 {
		t.Fatalf("concurrent final length = %d", index.Len())
	}
}

func BenchmarkHNSWRaBitQSearch(b *testing.B) {
	candidates := hnswRaBitQCandidates(2_000, 64)
	options := hnswRaBitQTestOptions(MetricL2)
	options.M = 12
	options.EFConstruction = 80
	index := buildHNSWRaBitQ(b, candidates, options)
	query := candidates[713].Vector
	for _, refine := range []bool{false, true} {
		name := "Approximate"
		if refine {
			name = "Refined"
		}
		b.Run(name, func(b *testing.B) {
			search := HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: 100, Refine: refine}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := index.SearchHNSWRaBitQ(context.Background(), query, search); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func hnswRaBitQTestOptions(metric Metric) HNSWRaBitQBuildOptions {
	options := DefaultHNSWRaBitQBuildOptions(metric)
	options.TotalBits = 7
	options.Clusters = 8
	options.MaxIterations = 8
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x726162697471
	return options
}

func hnswRaBitQCandidates(count, dimension int) []Candidate {
	vectors := raBitQGaussianVectors(count, dimension, uint64(count*17+dimension))
	candidates := make([]Candidate, count)
	for index := range candidates {
		candidates[index] = Candidate{Key: uint64(index*13 + 7), Vector: vectors[index]}
	}
	return candidates
}

func buildHNSWRaBitQ(t testing.TB, candidates []Candidate, options HNSWRaBitQBuildOptions) *HNSWRaBitQIndex {
	t.Helper()
	dimension := 64
	if len(candidates) != 0 {
		dimension = len(candidates[0].Vector)
	}
	builder, err := NewHNSWRaBitQBuilder(dimension, options)
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
