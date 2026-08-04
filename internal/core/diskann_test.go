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
	"sync"
	"testing"
)

func TestDiskANNBuildSearchMetricsFilterRadiusAndRefiner(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		t.Run(diskANNMetricName(metric), func(t *testing.T) {
			candidates := diskANNIndexCandidates(72, 8)
			options := DefaultDiskANNBuildOptions(metric)
			options.MaxDegree, options.ListSize, options.PQChunks = 8, 24, 4
			options.CacheCapacity = len(candidates)
			index := buildDiskANNIndex(t, candidates, options)
			if index.Dimension() != 8 || index.Metric() != metric || index.Len() != len(candidates) || index.PQChunks() != 4 {
				t.Fatalf("index metadata = dimension %d metric %d len %d chunks %d", index.Dimension(), index.Metric(), index.Len(), index.PQChunks())
			}
			if got, ok := index.EntryPoint(); !ok || !slices.ContainsFunc(candidates, func(candidate Candidate) bool { return candidate.Key == got }) {
				t.Fatalf("entry point = %d, %v", got, ok)
			}
			query := slices.Clone(candidates[17].Vector)
			linearOptions := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 12}, ListSize: len(candidates), Linear: true}
			want, err := index.SearchDiskANN(context.Background(), query, linearOptions)
			if err != nil {
				t.Fatal(err)
			}
			graphOptions := linearOptions
			graphOptions.Linear = false
			got, err := index.SearchDiskANN(context.Background(), query, graphOptions)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("graph = %#v, exact = %#v", got, want)
			}
			if got[0].Key != candidates[17].Key {
				t.Fatalf("self result = %#v", got[0])
			}

			filter := func(key uint64) bool { return key%2 == 0 }
			filteredOptions := graphOptions
			filteredOptions.Filter = filter
			filtered, err := index.SearchDiskANN(context.Background(), query, filteredOptions)
			if err != nil {
				t.Fatal(err)
			}
			for _, result := range filtered {
				if !filter(result.Key) {
					t.Fatalf("filter admitted %#v", result)
				}
			}
			if len(filtered) == 0 {
				t.Fatal("filter removed every result")
			}

			radiusOptions := graphOptions
			radiusOptions.Radius = want[min(5, len(want)-1)].Score
			if metric == MetricIP && radiusOptions.Radius <= 0 {
				radiusOptions.Radius = 0.0001
			}
			radius, err := index.SearchDiskANN(context.Background(), query, radiusOptions)
			if err != nil {
				t.Fatal(err)
			}
			for _, result := range radius {
				if !scoreWithinRadius(metric, result.Score, radiusOptions.Radius) {
					t.Fatalf("radius admitted %#v", result)
				}
			}

			refiner, err := NewOriginalVectorRefiner(index, metric)
			if err != nil {
				t.Fatal(err)
			}
			refined, err := refiner.Refine(context.Background(), query, got, SearchOptions{TopK: 5})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(refined, want[:5]) {
				t.Fatalf("refined = %#v, want %#v", refined, want[:5])
			}
			vector, found := index.Vector(candidates[9].Key)
			if !found || !slices.Equal(vector, candidates[9].Vector) {
				t.Fatal("original vector provider differs")
			}
			vector[0]++
			again, _ := index.Vector(candidates[9].Key)
			if slices.Equal(vector, again) {
				t.Fatal("vector provider returned an alias")
			}
		})
	}
}

func TestDiskANNBuilderEmptyValidationCancellationAndClose(t *testing.T) {
	defaults := DefaultDiskANNBuildOptions(MetricL2)
	if defaults.MaxDegree != 100 || defaults.ListSize != 50 || defaults.PQChunks != 0 || defaults.CacheCapacity != 1024 {
		t.Fatalf("defaults = %#v", defaults)
	}
	invalid := []DiskANNBuildOptions{
		{},
		func() DiskANNBuildOptions { value := defaults; value.MaxDegree = 0; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.ListSize = 0; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.PQChunks = -1; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.Workers = -1; return value }(),
		func() DiskANNBuildOptions { value := defaults; value.CacheCapacity = -1; return value }(),
	}
	for _, options := range invalid {
		if _, err := NewDiskANNBuilder(8, options); !errors.Is(err, ErrInvalidDiskANNOptions) {
			t.Fatalf("options %#v error = %v", options, err)
		}
	}
	if _, err := NewDiskANNBuilder(0, defaults); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dimension error = %v", err)
	}
	tooManyChunks := defaults
	tooManyChunks.PQChunks = 9
	if _, err := NewDiskANNBuilder(8, tooManyChunks); !errors.Is(err, ErrInvalidDiskANNOptions) {
		t.Fatalf("chunk error = %v", err)
	}

	builder, err := NewDiskANNBuilder(4, defaults)
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.Add(context.Background(), 7, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := builder.Add(context.Background(), 7, []float32{4, 3, 2, 1}); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	if err := builder.Add(context.Background(), 8, []float32{1}); err == nil {
		t.Fatal("invalid vector succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := builder.Add(canceled, 8, []float32{1, 2, 3, 4}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Add error = %v", err)
	}
	if _, err := builder.Build(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Build error = %v", err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := builder.Build(context.Background()); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("second Build error = %v", err)
	}
	if err := builder.Add(context.Background(), 8, []float32{1, 2, 3, 4}); !errors.Is(err, ErrBuilderClosed) {
		t.Fatalf("post-build Add error = %v", err)
	}
	if _, err := index.SearchDiskANN(context.Background(), []float32{1, 2, 3, 4}, DiskANNSearchOptions{}); !errors.Is(err, ErrInvalidDiskANNListSize) {
		t.Fatalf("invalid list error = %v", err)
	}
	if _, err := index.SearchWithOptions(context.Background(), []float32{1, 2, 3, 4}, SearchOptions{}); !errors.Is(err, ErrInvalidTopK) {
		t.Fatalf("invalid top-k error = %v", err)
	}
	if _, err := index.SearchDiskANN(canceled, []float32{1, 2, 3, 4}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Search error = %v", err)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := index.SearchDiskANN(context.Background(), []float32{1, 2, 3, 4}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1}); !errors.Is(err, ErrDiskANNClosed) {
		t.Fatalf("closed Search error = %v", err)
	}
	if _, found := index.Vector(7); found {
		t.Fatal("closed provider returned a vector")
	}

	emptyBuilder, err := NewDiskANNBuilder(4, defaults)
	if err != nil {
		t.Fatal(err)
	}
	empty, err := emptyBuilder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	results, err := empty.SearchDiskANN(context.Background(), []float32{0, 0, 0, 0}, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 1}, ListSize: 1})
	if err != nil || len(results) != 0 || empty.PQChunks() != 0 {
		t.Fatalf("empty search = %#v, %v", results, err)
	}
}

func TestDiskANNWarmCacheBoundAndSearchOwnership(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 6, 16, 3, 7
	index := buildDiskANNIndex(t, diskANNIndexCandidates(40, 6), options)
	warmed, err := index.WarmCache(context.Background(), 20)
	if err != nil {
		t.Fatal(err)
	}
	if warmed != 7 || index.nodes.cache.Len() != 7 {
		t.Fatalf("warmed %d cache len %d", warmed, index.nodes.cache.Len())
	}
	if _, err := index.WarmCache(context.Background(), -1); !errors.Is(err, ErrInvalidDiskANNOptions) {
		t.Fatalf("negative warm error = %v", err)
	}
	before := index.CacheStats()
	query := diskANNIndexCandidates(1, 6)[0].Vector
	first, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 5}, ListSize: 30})
	if err != nil {
		t.Fatal(err)
	}
	second, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 5}, ListSize: 30})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || index.CacheStats().Hits <= before.Hits {
		t.Fatalf("repeat search differs or missed cache: %#v %#v", first, second)
	}
}

func TestDiskANNApproximateRecall(t *testing.T) {
	const count, dimension, queryCount, topK = 320, 16, 8, 10
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 12, 48, 8, 64
	candidates := diskANNIndexCandidates(count, dimension)
	index := buildDiskANNIndex(t, candidates, options)
	hits := 0
	for queryIndex := 0; queryIndex < queryCount; queryIndex++ {
		query := candidates[(queryIndex*37+11)%count].Vector
		want, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
			SearchOptions: SearchOptions{TopK: topK}, ListSize: count, Linear: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		got, err := index.SearchDiskANN(context.Background(), query, DiskANNSearchOptions{
			SearchOptions: SearchOptions{TopK: topK}, ListSize: 64,
		})
		if err != nil {
			t.Fatal(err)
		}
		truth := make(map[uint64]struct{}, len(want))
		for _, result := range want {
			truth[result.Key] = struct{}{}
		}
		for _, result := range got {
			if _, found := truth[result.Key]; found {
				hits++
			}
		}
	}
	recall := float64(hits) / (queryCount * topK)
	if recall < 0.80 {
		t.Fatalf("recall@%d = %.3f", topK, recall)
	}
}

func TestDiskANNConcurrentSearchAndProviderReads(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricCosine)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 32
	candidates := diskANNIndexCandidates(96, 8)
	index := buildDiskANNIndex(t, candidates, options)
	query := candidates[31].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: 64}
	want, err := index.SearchDiskANN(context.Background(), query, search)
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	errorsCh := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func(worker int) {
			defer wait.Done()
			for iteration := 0; iteration < 20; iteration++ {
				got, err := index.SearchDiskANN(context.Background(), query, search)
				if err != nil {
					errorsCh <- err
					return
				}
				if !reflect.DeepEqual(got, want) {
					errorsCh <- errors.New("concurrent DiskANN result differs")
					return
				}
				candidate := candidates[(worker+iteration)%len(candidates)]
				if vector, found := index.Vector(candidate.Key); !found || !slices.Equal(vector, candidate.Vector) {
					errorsCh <- errors.New("concurrent DiskANN provider read differs")
					return
				}
			}
		}(worker)
	}
	wait.Wait()
	close(errorsCh)
	for err := range errorsCh {
		t.Fatal(err)
	}
}

func buildDiskANNIndex(t testing.TB, candidates []Candidate, options DiskANNBuildOptions) *DiskANNIndex {
	t.Helper()
	dimension := 8
	if len(candidates) != 0 {
		dimension = len(candidates[0].Vector)
	}
	builder, err := NewDiskANNBuilder(dimension, options)
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

func diskANNIndexCandidates(count, dimension int) []Candidate {
	result := make([]Candidate, count)
	for position := range result {
		vector := make([]float32, dimension)
		for component := range vector {
			angle := float64((position+3)*(component+5)) * 0.071
			vector[component] = float32(math.Sin(angle) + 0.45*math.Cos(angle*0.37) + float64(position%9)*0.025)
		}
		result[position] = Candidate{Key: uint64(1000 + position*7), Vector: vector}
	}
	return result
}

func diskANNMetricName(metric Metric) string {
	switch metric {
	case MetricL2:
		return "L2"
	case MetricIP:
		return "IP"
	case MetricCosine:
		return "Cosine"
	case MetricMIPSL2:
		return "MIPSL2"
	default:
		return "invalid"
	}
}
