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

func TestScalarQuantizedFlatSearchAndRefinement(t *testing.T) {
	t.Parallel()
	candidates := quantizedIndexCandidates(80)
	index, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricIP, QuantizationInt4, nil, candidates)
	if err != nil {
		t.Fatal(err)
	}
	query := []float32{3.25, -1.5, 7, 2}
	options := SearchOptions{
		TopK: 12,
		Filter: func(key uint64) bool {
			return key%3 != 0
		},
	}
	got, err := index.SearchWithOptions(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	want := exactQuantizedResults(t, QuantizationInt4, MetricIP, nil, query, candidates, options)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("quantized Flat = %#v, want %#v", got, want)
	}

	refiner, err := NewOriginalVectorRefiner(index, MetricIP)
	if err != nil {
		t.Fatal(err)
	}
	refined, err := RefinedSearch(context.Background(), index, refiner, query, options, 100)
	if err != nil {
		t.Fatal(err)
	}
	exact, err := TopK(context.Background(), MetricIP, query, filterDenseCandidates(candidates, options.Filter), options.TopK)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(refined, exact) {
		t.Fatalf("refined quantized Flat = %#v, exact %#v", refined, exact)
	}
	stored, found := index.Vector(candidates[0].Key)
	if !found || !slices.Equal(stored, candidates[0].Vector) {
		t.Fatalf("original vector = %v, %v", stored, found)
	}
	stored[0]++
	again, _ := index.Vector(candidates[0].Key)
	if slices.Equal(stored, again) {
		t.Fatal("Vector exposed mutable original storage")
	}
}

func TestScalarQuantizedFlatRotationAndRadius(t *testing.T) {
	t.Parallel()
	candidates := quantizedIndexCandidates(60)
	rotator, err := NewFHTRotatorFromSigns(4, []byte{0x13, 0x57, 0x9b, 0xdf})
	if err != nil {
		t.Fatal(err)
	}
	reformer, err := NewRotationReformer(rotator)
	if err != nil {
		t.Fatal(err)
	}
	index, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt8, reformer, candidates)
	if err != nil {
		t.Fatal(err)
	}
	query := candidates[17].Vector
	all, err := index.SearchWithOptions(context.Background(), query, SearchOptions{TopK: len(candidates)})
	if err != nil {
		t.Fatal(err)
	}
	radius := all[12].Score
	got, err := index.SearchWithOptions(context.Background(), query, SearchOptions{TopK: len(candidates), Radius: radius})
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range got {
		if result.Score > radius {
			t.Fatalf("score %v exceeds radius %v", result.Score, radius)
		}
	}
	want := exactQuantizedResults(t, QuantizationInt8, MetricL2, reformer, query, candidates, SearchOptions{
		TopK: len(candidates), Radius: radius,
	})
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("rotated radius results differ")
	}
}

func TestScalarQuantizedHNSWSmallGraphMatchesQuantizedFlat(t *testing.T) {
	t.Parallel()
	candidates := quantizedIndexCandidates(240)
	base := buildDenseHNSWFromCandidates(t, candidates)
	hnsw, err := NewScalarQuantizedHNSWIndex(context.Background(), base, QuantizationFP16, nil)
	if err != nil {
		t.Fatal(err)
	}
	flat, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationFP16, nil, candidates)
	if err != nil {
		t.Fatal(err)
	}
	query := candidates[113].Vector
	options := HNSWSearchOptions{
		SearchOptions: SearchOptions{
			TopK:   25,
			Filter: func(key uint64) bool { return key%2 == 1 },
		},
		EF:             40,
		PrefetchOffset: math.MaxUint32,
		PrefetchLines:  math.MaxUint32,
	}
	got, err := hnsw.SearchHNSW(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	want, err := flat.SearchWithOptions(context.Background(), query, options.SearchOptions)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("small quantized HNSW = %#v, Flat = %#v", got, want)
	}
	if err := base.Add(context.Background(), 999999, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if hnsw.Len() != len(candidates) {
		t.Fatalf("quantized HNSW snapshot length = %d", hnsw.Len())
	}
}

func TestScalarQuantizedHNSWLargeGraphPrefetchAndRecall(t *testing.T) {
	candidates := quantizedIndexCandidates(DefaultHNSWBruteForceThreshold + 200)
	base := buildDenseHNSWFromCandidates(t, candidates)
	index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, QuantizationInt8, nil)
	if err != nil {
		t.Fatal(err)
	}
	flat, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt8, nil, candidates)
	if err != nil {
		t.Fatal(err)
	}
	var matched, total int
	for queryIndex := 0; queryIndex < 20; queryIndex++ {
		query := candidates[(queryIndex*43+19)%len(candidates)].Vector
		plain, err := index.SearchHNSW(context.Background(), query, HNSWSearchOptions{
			SearchOptions: SearchOptions{TopK: 10}, EF: 120,
		})
		if err != nil {
			t.Fatal(err)
		}
		prefetched, err := index.SearchHNSW(context.Background(), query, HNSWSearchOptions{
			SearchOptions: SearchOptions{TopK: 10}, EF: 120, PrefetchOffset: 8, PrefetchLines: 0,
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(prefetched, plain) {
			t.Fatal("prefetch controls changed quantized HNSW results")
		}
		truth, err := flat.SearchWithOptions(context.Background(), query, SearchOptions{TopK: 10})
		if err != nil {
			t.Fatal(err)
		}
		keys := make(map[uint64]struct{}, len(truth))
		for _, result := range truth {
			keys[result.Key] = struct{}{}
		}
		for _, result := range plain {
			if _, found := keys[result.Key]; found {
				matched++
			}
		}
		total += len(truth)
	}
	if recall := float64(matched) / float64(total); recall < .75 {
		t.Fatalf("quantized HNSW recall@10 = %.3f, want >= .75", recall)
	}
}

func TestScalarQuantizedIVFFullProbeMatchesQuantizedFlat(t *testing.T) {
	t.Parallel()
	candidates := quantizedIndexCandidates(320)
	base := buildDenseIVFFromCandidates(t, candidates, 16)
	index, err := NewScalarQuantizedIVFIndex(context.Background(), base, QuantizationInt4, nil)
	if err != nil {
		t.Fatal(err)
	}
	flat, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt4, nil, candidates)
	if err != nil {
		t.Fatal(err)
	}
	query := candidates[217].Vector
	options := SearchOptions{
		TopK:   31,
		Filter: func(key uint64) bool { return key%5 != 0 },
	}
	got, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: options, NProbe: index.NList(),
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := flat.SearchWithOptions(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("full-probe quantized IVF = %#v, Flat = %#v", got, want)
	}
	if err := base.Add(context.Background(), 999999, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if index.Len() != len(candidates) {
		t.Fatalf("quantized IVF snapshot length = %d", index.Len())
	}
}

func TestScalarQuantizedIndexValidation(t *testing.T) {
	t.Parallel()
	valid := []Candidate{{Key: 1, Vector: []float32{1, 2, 3, 4}}}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewScalarQuantizedFlatIndex(canceled, 4, MetricL2, QuantizationInt8, nil, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled constructor error = %v", err)
	}
	if _, err := NewScalarQuantizedFlatIndex(nil, 4, MetricL2, QuantizationInt8, nil, valid); err == nil {
		t.Fatal("nil constructor context succeeded")
	}
	if _, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, 0, nil, valid); !errors.Is(err, ErrInvalidQuantization) {
		t.Fatalf("kind error = %v", err)
	}
	duplicate := append(slices.Clone(valid), valid[0])
	if _, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt8, nil, duplicate); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("duplicate error = %v", err)
	}
	spliced := []Candidate{{Key: 1, Vector: []float32{1, 2, 3}}, {Key: 2, Vector: []float32{4, 5, 6, 7, 8}}}
	if _, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt8, nil, spliced); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("per-candidate dimension error = %v", err)
	}
	nonfinite := []Candidate{{Key: 1, Vector: []float32{1, 2, 3, float32(math.NaN())}}}
	if _, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt8, nil, nonfinite); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("finite error = %v", err)
	}
	wrongRotator, _ := NewFHTRotatorFromSigns(2, []byte{1, 2, 3, 4})
	wrongReformer, _ := NewRotationReformer(wrongRotator)
	if _, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt8, wrongReformer, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("reformer error = %v", err)
	}
	if _, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt8, truncatingReformer{dimension: 4}, valid); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("reformer output error = %v", err)
	}

	baseHNSW := buildDenseHNSWFromCandidates(t, valid)
	if _, err := NewScalarQuantizedHNSWIndex(nil, baseHNSW, QuantizationInt8, nil); err == nil {
		t.Fatal("nil HNSW quantization context succeeded")
	}
	if _, err := NewScalarQuantizedHNSWIndex(context.Background(), nil, QuantizationInt8, nil); err == nil {
		t.Fatal("nil HNSW source succeeded")
	}
	baseIVF := buildDenseIVFFromCandidates(t, valid, 1)
	if _, err := NewScalarQuantizedIVFIndex(context.Background(), nil, QuantizationInt8, nil); err == nil {
		t.Fatal("nil IVF source succeeded")
	}
	index, err := NewScalarQuantizedIVFIndex(context.Background(), baseIVF, QuantizationInt8, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := index.SearchIVF(context.Background(), valid[0].Vector, IVFSearchOptions{SearchOptions: SearchOptions{TopK: 1}}); !errors.Is(err, ErrInvalidIVFNProbe) {
		t.Fatalf("NProbe error = %v", err)
	}
}

func quantizedIndexCandidates(count int) []Candidate {
	result := make([]Candidate, count)
	for position := range result {
		result[position] = Candidate{
			Key: uint64(position*17 + 3),
			Vector: []float32{
				float32(position%23) - 7.25,
				float32((position*7)%31) - 11.5,
				float32((position*13)%37) + .125,
				float32((position*19)%41) - 3.75,
			},
		}
	}
	return result
}

func buildDenseHNSWFromCandidates(t testing.TB, candidates []Candidate) *HNSWIndex {
	t.Helper()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M = 12
	options.EFConstruction = 80
	options.Seed = 0x123456789abcdef
	builder, err := NewHNSWBuilder(4, options)
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

func buildDenseIVFFromCandidates(t testing.TB, candidates []Candidate, nlist int) *IVFIndex {
	t.Helper()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = nlist
	options.NIterations = 12
	options.Seed = 0xabcdef
	builder, err := NewIVFBuilder(4, options)
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

func exactQuantizedResults(
	t testing.TB,
	kind Quantization,
	metric Metric,
	reformer DenseReformer,
	query []float32,
	candidates []Candidate,
	options SearchOptions,
) []Result {
	t.Helper()
	transformedQuery := slices.Clone(query)
	var err error
	if reformer != nil {
		transformedQuery, err = reformer.Transform(query)
		if err != nil {
			t.Fatal(err)
		}
	}
	queryCode, err := QuantizeVector(kind, transformedQuery)
	if err != nil {
		t.Fatal(err)
	}
	var results []Result
	for _, candidate := range candidates {
		if options.Filter != nil && !options.Filter(candidate.Key) {
			continue
		}
		vector := slices.Clone(candidate.Vector)
		if reformer != nil {
			vector, err = reformer.Transform(candidate.Vector)
			if err != nil {
				t.Fatal(err)
			}
		}
		code, err := QuantizeVector(kind, vector)
		if err != nil {
			t.Fatal(err)
		}
		score, err := QuantizedDistance(metric, code, queryCode)
		if err != nil {
			t.Fatal(err)
		}
		if scoreWithinRadius(metric, score, options.Radius) {
			results = append(results, Result{Key: candidate.Key, Score: score})
		}
	}
	return MergeSearchResults(metric, options.TopK, results)
}

func filterDenseCandidates(candidates []Candidate, filter CandidateFilter) []Candidate {
	filtered := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		if filter == nil || filter(candidate.Key) {
			filtered = append(filtered, candidate)
		}
	}
	return filtered
}

type truncatingReformer struct{ dimension int }

func (r truncatingReformer) Dimension() int { return r.dimension }
func (r truncatingReformer) Transform(vector []float32) ([]float32, error) {
	return slices.Clone(vector[:len(vector)-1]), nil
}
func (r truncatingReformer) Revert(vector []float32) ([]float32, error) {
	return slices.Clone(vector), nil
}
