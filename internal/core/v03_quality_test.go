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
	"fmt"
	"testing"
)

// TestV03IVFPartialProbeRecall locks the quality boundary that is not covered
// by IVF's full-probe exactness tests. The corpus, training seed, and query
// selection are deterministic so a graph/training change cannot silently buy
// speed by crossing the release's recall floor.
func TestV03IVFPartialProbeRecall(t *testing.T) {
	candidates := quantizedIndexCandidates(2048)
	index := buildDenseIVFFromCandidates(t, candidates, 32)

	var matched, total int
	for queryIndex := range 24 {
		query := candidates[(queryIndex*79+31)%len(candidates)].Vector
		got, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
			SearchOptions: SearchOptions{TopK: 10},
			NProbe:        8,
		})
		if err != nil {
			t.Fatal(err)
		}
		want, err := TopK(context.Background(), MetricL2, query, candidates, 10)
		if err != nil {
			t.Fatal(err)
		}
		matched += resultOverlap(got, want)
		total += len(want)
	}
	if recall := float64(matched) / float64(total); recall < .90 {
		t.Fatalf("IVF recall@10 = %.3f, want >= .90", recall)
	}
}

// BenchmarkV03DenseSearchQuality records latency, allocation, and recall on a
// common corpus. It complements the algorithm-specific build/search
// benchmarks and makes performance comparisons meaningful only at their
// reported quality point.
func BenchmarkV03DenseSearchQuality(b *testing.B) {
	candidates := quantizedIndexCandidates(10_000)
	query := candidates[4321].Vector
	truth, err := TopK(context.Background(), MetricL2, query, candidates, 10)
	if err != nil {
		b.Fatal(err)
	}

	flat, err := NewDenseFlatIndex(4, MetricL2)
	if err != nil {
		b.Fatal(err)
	}
	for _, candidate := range candidates {
		if err := flat.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			b.Fatal(err)
		}
	}
	ivf := buildDenseIVFFromCandidates(b, candidates, 64)
	hnsw := buildDenseHNSWFromCandidates(b, candidates)
	quantizedHNSW, err := NewScalarQuantizedHNSWIndex(
		context.Background(), hnsw, QuantizationInt8, nil,
	)
	if err != nil {
		b.Fatal(err)
	}

	benchmarks := []struct {
		name   string
		search func() ([]Result, error)
	}{
		{
			name: "Flat",
			search: func() ([]Result, error) {
				return flat.Search(context.Background(), query, 10)
			},
		},
		{
			name: "IVF/NProbe=8",
			search: func() ([]Result, error) {
				return ivf.SearchIVF(context.Background(), query, IVFSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, NProbe: 8,
				})
			},
		},
		{
			name: "HNSW/EF=100",
			search: func() ([]Result, error) {
				return hnsw.SearchHNSW(context.Background(), query, HNSWSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, EF: 100,
				})
			},
		},
		{
			name: "HNSW-INT8/EF=100",
			search: func() ([]Result, error) {
				return quantizedHNSW.SearchHNSW(context.Background(), query, HNSWSearchOptions{
					SearchOptions: SearchOptions{TopK: 10}, EF: 100,
				})
			},
		},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			got, err := benchmark.search()
			if err != nil {
				b.Fatal(err)
			}
			recall := float64(resultOverlap(got, truth)) / float64(len(truth))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := benchmark.search(); err != nil {
					b.Fatal(fmt.Errorf("search: %w", err))
				}
			}
			b.ReportMetric(recall, "recall@10")
		})
	}
}

func resultOverlap(got, want []Result) int {
	keys := make(map[uint64]struct{}, len(want))
	for _, result := range want {
		keys[result.Key] = struct{}{}
	}
	var matched int
	for _, result := range got {
		if _, found := keys[result.Key]; found {
			matched++
		}
	}
	return matched
}
