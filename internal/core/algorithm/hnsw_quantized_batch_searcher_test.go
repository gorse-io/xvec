// Copyright 2026-present the xvec project
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
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestQuantizedHNSWInt8FP16BatchMatchesScalar(t *testing.T) {
	for _, kind := range []Quantization{QuantizationInt8, QuantizationFP16} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {

			const dimension, count = 33, 80
			for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
				t.Run(fmt.Sprint(metric), func(t *testing.T) {
					base := &HNSWIndex{
						dimension: dimension, options: HNSWBuildOptions{Metric: metric, M: count, EFConstruction: count},
						keys: make([]uint64, count), vectors: make([]float32, dimension*count),
						neighbors: make([][][]int, count), levels: make([]int, count),
					}
					for j := range count {
						base.keys[j] = uint64(count - j)
						for d := range dimension {
							if j > 1 {
								base.vectors[j*dimension+d] = float32((j*13+d*7)%31 - 15)
							}
						}
						base.neighbors[j] = make([][]int, 1)
						for n := range count {
							if n != j {
								base.neighbors[j][0] = append(base.neighbors[j][0], n)
							}
						}
					}
					index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, kind, nil)
					require.NoError(t, err)
					visited := acquireHNSWVisited(count)
					defer releaseHNSWVisited(visited)
					for _, query := range [][]float32{make([]float32, dimension), base.vectors[3*dimension : 4*dimension]} {
						code, err := index.vectors.quantizedQuery(query)
						require.NoError(t, err)
						options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: count}, EF: count, PrefetchOffset: 8}
						scoreAt := func(position int) (float32, error) {
							return QuantizedDistance(metric, index.vectors.codes[position], code)
						}
						want, err := index.searchBase(context.Background(), 0, count, options, scoreAt, nil, visited)
						require.NoError(t, err)
						got, err := index.searchBaseQuantized(context.Background(), code, 0, count, options, visited)
						require.NoError(t, err)
						require.Equal(t, want, got)
					}
				})
			}

		})
	}
}

func TestQuantizedHNSWInt4BatchMatchesScalar(t *testing.T) {
	const dimension, count = 32, 80
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		t.Run(fmt.Sprint(metric), func(t *testing.T) {
			base := &HNSWIndex{
				dimension: dimension, options: HNSWBuildOptions{Metric: metric, M: count, EFConstruction: count},
				keys: make([]uint64, count), vectors: make([]float32, dimension*count),
				neighbors: make([][][]int, count), levels: make([]int, count),
			}
			for j := range count {
				base.keys[j] = uint64(count - j)
				for d := range dimension {
					base.vectors[j*dimension+d] = float32((j*13+d*7)%31 - 15)
				}
				base.neighbors[j] = make([][]int, 1)
				for n := range count {
					if n != j {
						base.neighbors[j][0] = append(base.neighbors[j][0], n)
					}
				}
			}
			index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, QuantizationInt4, nil)
			require.NoError(t, err)
			visited := acquireHNSWVisited(count)
			defer releaseHNSWVisited(visited)
			query := base.vectors[3*dimension : 4*dimension]
			code, err := index.vectors.quantizedQuery(query)
			require.NoError(t, err)
			options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: count}, EF: count, PrefetchOffset: 8}
			scoreAt := func(position int) (float32, error) {
				return QuantizedDistance(metric, index.vectors.codes[position], code)
			}
			want, err := index.searchBase(context.Background(), 0, count, options, scoreAt, nil, visited)
			require.NoError(t, err)
			got, err := index.searchBaseQuantized(context.Background(), code, 0, count, options, visited)
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
}

func TestQuantizedHNSWInt8FP16BatchExpandsBoundaryTies(t *testing.T) {
	for _, kind := range []Quantization{QuantizationInt8, QuantizationFP16} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {

			base := &HNSWIndex{
				dimension: 1, options: HNSWBuildOptions{Metric: MetricL2, M: 2, EFConstruction: 3},
				keys: []uint64{1, 30, 2, 10, 3}, vectors: []float32{0, 1, .5, 1, .25},
				neighbors: [][][]int{{{2, 1}}, {{4}}, {{3}}, {nil}, {nil}},
			}
			index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, kind, nil)
			require.NoError(t, err)
			code, err := index.vectors.quantizedQuery([]float32{0})
			require.NoError(t, err)
			visited := acquireHNSWVisited(5)
			defer releaseHNSWVisited(visited)
			options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 3}, EF: 3}
			got, err := index.searchBaseQuantized(context.Background(), code, 0, 3, options, visited)
			require.NoError(t, err)
			require.Equal(t, []hnswScoredNode{{position: 0, score: 0}, {position: 4, score: .0625}, {position: 2, score: .25}}, got)
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, err = index.searchBaseQuantized(ctx, code, 0, 3, options, visited)
			require.ErrorIs(t, err, context.Canceled)

		})
	}
}

func TestQuantizedHNSWInt8FP16FallbackCrossesRejectedBridge(t *testing.T) {
	for _, kind := range []Quantization{QuantizationInt8, QuantizationFP16} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {

			const count = DefaultHNSWBruteForceThreshold + 1
			base := &HNSWIndex{
				dimension: 1, options: HNSWBuildOptions{Metric: MetricL2, M: 2, EFConstruction: 3},
				keys: make([]uint64, count), vectors: make([]float32, count),
				neighbors: make([][][]int, count), levels: make([]int, count),
			}
			for j := range count {
				base.keys[j] = uint64(j + 1)
				base.neighbors[j] = [][]int{nil}
			}
			base.vectors[0], base.vectors[1] = 1, 10
			base.neighbors[0][0], base.neighbors[1][0] = []int{1}, []int{2}
			index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, kind, nil)
			require.NoError(t, err)
			for _, options := range []HNSWSearchOptions{
				{SearchOptions: SearchOptions{TopK: 1, Radius: .5}, EF: 1},
				{SearchOptions: SearchOptions{TopK: 1, Filter: func(key uint64) bool { return key == 3 }}, EF: 1},
				{SearchOptions: SearchOptions{TopK: 1}, EF: maxBlockHeapSearchCapacity + 1},
			} {
				got, err := index.SearchHNSW(context.Background(), []float32{0}, options)
				require.NoError(t, err)
				require.Equal(t, []Result{{Key: 3, Score: 0}}, got)
			}

		})
	}
}

func BenchmarkQuantizedHNSWInt8Batch(b *testing.B) {
	const dimension, count = 128, 2000
	candidates := make([]Candidate, count)
	for j := range candidates {
		candidates[j] = Candidate{Key: uint64(j), Vector: make([]float32, dimension)}
		for d := range dimension {
			candidates[j].Vector[d] = float32(math.Sin(float64(j*17 + d*7)))
		}
	}
	buildOptions := DefaultHNSWBuildOptions(MetricL2)
	buildOptions.M, buildOptions.EFConstruction = 15, 100
	builder, err := NewHNSWBuilder(dimension, buildOptions)
	if err != nil {
		b.Fatal(err)
	}
	for _, candidate := range candidates {
		if err := builder.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			b.Fatal(err)
		}
	}
	base, err := builder.Build(context.Background())
	if err != nil {
		b.Fatal(err)
	}
	index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, QuantizationInt8, nil)
	if err != nil {
		b.Fatal(err)
	}
	query := candidates[111].Vector
	for _, batch := range []bool{false, true} {
		b.Run(fmt.Sprintf("batch=%v", batch), func(b *testing.B) {
			options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 100}, EF: 180}
			if !batch {
				options.Filter = func(uint64) bool { return true }
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := index.SearchHNSW(context.Background(), query, options); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// The dual-heap path must preserve scores, tie ordering, rejected bridges and
// stopping decisions when scoring a neighbor list in SIMD batches.
func TestQuantizedHNSWDualHeapBatchMatchesScalar(t *testing.T) {
	const count, dimension = DefaultHNSWBruteForceThreshold + 101, 66
	ctx := context.Background()
	for _, kind := range []Quantization{QuantizationInt4, QuantizationInt8, QuantizationFP16} {
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			t.Run(fmt.Sprintf("kind=%v/metric=%v", kind, metric), func(t *testing.T) {
				base := &HNSWIndex{
					dimension: dimension, options: HNSWBuildOptions{Metric: metric, M: 16, EFConstruction: 300},
					keys: make([]uint64, count), vectors: make([]float32, dimension*count),
					neighbors: make([][][]int, count), levels: make([]int, count),
				}
				for j := range count {
					base.keys[j] = uint64(count - j)
					for d := range dimension {
						// Include duplicate codes, constants, and zero vectors.
						if j%7 != 0 {
							base.vectors[j*dimension+d] = float32(((j%137)*13+d*7)%31 - 15)
						}
					}
					base.neighbors[j] = [][]int{{(j + 1) % count, (j + 1) % count}}
					for n := range 30 {
						base.neighbors[j][0] = append(base.neighbors[j][0], (j*17+n*37)%count)
					}
				}
				index, err := NewScalarQuantizedHNSWIndex(ctx, base, kind, nil)
				require.NoError(t, err)
				visited := acquireHNSWVisited(count)
				defer releaseHNSWVisited(visited)
				for _, query := range [][]float32{make([]float32, dimension), base.vectors[3*dimension : 4*dimension]} {
					code, err := index.vectors.quantizedQuery(query)
					require.NoError(t, err)
					for _, options := range []HNSWSearchOptions{
						{SearchOptions: SearchOptions{TopK: 100}, EF: 300},
						{SearchOptions: SearchOptions{TopK: 17, Filter: func(key uint64) bool { return key%3 == 0 }}, EF: 17},
						{SearchOptions: SearchOptions{TopK: 100, Filter: func(key uint64) bool { return key%11 == 0 }}, EF: 300},
						{SearchOptions: SearchOptions{TopK: 17, Radius: .5}, EF: 17},
						{SearchOptions: SearchOptions{TopK: 100, Radius: .5}, EF: 300},
					} {
						scoreAt := func(position int) (float32, error) {
							return QuantizedDistance(metric, index.vectors.codes[position], code)
						}
						want, err := index.searchBase(ctx, 0, options.EF, options, scoreAt, nil, visited)
						require.NoError(t, err)
						got, err := index.searchBase(ctx, 0, options.EF, options, scoreAt, &code, visited)
						require.NoError(t, err)
						require.Equal(t, want, got)
						results, err := index.SearchHNSW(ctx, query, options)
						require.NoError(t, err)
						wantResults := make([]Result, min(len(want), options.TopK))
						for j := range wantResults {
							wantResults[j] = Result{Key: base.keys[want[j].position], Score: want[j].score}
						}
						require.Equal(t, wantResults, results)
					}
				}
			})
		}
	}
}
