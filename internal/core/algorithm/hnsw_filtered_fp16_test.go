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
	"math"
	"testing"

	mathbatch "github.com/gorse-io/xvec/internal/ailego/math_batch"
	"github.com/stretchr/testify/require"
)

func TestScalarQuantizedHNSWFilteredFP16Prefetch(t *testing.T) {
	candidates := quantizedIndexCandidates(DefaultHNSWBruteForceThreshold + 200)
	base := buildDenseHNSWFromCandidates(t, candidates)
	index, err := NewScalarQuantizedHNSWIndex(context.Background(), base, QuantizationFP16, nil)
	require.NoError(t, err)
	for _, radius := range []float32{0, 20} {
		options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20, Radius: radius, Filter: func(key uint64) bool { return key%2 == 0 }}, EF: 120}
		for _, q := range []int{19, 70, 123} {
			want, err := index.SearchHNSW(context.Background(), candidates[q].Vector, options)
			require.NoError(t, err)
			for _, hint := range [][2]uint32{{8, 0}, {8, 1}, {math.MaxUint32, math.MaxUint32}} {
				prefetched := options
				prefetched.PrefetchOffset, prefetched.PrefetchLines = hint[0], hint[1]
				got, err := index.SearchHNSW(context.Background(), candidates[q].Vector, prefetched)
				require.NoError(t, err)
				require.Equal(t, want, got)
			}
		}
	}
}

func TestScalarQuantizedHNSWFP16CosineCacheMatchesUncached(t *testing.T) {
	ctx := context.Background()
	const count = DefaultHNSWBruteForceThreshold + 101
	for _, dimension := range []int{7, 33, 768} {
		base := &HNSWIndex{
			dimension: dimension, options: HNSWBuildOptions{Metric: MetricCosine, M: 16, EFConstruction: 300},
			keys: make([]uint64, count), vectors: make([]float32, count*dimension),
			neighbors: make([][][]int, count), levels: make([]int, count),
		}
		for row := range count {
			base.keys[row] = uint64(count - row + 5000)
			for d := range dimension {
				if row%7 != 0 {
					base.vectors[row*dimension+d] = float32(((row%137)*13+d*7)%31-15) / 16
				}
			}
			base.neighbors[row] = make([][]int, 1)
			for n := range 32 {
				base.neighbors[row][0] = append(base.neighbors[row][0], (row+n*37+1)%count)
			}
		}
		index, err := NewScalarQuantizedHNSWIndex(ctx, base, QuantizationFP16, nil)
		require.NoError(t, err)
		if mathbatch.FP16CosineCacheCompatible() {
			require.Len(t, index.fp16Magnitudes, count)
			require.Equal(t, float32(0), index.fp16Magnitudes[0])
		} else {
			require.Nil(t, index.fp16Magnitudes)
		}
		uncached := *index
		uncached.fp16Magnitudes = nil
		for _, query := range [][]float32{make([]float32, dimension), base.vectors[3*dimension : 4*dimension]} {
			for _, filter := range []CandidateFilter{nil, func(key uint64) bool { return key%3 == 0 }} {
				options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 100, Filter: filter}, EF: 300, PrefetchOffset: 8}
				want, err := uncached.SearchHNSW(ctx, query, options)
				require.NoError(t, err)
				got, err := index.SearchHNSW(ctx, query, options)
				require.NoError(t, err)
				require.Equal(t, want, got, "dimension=%d", dimension)
			}
		}
	}
}
