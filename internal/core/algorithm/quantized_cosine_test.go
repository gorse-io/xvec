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
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIntegerCosineIndexesUseNormalizedInnerProduct(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []Quantization{QuantizationInt4, QuantizationInt8} {
		for _, rotate := range []bool{false, true} {
			t.Run(fmt.Sprintf("kind=%d/rotate=%v", kind, rotate), func(t *testing.T) {
				candidates := []Candidate{
					{Key: 1, Vector: []float32{7, -3, 2, .13}},
					{Key: 2, Vector: []float32{.3, 9, -.4, 2}},
					{Key: 3, Vector: []float32{0, 0, 0, 0}},
					{Key: 4, Vector: []float32{5, 5, 5, 5}},
					{Key: 5, Vector: []float32{-2, -2, -2, -2}},
				}
				var reformer DenseReformer
				if rotate {
					rotator, err := NewFHTRotatorFromSigns(4, []byte{0x13, 0x57, 0x9b, 0xdf})
					require.NoError(t, err)
					reformer, err = NewRotationReformer(rotator)
					require.NoError(t, err)
				}
				index, err := NewScalarQuantizedFlatIndex(ctx, 4, MetricCosine, kind, reformer, candidates)
				require.NoError(t, err)
				for _, query := range [][]float32{{8, -2, .7, .23}, {0, 0, 0, 0}, {3, 3, 3, 3}} {
					untouched := slices.Clone(query)
					queryDecoded := normalizedIntegerOracle(t, kind, reformer, query)
					want := make([]Result, len(candidates))
					differsFromCosine := false
					for j, candidate := range candidates {
						decoded := normalizedIntegerOracle(t, kind, reformer, candidate.Vector)
						var dot, leftNorm, rightNorm float64
						for d, v := range decoded {
							dot += float64(v) * float64(queryDecoded[d])
							leftNorm += float64(v) * float64(v)
							rightNorm += float64(queryDecoded[d]) * float64(queryDecoded[d])
						}
						score := 1 - dot
						if leftNorm == 0 && rightNorm == 0 {
							score = 0
						}
						want[j] = Result{Key: candidate.Key, Score: float32(score)}
						if leftNorm*rightNorm > 0 && math.Abs(score-(1-dot/math.Sqrt(leftNorm*rightNorm))) > 1e-5 {
							differsFromCosine = true
						}
						original, found := index.Vector(candidate.Key)
						require.True(t, found)
						require.Equal(t, candidate.Vector, original)
					}
					if slices.ContainsFunc(query, func(v float32) bool { return v != 0 }) {
						require.True(t, differsFromCosine, "fixture must distinguish inner product from re-normalized cosine")
					}
					want = MergeSearchResults(MetricCosine, len(want), want)
					got, err := index.Search(ctx, query, len(candidates))
					require.NoError(t, err)
					for j, result := range got {
						require.Equal(t, want[j].Key, result.Key)
						require.InDelta(t, want[j].Score, result.Score, 2e-6)
					}
					require.Equal(t, untouched, query, "normalizing a query must not mutate the caller's vector")
					filtered, err := index.SearchWithOptions(ctx, query, SearchOptions{TopK: len(candidates), Radius: 1, Filter: func(key uint64) bool { return key != 2 }})
					require.NoError(t, err)
					expected := slices.DeleteFunc(slices.Clone(got), func(r Result) bool { return r.Key == 2 || r.Score > 1 })
					require.Equal(t, expected, filtered)
					refiner, err := NewOriginalVectorRefiner(index, MetricCosine)
					require.NoError(t, err)
					refined, err := RefinedSearch(ctx, index, refiner, query, SearchOptions{TopK: 2}, 10)
					require.NoError(t, err)
					exact, err := TopK(ctx, MetricCosine, query, candidates, 2)
					require.NoError(t, err)
					require.Equal(t, exact, refined)
				}
			})
		}
	}
}

// Decode independently normalized inputs so the oracle never uses the index
// distance path or its cached integer moments.
func normalizedIntegerOracle(t *testing.T, kind Quantization, reformer DenseReformer, input []float32) []float32 {
	t.Helper()
	vector := slices.Clone(input)
	if reformer != nil {
		var err error
		vector, err = reformer.Transform(vector)
		require.NoError(t, err)
	}
	var norm float64
	for _, v := range vector {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for j := range vector {
			vector[j] = float32(float64(vector[j]) / norm)
		}
	}
	code, err := QuantizeVector(kind, vector)
	require.NoError(t, err)
	decoded, err := code.Decode()
	require.NoError(t, err)
	return decoded
}

func TestInt4CosineHNSWSearchAndReopen(t *testing.T) {
	ctx := context.Background()
	candidates := quantizedIndexCandidates(DefaultHNSWBruteForceThreshold + 100)
	options := DefaultHNSWBuildOptions(MetricCosine)
	options.M, options.EFConstruction = 12, 100
	builder, err := NewHNSWBuilder(4, options)
	require.NoError(t, err)
	for _, candidate := range candidates {
		require.NoError(t, builder.Add(ctx, candidate.Key, candidate.Vector))
	}
	index, err := builder.BuildInt4WithWorkers(ctx, 1, nil)
	require.NoError(t, err)
	path := t.TempDir() + "/hnsw"
	require.NoError(t, index.Save(ctx, path))
	reopened, err := OpenScalarQuantizedHNSWIndex(ctx, path, QuantizationInt4, nil)
	require.NoError(t, err)
	flat, err := NewScalarQuantizedFlatIndex(ctx, 4, MetricCosine, QuantizationInt4, nil, candidates)
	require.NoError(t, err)
	query := []float32{3.25, -1.5, 7, 2}
	truth, err := flat.Search(ctx, query, 10)
	require.NoError(t, err)
	for _, ef := range []int{180, 300} {
		params := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: ef}
		got, err := index.SearchHNSW(ctx, query, params)
		require.NoError(t, err)
		saved, err := reopened.SearchHNSW(ctx, query, params)
		require.NoError(t, err)
		require.Equal(t, got, saved)
		matched := 0
		for _, result := range got {
			if slices.ContainsFunc(truth, func(r Result) bool { return r.Key == result.Key }) {
				matched++
			}
		}
		require.GreaterOrEqual(t, matched, 9)
	}
}
