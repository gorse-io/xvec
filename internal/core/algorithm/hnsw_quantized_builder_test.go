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
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInt8HNSWBuildUsesQuantizedDistances(t *testing.T) {
	const count = 96
	ctx := context.Background()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		for _, rotate := range []bool{false, true} {
			t.Run(fmt.Sprintf("metric=%d/rotate=%v", metric, rotate), func(t *testing.T) {
				dimension := 33
				if rotate {
					dimension = 32
				}
				options := DefaultHNSWBuildOptions(metric)
				options.M, options.EFConstruction = 4, 24
				var reformer DenseReformer
				if rotate {
					rotator, err := NewFHTRotatorFromSigns(dimension, []byte{1, 22, 43, 64, 85, 106, 127, 148, 169, 190, 211, 232, 253, 18, 39, 60})
					require.NoError(t, err)
					reformer, err = NewRotationReformer(rotator)
					require.NoError(t, err)
				}
				builder, err := NewHNSWBuilder(dimension, options)
				require.NoError(t, err)
				codes := make([]QuantizedVector, count)
				for n := range count {
					vector := make([]float32, dimension)
					for d := range vector {
						vector[d] = float32(math.Sin(float64(n*dimension + d)))
					}
					// FP32 distinguishes these vectors; INT8 loses the small components.
					vector[0], vector[1] = -1000, 1000
					require.NoError(t, builder.Add(ctx, uint64(n), vector))
					transformed := vector
					if reformer != nil {
						transformed, err = reformer.Transform(vector)
						require.NoError(t, err)
					}
					codes[n], err = QuantizeVector(QuantizationInt8, transformed)
					require.NoError(t, err)
				}
				index, err := builder.BuildInt8WithWorkers(ctx, 1, reformer)
				require.NoError(t, err)
				reference := make([][][]int, count)
				for n, level := range index.base.levels {
					reference[n] = make([][]int, level+1)
				}
				entry, level, err := buildParallelHNSW(ctx, 1, options, index.base.levels, reference,
					func(left, right int) (float32, error) { return QuantizedDistance(metric, codes[left], codes[right]) })
				require.NoError(t, err)
				require.Equal(t, reference, index.base.neighbors)
				require.Equal(t, entry, index.base.entryPoint)
				require.Equal(t, level, index.base.maxLevel)
				require.Equal(t, codes, index.vectors.codes)
				path := filepath.Join(t.TempDir(), "quantized.hnsw")
				require.NoError(t, index.Save(ctx, path))
				reopened, err := OpenScalarQuantizedHNSWIndex(ctx, path, QuantizationInt8, reformer)
				require.NoError(t, err)
				require.Equal(t, codes, reopened.vectors.codes)
				require.Equal(t, index.base.neighbors, reopened.base.neighbors)
				assertHNSWGraphInvariants(t, index.base)
				if metric == MetricL2 && !rotate {
					fp32 := make([][][]int, count)
					for n, level := range index.base.levels {
						fp32[n] = make([][]int, level+1)
					}
					_, _, err = buildParallelHNSW(ctx, 1, options, index.base.levels, fp32, index.base.computeDistanceAt)
					require.NoError(t, err)
					require.NotEqual(t, fp32, index.base.neighbors, "fixture must detect accidental FP32 construction")
				}
				require.ErrorIs(t, builder.Add(ctx, 999, make([]float32, dimension)), ErrBuilderClosed)
				_, err = builder.BuildInt8WithWorkers(ctx, 1, reformer)
				require.ErrorIs(t, err, ErrBuilderClosed)
			})
		}
	}
}

func TestInt4HNSWBuildUsesQuantizedDistances(t *testing.T) {
	const dimension, count = 32, 96
	ctx := context.Background()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M, options.EFConstruction = 4, 24
	builder, err := NewHNSWBuilder(dimension, options)
	require.NoError(t, err)
	codes := make([]QuantizedVector, count)
	for n := range count {
		vector := make([]float32, dimension)
		for d := range vector {
			vector[d] = float32(math.Sin(float64(n*dimension + d)))
		}
		vector[0], vector[1] = -1000, 1000
		require.NoError(t, builder.Add(ctx, uint64(n), vector))
		codes[n], err = QuantizeVector(QuantizationInt4, vector)
		require.NoError(t, err)
	}
	index, err := builder.BuildInt4WithWorkers(ctx, 1, nil)
	require.NoError(t, err)
	reference := make([][][]int, count)
	for n, level := range index.base.levels {
		reference[n] = make([][]int, level+1)
	}
	entry, level, err := buildParallelHNSW(ctx, 1, options, index.base.levels, reference,
		func(left, right int) (float32, error) { return QuantizedDistance(MetricL2, codes[left], codes[right]) })
	require.NoError(t, err)
	require.Equal(t, reference, index.base.neighbors)
	require.Equal(t, entry, index.base.entryPoint)
	require.Equal(t, level, index.base.maxLevel)
	require.Equal(t, codes, index.vectors.codes)
}

func TestInt8HNSWBuildRecallAndPersistence(t *testing.T) {
	ctx := context.Background()
	const dimension, count = 33, DefaultHNSWBruteForceThreshold + 100
	candidates := make([]Candidate, count)
	for n := range candidates {
		vector := make([]float32, dimension)
		for d := range vector {
			vector[d] = float32(math.Sin(float64((n + 1) * (d + 7))))
		}
		candidates[n] = Candidate{Key: uint64(n), Vector: vector}
	}
	for _, workers := range []int{1, 4} {
		t.Run(fmt.Sprint(workers), func(t *testing.T) {
			options := DefaultHNSWBuildOptions(MetricCosine)
			options.M, options.EFConstruction = 12, 100
			builder, err := NewHNSWBuilder(dimension, options)
			require.NoError(t, err)
			for _, c := range candidates {
				require.NoError(t, builder.Add(ctx, c.Key, c.Vector))
			}
			index, err := builder.BuildInt8WithWorkers(ctx, workers, nil)
			require.NoError(t, err)
			assertHNSWGraphInvariants(t, index.base)
			path := filepath.Join(t.TempDir(), "hnsw")
			require.NoError(t, index.Save(ctx, path))
			reopened, err := OpenScalarQuantizedHNSWIndex(ctx, path, QuantizationInt8, nil)
			require.NoError(t, err)
			require.Equal(t, index.base.neighbors, reopened.base.neighbors)
			flat, err := NewScalarQuantizedFlatIndex(ctx, dimension, MetricCosine, QuantizationInt8, nil, candidates)
			require.NoError(t, err)
			matched := 0
			for q := range 20 {
				query := candidates[q*43].Vector
				options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 10}, EF: 180}
				got, err := index.SearchHNSW(ctx, query, options)
				require.NoError(t, err)
				saved, err := reopened.SearchHNSW(ctx, query, options)
				require.NoError(t, err)
				require.Equal(t, got, saved)
				truth, err := flat.Search(ctx, query, 10)
				require.NoError(t, err)
				for _, result := range got {
					if slices.ContainsFunc(truth, func(want Result) bool { return want.Key == result.Key }) {
						matched++
					}
				}
				original, found := index.Vector(candidates[q*43].Key)
				require.True(t, found)
				require.Equal(t, query, original)
				original[0] = 12345
				originalAgain, _ := index.Vector(candidates[q*43].Key)
				require.Equal(t, query, originalAgain)
				refined, err := index.SearchWithOptions(ctx, query, SearchOptions{TopK: 1})
				require.NoError(t, err)
				require.Equal(t, candidates[q*43].Key, refined[0].Key)
				require.InDelta(t, 0, refined[0].Score, 1e-5)
			}
			require.GreaterOrEqual(t, float64(matched)/200, .95)
		})
	}
}

func TestInt8HNSWBuildValidation(t *testing.T) {
	ctx := context.Background()
	options := DefaultHNSWBuildOptions(MetricL2)
	var absent *HNSWBuilder
	_, err := absent.BuildInt8WithWorkers(ctx, 1, nil)
	require.Error(t, err)
	builder, err := NewHNSWBuilder(3, options)
	require.NoError(t, err)
	_, err = builder.BuildInt8WithWorkers(nil, 1, nil)
	require.Error(t, err)
	_, err = builder.BuildInt8WithWorkers(ctx, 0, nil)
	require.ErrorIs(t, err, ErrInvalidHNSWWorkers)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = builder.BuildInt8WithWorkers(cancelled, 1, nil)
	require.ErrorIs(t, err, context.Canceled)
	_, err = builder.BuildInt8WithWorkers(ctx, 1, truncatingReformer{dimension: 2})
	require.Error(t, err)
	require.NoError(t, builder.Add(ctx, 7, []float32{1, 2, 3}))
	_, err = builder.BuildInt8WithWorkers(ctx, 1, truncatingReformer{dimension: 3})
	require.Error(t, err)
	index, err := builder.BuildInt8WithWorkers(ctx, 1, nil)
	require.NoError(t, err)
	require.Equal(t, 1, index.Len())
	for _, workers := range []int{1, 4} {
		empty, err := NewHNSWBuilder(3, options)
		require.NoError(t, err)
		index, err := empty.BuildInt8WithWorkers(ctx, workers, nil)
		require.NoError(t, err)
		require.Zero(t, index.Len())
	}
	oddInt4, err := NewHNSWBuilder(3, options)
	require.NoError(t, err)
	_, err = oddInt4.BuildInt4WithWorkers(ctx, 1, nil)
	require.ErrorIs(t, err, ErrOddInt4Dimension)
	half, err := NewHNSWBuilderFP16(3, options)
	require.NoError(t, err)
	_, err = half.BuildInt8WithWorkers(ctx, 1, nil)
	require.Error(t, err)
}

// Scorer failures must not leave graph locks held or publish a partial update.
func TestInt8HNSWBatchBuildScorerError(t *testing.T) {
	ctx := context.Background()
	failure := errors.New("batch score failed")
	graph := parallelHNSWGraph{
		options:    HNSWBuildOptions{Metric: MetricL2, M: 1},
		levels:     []int{0, 0, 0, 0},
		neighbors:  [][][]int{{{1, 2}}, {{0}}, {{0}}, {{}}},
		nodeLocks:  make([]sync.RWMutex, 4),
		score:      func(left, right int) (float32, error) { return 0, nil },
		scoreBatch: func(query int, positions []int, scratch *hnswVisited) error { return failure },
	}
	scratch := acquireHNSWVisited(4)
	defer releaseHNSWVisited(scratch)
	_, err := graph.searchLayer(ctx, 3, []int{0}, 4, 0, scratch)
	require.ErrorIs(t, err, failure)
	require.True(t, graph.nodeLocks[0].TryLock(), "search error leaked the read lock")
	graph.nodeLocks[0].Unlock()
	err = graph.mergeNeighbors(ctx, 0, []int{3}, 0, scratch)
	require.ErrorIs(t, err, failure)
	require.Equal(t, []int{1, 2}, graph.neighbors[0][0])
	require.True(t, graph.nodeLocks[0].TryLock(), "merge error leaked the write lock")
	graph.nodeLocks[0].Unlock()
}
