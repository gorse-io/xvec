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
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFP16HNSWBuildUsesQuantizedDistances(t *testing.T) {
	ctx := context.Background()
	const dimension, count = 17, 96
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M, options.EFConstruction = 4, 24
	builder, err := NewHNSWBuilder(dimension, options)
	require.NoError(t, err)
	codes := make([]QuantizedVector, count)
	for n := range count {
		vector := make([]float32, dimension)
		for d := range vector {
			vector[d] = 1 + float32(math.Sin(float64(n*dimension+d)))*0.0001
		}
		require.NoError(t, builder.Add(ctx, uint64(n), vector))
		codes[n], err = QuantizeVector(QuantizationFP16, vector)
		require.NoError(t, err)
	}
	index, err := builder.BuildFP16WithWorkers(ctx, 1, nil)
	require.NoError(t, err)
	reference := make([][][]int, count)
	fp32 := make([][][]int, count)
	for n, level := range index.base.levels {
		reference[n] = make([][]int, level+1)
		fp32[n] = make([][]int, level+1)
	}
	entry, level, err := buildParallelHNSW(ctx, 1, options, index.base.levels, reference,
		func(left, right int) (float32, error) { return QuantizedDistance(MetricL2, codes[left], codes[right]) })
	require.NoError(t, err)
	require.Equal(t, reference, index.base.neighbors)
	require.Equal(t, entry, index.base.entryPoint)
	require.Equal(t, level, index.base.maxLevel)
	_, _, err = buildParallelHNSW(ctx, 1, options, index.base.levels, fp32, index.base.computeDistanceAt)
	require.NoError(t, err)
	require.NotEqual(t, fp32, index.base.neighbors, "fixture detects accidental FP32 construction")
	path := filepath.Join(t.TempDir(), "hnsw")
	require.NoError(t, index.Save(ctx, path))
	reopened, err := OpenScalarQuantizedHNSWIndex(ctx, path, QuantizationFP16, nil)
	require.NoError(t, err)
	require.Equal(t, index.base.neighbors, reopened.base.neighbors)
	require.Equal(t, codes, reopened.vectors.codes)
	require.ErrorIs(t, builder.Add(ctx, 999, make([]float32, dimension)), ErrBuilderClosed)
	_, err = builder.BuildFP16WithWorkers(ctx, 1, nil)
	require.ErrorIs(t, err, ErrBuilderClosed)
}

func TestHNSWBuildBatchScores(t *testing.T) {
	ctx := context.Background()
	for _, dimension := range []int{1, 7, 8, 17, 33} {
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			t.Run(fmt.Sprintf("dim=%d/metric=%v", dimension, metric), func(t *testing.T) {
				candidates := make([]Candidate, 9)
				for j := range candidates {
					candidates[j] = Candidate{Key: uint64(j), Vector: make([]float32, dimension)}
					for d := range dimension {
						if j > 0 {
							candidates[j].Vector[d] = float32(math.Sin(float64(j*17 + d*3)))
						}
					}
				}
				for _, nativeHalf := range []bool{false, true} {
					options := DefaultHNSWBuildOptions(metric)
					builder, err := newHNSWBuilder(dimension, options, nativeHalf)
					require.NoError(t, err)
					for _, c := range candidates {
						require.NoError(t, builder.Add(ctx, c.Key, c.Vector))
					}
					index, err := builder.BuildWithWorkers(ctx, 2)
					require.NoError(t, err)
					scratch := acquireHNSWVisited(len(candidates))
					defer releaseHNSWVisited(scratch)
					for count := range 10 {
						positions := make([]int, count)
						for j := range positions {
							positions[j] = j
						}
						for _, query := range []int{0, 3} {
							require.NoError(t, index.computeBuildDistances(query, positions, scratch))
							for j, position := range positions {
								want, err := index.computeDistanceAt(query, position)
								require.NoError(t, err)
								require.InDelta(t, want, scratch.batchScores[j], 2e-5*max(1, math.Abs(float64(want))))
							}
						}
					}
				}
				flat, err := NewScalarQuantizedFlatIndex(ctx, dimension, metric, QuantizationFP16, nil, candidates)
				require.NoError(t, err)
				storage := flat.vectors
				scorers, err := fp16BuildScorers(ctx, storage)
				require.NoError(t, err)
				scratch := acquireHNSWVisited(len(candidates))
				defer releaseHNSWVisited(scratch)
				for _, query := range []int{0, 3} {
					for count := range 10 {
						positions := make([]int, count)
						for j := range positions {
							positions[j] = j
						}
						require.NoError(t, scorers.batch(query, positions, scratch))
						for j, position := range positions {
							want, err := QuantizedDistance(metric, storage.codes[query], storage.codes[position])
							require.NoError(t, err)
							pair, err := scorers.score(query, position)
							require.NoError(t, err)
							require.InDelta(t, want, pair, 2e-5*max(1, math.Abs(float64(want))))
							require.InDelta(t, want, scratch.batchScores[j], 2e-5*max(1, math.Abs(float64(want))))
						}
					}
				}
			})
		}
	}
}

func TestFP16HNSWBuildValidation(t *testing.T) {
	ctx := context.Background()
	var absent *HNSWBuilder
	_, err := absent.BuildFP16WithWorkers(ctx, 1, nil)
	require.Error(t, err)
	builder, err := NewHNSWBuilder(3, DefaultHNSWBuildOptions(MetricCosine))
	require.NoError(t, err)
	_, err = builder.BuildFP16WithWorkers(nil, 1, nil)
	require.Error(t, err)
	_, err = builder.BuildFP16WithWorkers(ctx, 0, nil)
	require.ErrorIs(t, err, ErrInvalidHNSWWorkers)
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = builder.BuildFP16WithWorkers(canceled, 1, nil)
	require.ErrorIs(t, err, context.Canceled)
	_, err = builder.BuildFP16WithWorkers(ctx, 1, truncatingReformer{dimension: 2})
	require.Error(t, err)
	for _, workers := range []int{1, 4} {
		empty, err := NewHNSWBuilder(3, DefaultHNSWBuildOptions(MetricCosine))
		require.NoError(t, err)
		index, err := empty.BuildFP16WithWorkers(ctx, workers, nil)
		require.NoError(t, err)
		require.Zero(t, index.Len())
	}
	flat, err := NewScalarQuantizedFlatIndex(ctx, 3, MetricCosine, QuantizationFP16, nil, []Candidate{{Key: 1, Vector: []float32{0, 1, 2}}})
	require.NoError(t, err)
	_, err = fp16BuildScorers(canceled, flat.vectors)
	require.ErrorIs(t, err, context.Canceled)
}
