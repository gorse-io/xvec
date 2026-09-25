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
	"slices"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	mathbatch "github.com/gorse-io/xvec/internal/ailego/math_batch"
)

// BuildInt8WithWorkers quantizes the collected vectors before graph insertion.
// Navigation, neighbor selection and reverse-edge pruning all use the same
// INT8 distance as search. A single worker is deterministic; concurrent builds
// may produce different topologies. Originals are retained for refinement and
// the existing on-disk format. This consumes the builder on success.
func (b *HNSWBuilder) BuildInt8WithWorkers(
	ctx context.Context, workers int, reformer DenseReformer,
) (*ScalarQuantizedHNSWIndex, error) {
	return b.buildScalarQuantizedWithWorkers(ctx, workers, QuantizationInt8, reformer)
}

// BuildInt4WithWorkers quantizes the collected vectors before graph insertion.
// It otherwise follows the same ownership and determinism contract as
// BuildInt8WithWorkers.
func (b *HNSWBuilder) BuildInt4WithWorkers(
	ctx context.Context, workers int, reformer DenseReformer,
) (*ScalarQuantizedHNSWIndex, error) {
	return b.buildScalarQuantizedWithWorkers(ctx, workers, QuantizationInt4, reformer)
}

func (b *HNSWBuilder) buildScalarQuantizedWithWorkers(
	ctx context.Context, workers int, kind Quantization, reformer DenseReformer,
) (*ScalarQuantizedHNSWIndex, error) {
	var vectors *scalarQuantizedVectors
	base, err := b.buildWithDistance(ctx, workers, func(index *HNSWIndex) (hnswBuildScorers, error) {
		if index.fp16 {
			return hnswBuildScorers{}, errors.New("core: integer-quantized HNSW construction requires an FP32 builder")
		}
		if kind == QuantizationInt4 && index.dimension%2 != 0 {
			return hnswBuildScorers{}, ErrOddInt4Dimension
		}
		var err error
		vectors, err = newOwnedScalarQuantizedVectors(
			ctx, index.dimension, index.options.Metric, kind, reformer, index.keys, index.vectors,
		)
		if err != nil {
			return hnswBuildScorers{}, err
		}
		return hnswBuildScorers{score: func(left, right int) (float32, error) {
			// Codes are immutable and validated during quantization. Reuse
			// the exact SIMD dot product and normal score reconstruction.
			leftCode, rightCode := vectors.codes[left], vectors.codes[right]
			dot := integerCodeDotInt64(leftCode, rightCode)
			return quantizedDistanceFromDot(vectors.metric, leftCode, rightCode, float64(dot))
		}, batch: func(query int, positions []int, scratch *hnswVisited) error {
			count := len(positions)
			scratch.batchCodes = slices.Grow(scratch.batchCodes[:0], count)[:count]
			scratch.batchCodeDots = slices.Grow(scratch.batchCodeDots[:0], count)[:count]
			scratch.batchScores = slices.Grow(scratch.batchScores[:0], count)[:count]
			for j, position := range positions {
				scratch.batchCodes[j] = vectors.codes[position].codes
			}
			left := vectors.codes[query]
			integerCodeDots(kind, left.codes, scratch.batchCodes, scratch.batchCodeDots)
			for j, position := range positions {
				score, err := quantizedDistanceFromDot(vectors.metric, left, vectors.codes[position], float64(scratch.batchCodeDots[j]))
				if err != nil {
					return err
				}
				scratch.batchScores[j] = score
			}
			return nil
		}}, nil
	})
	if err != nil {
		return nil, err
	}
	// The graph was just built and is exclusively owned; no snapshot or
	// second quantization is needed before publishing the immutable index.
	return &ScalarQuantizedHNSWIndex{base: base, vectors: vectors}, nil
}

func integerCodeDotInt64(left, right QuantizedVector) int64 {
	if left.kind == QuantizationInt8 {
		return mathutil.InnerProductInt8(left.codes, right.codes)
	}
	return mathutil.InnerProductInt4(left.codes, right.codes)
}

func integerCodeDots(kind Quantization, query []byte, candidates [][]byte, output []int64) {
	if kind == QuantizationInt8 {
		mathbatch.InnerProductsInt8(query, candidates, output)
		return
	}
	mathbatch.InnerProductsInt4(query, candidates, output)
}
