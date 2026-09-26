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
	"slices"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	mathbatch "github.com/gorse-io/xvec/internal/ailego/math_batch"
)

// BuildScalarQuantizedHNSWWithBorrowedVectors constructs an immutable graph
// over collection-owned FP32 vectors. The caller must not modify the vectors
// for the index lifetime. Candidate keys and slice headers are copied.
func BuildScalarQuantizedHNSWWithBorrowedVectors(ctx context.Context, dimension int, options HNSWBuildOptions, kind Quantization, reformer DenseReformer, candidates []Candidate, workers int) (*ScalarQuantizedHNSWIndex, error) {
	if !kind.valid() {
		return nil, ErrInvalidQuantization
	}
	builder, err := newHNSWBuilderWithBorrowedVectors(ctx, dimension, options, candidates)
	if err != nil {
		return nil, err
	}
	return builder.buildScalarQuantizedWithWorkers(ctx, workers, kind, reformer)
}

// BuildHNSWWithBorrowedVectors shares immutable original rows. Keys and row
// headers are copied. Add clones the borrowed generation before modifying it.
func BuildHNSWWithBorrowedVectors(ctx context.Context, dimension int, options HNSWBuildOptions, candidates []Candidate, workers int) (*HNSWIndex, error) {
	builder, err := newHNSWBuilderWithBorrowedVectors(ctx, dimension, options, candidates)
	if err != nil {
		return nil, err
	}
	return builder.BuildWithWorkers(ctx, workers)
}

func newHNSWBuilderWithBorrowedVectors(ctx context.Context, dimension int, options HNSWBuildOptions, candidates []Candidate) (*HNSWBuilder, error) {
	if ctx == nil {
		return nil, errors.New("core: nil borrowed HNSW build context")
	}
	builder, err := NewHNSWBuilder(dimension, options)
	if err != nil {
		return nil, err
	}
	builder.keys = make([]uint64, len(candidates))
	builder.vectorRows = make([][]float32, len(candidates))
	for position, candidate := range candidates {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := mathutil.ValidateDense(candidate.Vector, dimension); err != nil {
			return nil, err
		}
		if _, found := builder.positions[candidate.Key]; found {
			return nil, fmt.Errorf("%w: %d", ErrDuplicateKey, candidate.Key)
		}
		builder.keys[position] = candidate.Key
		builder.positions[candidate.Key] = position
		builder.vectorRows[position] = candidate.Vector[:dimension:dimension]
	}
	return builder, nil
}

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

// BuildFP16WithWorkers quantizes before graph insertion so navigation, neighbor
// selection and reverse-edge pruning all use binary16 scores. Originals remain
// available for refinement and persistence. This consumes the builder on success.
func (b *HNSWBuilder) BuildFP16WithWorkers(ctx context.Context, workers int, reformer DenseReformer) (*ScalarQuantizedHNSWIndex, error) {
	return b.buildScalarQuantizedWithWorkers(ctx, workers, QuantizationFP16, reformer)
}

func (b *HNSWBuilder) buildScalarQuantizedWithWorkers(
	ctx context.Context, workers int, kind Quantization, reformer DenseReformer,
) (*ScalarQuantizedHNSWIndex, error) {
	var vectors *scalarQuantizedVectors
	base, err := b.buildWithDistance(ctx, workers, func(index *HNSWIndex) (hnswBuildScorers, error) {
		if index.fp16 {
			return hnswBuildScorers{}, errors.New("core: scalar-quantized HNSW construction requires an FP32 builder")
		}
		if kind == QuantizationInt4 && index.dimension%2 != 0 {
			return hnswBuildScorers{}, ErrOddInt4Dimension
		}
		var err error
		vectors, err = newScalarQuantizedVectorStorage(
			ctx, index.dimension, index.options.Metric, kind, reformer, index.keys, index.vectors, index.vectorRows,
		)
		if err != nil {
			return hnswBuildScorers{}, err
		}
		if kind == QuantizationFP16 {
			return fp16BuildScorers(ctx, vectors)
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

// Magnitudes are temporary construction state. Persisted codes and public query
// scoring are unchanged; the cache avoids repeated norms during O(M^2) pruning.
func fp16BuildScorers(ctx context.Context, vectors *scalarQuantizedVectors) (hnswBuildScorers, error) {
	metric := vectors.metric
	var magnitudes []float32
	if metric == MetricCosine {
		magnitudes = make([]float32, len(vectors.codes))
		for j, code := range vectors.codes {
			if err := ctx.Err(); err != nil {
				return hnswBuildScorers{}, err
			}
			squared := fp16CodeDistance(MetricIP, code.codes, code.codes)
			magnitudes[j] = float32(math.Sqrt(float64(max(float32(0), squared))))
		}
	}
	return hnswBuildScorers{
		score: func(left, right int) (float32, error) {
			if metric == MetricCosine {
				dot := fp16CodeDistance(MetricIP, vectors.codes[left].codes, vectors.codes[right].codes)
				return buildCosineFromDot(dot, magnitudes[left], magnitudes[right]), nil
			}
			return fp16CodeDistance(metric, vectors.codes[left].codes, vectors.codes[right].codes), nil
		},
		batch: func(query int, positions []int, scratch *hnswVisited) error {
			count := len(positions)
			scratch.batchCodes = slices.Grow(scratch.batchCodes[:0], count)[:count]
			scratch.batchScores = slices.Grow(scratch.batchScores[:0], count)[:count]
			for j, position := range positions {
				scratch.batchCodes[j] = vectors.codes[position].codes
			}
			prefetchQuantizedHNSWNeighbors(vectors.codes, positions, 8, 1)
			scoreMetric := metric
			if metric == MetricCosine {
				scoreMetric = MetricIP
			}
			fp16CodeDistances(scoreMetric, vectors.codes[query].codes, scratch.batchCodes, scratch.batchScores)
			if metric == MetricCosine {
				for j, position := range positions {
					scratch.batchScores[j] = buildCosineFromDot(scratch.batchScores[j], magnitudes[query], magnitudes[position])
				}
			}
			return nil
		},
	}, nil
}

func buildCosineFromDot(dot, leftMagnitude, rightMagnitude float32) float32 {
	if leftMagnitude == 0 && rightMagnitude == 0 {
		return 0
	}
	if leftMagnitude == 0 || rightMagnitude == 0 {
		return 1
	}
	return 1 - min(float32(1), max(float32(-1), dot/(leftMagnitude*rightMagnitude)))
}
