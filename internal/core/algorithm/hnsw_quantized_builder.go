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

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
)

// BuildInt8WithWorkers quantizes the collected vectors before graph insertion.
// Navigation, neighbor selection and reverse-edge pruning all use the same
// INT8 distance as search. A single worker is deterministic; concurrent builds
// may produce different topologies. Originals are retained for refinement and
// the existing on-disk format. This consumes the builder on success.
func (b *HNSWBuilder) BuildInt8WithWorkers(
	ctx context.Context, workers int, reformer DenseReformer,
) (*ScalarQuantizedHNSWIndex, error) {
	var vectors *scalarQuantizedVectors
	base, err := b.buildWithDistance(ctx, workers, func(index *HNSWIndex) (func(int, int) (float32, error), error) {
		if index.fp16 {
			return nil, errors.New("core: INT8 HNSW construction requires an FP32 builder")
		}
		var err error
		vectors, err = newScalarQuantizedVectors(
			ctx, index.dimension, index.options.Metric, QuantizationInt8, reformer, index.keys, index.vectors,
		)
		if err != nil {
			return nil, err
		}
		return func(left, right int) (float32, error) {
			// Codes are immutable and validated during quantization. Reuse
			// the exact SIMD dot product and normal score reconstruction.
			leftCode, rightCode := vectors.codes[left], vectors.codes[right]
			dot := mathutil.InnerProductInt8(leftCode.codes, rightCode.codes)
			return quantizedDistanceFromDot(vectors.metric, leftCode, rightCode, float64(dot))
		}, nil
	})
	if err != nil {
		return nil, err
	}
	// The graph was just built and is exclusively owned; no snapshot or
	// second quantization is needed before publishing the immutable index.
	return &ScalarQuantizedHNSWIndex{base: base, vectors: vectors}, nil
}
