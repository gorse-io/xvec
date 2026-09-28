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
)

// BuildVamanaWithBorrowedVectors builds an index over immutable FP32 rows.
// The caller must keep the rows immutable and alive for the index's lifetime.
// The candidate slice itself is not retained. Add uses copy-on-write storage.
func BuildVamanaWithBorrowedVectors(ctx context.Context, dimension int, options VamanaBuildOptions, candidates []Candidate, workers int) (*VamanaIndex, error) {
	builder, err := newVamanaBuilderWithBorrowedRows(ctx, dimension, options, candidates)
	if err != nil {
		return nil, err
	}
	return builder.BuildInterleavedWithWorkers(ctx, workers)
}

// BuildScalarQuantizedVamanaWithBorrowedVectors builds and quantizes immutable
// FP32 rows without cloning the originals. Ownership matches
// BuildVamanaWithBorrowedVectors; codes and topology are owned by the result.
func BuildScalarQuantizedVamanaWithBorrowedVectors(ctx context.Context, dimension int, options VamanaBuildOptions, candidates []Candidate, workers int, kind Quantization, reformer DenseReformer) (*ScalarQuantizedVamanaIndex, error) {
	if !kind.valid() {
		return nil, ErrInvalidQuantization
	}
	builder, err := newVamanaBuilderWithBorrowedRows(ctx, dimension, options, candidates)
	if err != nil {
		return nil, err
	}
	return builder.BuildScalarQuantizedInterleavedWithWorkers(ctx, workers, kind, reformer)
}

func newVamanaBuilderWithBorrowedRows(ctx context.Context, dimension int, options VamanaBuildOptions, candidates []Candidate) (*VamanaBuilder, error) {
	if ctx == nil {
		return nil, errors.New("core: nil borrowed Vamana context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	builder, err := NewVamanaBuilder(dimension, options)
	if err != nil {
		return nil, err
	}
	if len(candidates) > maxPlatformInt()/dimension {
		return nil, ErrVamanaCapacity
	}
	builder.keys = make([]uint64, len(candidates))
	builder.vectorRows = make([][]float32, len(candidates))
	builder.positions = make(map[uint64]int, len(candidates))
	for position, candidate := range candidates {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := validateTrainingVector(candidate.Vector, dimension); err != nil {
			return nil, err
		}
		if _, duplicate := builder.positions[candidate.Key]; duplicate {
			return nil, fmt.Errorf("%w: %d", ErrDuplicateKey, candidate.Key)
		}
		builder.keys[position], builder.positions[candidate.Key] = candidate.Key, position
		builder.vectorRows[position] = candidate.Vector[:dimension:dimension]
	}
	return builder, nil
}
