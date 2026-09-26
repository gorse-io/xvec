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
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// OpenScalarQuantizedHNSWIndexWithEncodedVectors retains immutable little-endian
// FP32 originals supplied by the collection. It verifies every byte against the
// persisted artifact, then builds scalar codes using reusable decoding buffers.
// The caller must keep the original bytes immutable and alive for the index's
// lifetime. The map itself is not retained. No reference to the temporary
// artifact mapping is retained.
func OpenScalarQuantizedHNSWIndexWithEncodedVectors(ctx context.Context, path string, kind Quantization, reformer DenseReformer, originals map[uint64][]byte, useMmap bool) (*ScalarQuantizedHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil encoded HNSW context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if originals == nil {
		return nil, errors.New("core: nil encoded HNSW originals")
	}
	base, err := openHNSWIndexWithStorage(ctx, path, nil, originals, useMmap)
	if err != nil {
		return nil, err
	}
	return newOwnedScalarQuantizedHNSWIndex(ctx, base, kind, reformer)
}

// OpenHNSWIndexWithBorrowedVectors verifies the persisted graph while sharing
// immutable collection originals. Keys and row headers are copied; callers must
// retain the vectors unchanged. Add clones them before publishing a new generation.
func OpenHNSWIndexWithBorrowedVectors(ctx context.Context, path string, candidates []Candidate, useMmap bool) (*HNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil borrowed HNSW context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	originals := make(map[uint64][]float32, len(candidates))
	for _, candidate := range candidates {
		if _, exists := originals[candidate.Key]; exists {
			return nil, fmt.Errorf("%w: %d", ErrDuplicateKey, candidate.Key)
		}
		originals[candidate.Key] = candidate.Vector
	}
	return openHNSWIndexWithBorrowedVectors(ctx, path, originals, useMmap)
}

type encodedHNSWVectorReader [][]byte

func (r encodedHNSWVectorReader) ReadVector(position int, destination []float32) error {
	if position < 0 || position >= len(r) || len(r[position]) != len(destination)*4 {
		return errors.New("core: invalid encoded HNSW vector read")
	}
	decodeHNSWVector(r[position], destination)
	return nil
}

func decodeHNSWVector(encoded []byte, destination []float32) {
	for component := range destination {
		destination[component] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[component*4:]))
	}
}
