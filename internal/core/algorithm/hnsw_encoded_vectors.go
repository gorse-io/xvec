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
	"math"
)

// OpenScalarQuantizedHNSWIndexWithEncodedVectors retains immutable little-endian
// FP32 originals supplied by the collection. It verifies every byte against the
// persisted artifact, then builds scalar codes using reusable decoding buffers.
// The caller must keep the original bytes immutable and alive for the index's
// lifetime. The map itself is not retained. No reference to the temporary
// artifact mapping is retained. With useMmap, FP16 codes use anonymous mapped
// memory; the caller must call Close when the index and its Flat view are idle.
func OpenScalarQuantizedHNSWIndexWithEncodedVectors(ctx context.Context, path string, kind Quantization, reformer DenseReformer, originals map[uint64][]byte, useMmap bool) (*ScalarQuantizedHNSWIndex, error) {
	return openScalarQuantizedHNSWEncoded(ctx, path, kind, reformer, originals, useMmap, false)
}

// OpenScalarQuantizedHNSWIndexWithDeferredFP16Codes fully verifies the artifact
// and originals, but keeps only one FP16 row during validation. Its Flat view
// encodes filtered candidates into query-local scratch. The first HNSW search
// materializes the shared code arena; canceled initialization can be retried.
func OpenScalarQuantizedHNSWIndexWithDeferredFP16Codes(ctx context.Context, path string, reformer DenseReformer, originals map[uint64][]byte, useMmap bool) (*ScalarQuantizedHNSWIndex, error) {
	return openScalarQuantizedHNSWEncoded(ctx, path, QuantizationFP16, reformer, originals, useMmap, true)
}

func openScalarQuantizedHNSWEncoded(ctx context.Context, path string, kind Quantization, reformer DenseReformer, originals map[uint64][]byte, useMmap, deferred bool) (*ScalarQuantizedHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil encoded HNSW context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if originals == nil {
		return nil, errors.New("core: nil encoded HNSW originals")
	}
	base, err := openHNSWIndexWithStorage(ctx, path, nil, originals, useMmap, false)
	if err != nil {
		return nil, err
	}
	if deferred {
		vectors, err := newScalarQuantizedVectorStorageWithCodesMode(ctx, base.dimension, base.options.Metric, kind, reformer, base.keys, nil, nil, encodedHNSWVectorReader(base.encodedVectors), nil, true)
		if err != nil {
			return nil, err
		}
		return &ScalarQuantizedHNSWIndex{base: base, vectors: vectors, deferredMmap: useMmap}, nil
	}
	return newOwnedScalarQuantizedHNSWIndexWithMmap(ctx, base, kind, reformer, useMmap)
}

// OpenHNSWIndexWithEncodedOriginals verifies the artifact against encoded
// collection originals and retains one owned contiguous FP32 scoring array.
// The collection can avoid a second decoded copy without changing search locality.
// Neither the supplied originals nor the temporary artifact mapping are retained.
func OpenHNSWIndexWithEncodedOriginals(ctx context.Context, path string, originals map[uint64][]byte, useMmap bool) (*HNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil encoded HNSW context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if originals == nil {
		return nil, errors.New("core: nil encoded HNSW originals")
	}
	return openHNSWIndexWithStorage(ctx, path, nil, originals, useMmap, true)
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
