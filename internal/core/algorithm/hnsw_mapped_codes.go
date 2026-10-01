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

	mmap "github.com/blevesearch/mmap-go"
)

// ErrScalarQuantizedIndexClosed identifies a search on a released code arena.
var ErrScalarQuantizedIndexClosed = errors.New("core: scalar-quantized index is closed")

func newOwnedScalarQuantizedHNSWIndexWithMmap(ctx context.Context, base *HNSWIndex, kind Quantization, reformer DenseReformer, useMmap bool) (*ScalarQuantizedHNSWIndex, error) {
	var reader DenseVectorReader
	if base.encodedVectors != nil {
		reader = encodedHNSWVectorReader(base.encodedVectors)
	}
	var codes mmap.MMap
	if useMmap && kind == QuantizationFP16 && len(base.keys) != 0 {
		if len(base.keys) > maxPlatformInt()/(base.dimension*2) {
			return nil, fmt.Errorf("%w: scalar codes exceed platform capacity", ErrInvalidQuantizedVector)
		}
		var err error
		codes, err = mmap.MapRegion(nil, len(base.keys)*base.dimension*2, mmap.RDWR, mmap.ANON, 0)
		if err != nil {
			return nil, fmt.Errorf("core: map FP16 code arena: %w", err)
		}
	}
	complete := false
	defer func() {
		if !complete && codes != nil {
			_ = codes.Unmap()
		}
	}()
	vectors, err := newScalarQuantizedVectorStorageWithCodes(ctx, base.dimension, base.options.Metric, kind, reformer, base.keys, base.vectors, base.vectorRows, reader, codes)
	if err != nil {
		return nil, err
	}
	index, err := newScalarQuantizedHNSWWithStorage(ctx, base, vectors)
	if err != nil {
		return nil, err
	}
	vectors.mappedCodes = codes
	complete = true
	return index, nil
}

func (s *scalarQuantizedVectors) lockCodes() error {
	s.codeMu.RLock()
	if s.closed {
		s.codeMu.RUnlock()
		return ErrScalarQuantizedIndexClosed
	}
	return nil
}

// Code publication and Close use the same lock as Flat searches. No reader can
// see a partial arena, and a canceled construction releases its mapping.
func (i *ScalarQuantizedHNSWIndex) ensureCodes(ctx context.Context) error {
	s := i.vectors
	if !s.lazyFP16 {
		return nil
	}
	if err := s.lockCodes(); err != nil {
		return err
	}
	ready := s.codes != nil
	s.codeMu.RUnlock()
	if ready {
		return nil
	}
	if ctx == nil {
		return errors.New("core: nil deferred FP16 context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.codeMu.Lock()
	defer s.codeMu.Unlock()
	if s.closed {
		return ErrScalarQuantizedIndexClosed
	}
	if s.codes != nil {
		return nil
	}
	index, err := newOwnedScalarQuantizedHNSWIndexWithMmap(ctx, i.base, QuantizationFP16, s.reformer, i.deferredMmap)
	if err != nil {
		return err
	}
	s.codes, s.mappedCodes = index.vectors.codes, index.vectors.mappedCodes
	s.fp16Cache = nil
	i.fp16Magnitudes = index.fp16Magnitudes
	return nil
}

// Close releases the code arena after in-flight HNSW and shared Flat searches
// finish. It is idempotent. Subsequent searches through either view return
// ErrScalarQuantizedIndexClosed. Original vectors and topology are unchanged.
func (i *ScalarQuantizedHNSWIndex) Close() error {
	if i == nil || i.vectors == nil {
		return nil
	}
	s := i.vectors
	s.codeMu.Lock()
	defer s.codeMu.Unlock()
	if s.closed {
		return nil
	}
	if s.mappedCodes != nil {
		if err := s.mappedCodes.Unmap(); err != nil {
			return fmt.Errorf("core: unmap FP16 code arena: %w", err)
		}
	}
	s.codes = nil
	s.fp16Cache = nil
	s.closed = true
	return nil
}
