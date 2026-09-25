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
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/math"
)

// ScalarQuantizedFlatIndex stores original vectors for optional refinement and
// immutable FP16/INT8/INT4 codes for first-stage scoring.
type ScalarQuantizedFlatIndex struct {
	vectors *scalarQuantizedVectors
}

// NewScalarQuantizedFlatIndex validates and owns a scalar-quantized copy of
// candidates. An optional reformer is applied before quantization to both
// stored vectors and queries.
func NewScalarQuantizedFlatIndex(
	ctx context.Context,
	dimension int,
	metric Metric,
	kind Quantization,
	reformer DenseReformer,
	candidates []Candidate,
) (*ScalarQuantizedFlatIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized index context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if len(candidates) > maxPlatformInt()/dimension {
		return nil, fmt.Errorf("%w: vector storage exceeds platform capacity", ErrInvalidQuantizedVector)
	}
	keys := make([]uint64, len(candidates))
	vectors := make([]float32, len(candidates)*dimension)
	for position, candidate := range candidates {
		if len(candidate.Vector) != dimension {
			return nil, fmt.Errorf("%w: candidate %d has %d, want %d", ErrInvalidDimension, position, len(candidate.Vector), dimension)
		}
		keys[position] = candidate.Key
		copy(vectors[position*dimension:(position+1)*dimension], candidate.Vector)
	}
	storage, err := newOwnedScalarQuantizedVectors(ctx, dimension, metric, kind, reformer, keys, vectors)
	if err != nil {
		return nil, err
	}
	return &ScalarQuantizedFlatIndex{vectors: storage}, nil
}

func (i *ScalarQuantizedFlatIndex) Dimension() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.dimension
}

func (i *ScalarQuantizedFlatIndex) Metric() Metric {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.metric
}

func (i *ScalarQuantizedFlatIndex) Len() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return len(i.vectors.keys)
}

func (i *ScalarQuantizedFlatIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil || i.vectors == nil {
		return nil, false
	}
	return i.vectors.vector(key)
}

func (i *ScalarQuantizedFlatIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	if k < 0 {
		return nil, errors.New("core: negative scalar-quantized Flat top-k")
	}
	if k == 0 {
		if i == nil || i.vectors == nil {
			return nil, errors.New("core: nil scalar-quantized Flat index")
		}
		if ctx == nil {
			return nil, errors.New("core: nil scalar-quantized Flat search context")
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := i.vectors.validateQuery(query); err != nil {
			return nil, err
		}
		return []Result{}, nil
	}
	return i.SearchWithOptions(ctx, query, SearchOptions{TopK: k})
}

func (i *ScalarQuantizedFlatIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	if i == nil || i.vectors == nil {
		return nil, errors.New("core: nil scalar-quantized Flat index")
	}
	positions := make([]int, len(i.vectors.keys))
	for position := range positions {
		positions[position] = position
	}
	return i.vectors.search(ctx, query, options, positions)
}

// SearchGroups scans scalar codes and retains the best candidates inside each
// resolved group.
func (i *ScalarQuantizedFlatIndex) SearchGroups(
	ctx context.Context,
	query []float32,
	options GroupByOptions,
) ([]GroupResult, error) {
	if i == nil || i.vectors == nil {
		return nil, errors.New("core: nil scalar-quantized Flat index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized Flat group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	queryCode, err := i.vectors.quantizedQuery(query)
	if err != nil {
		return nil, err
	}
	accumulator := newGroupAccumulator(i.vectors.metric, options.TopKPerGroup)
	for position, key := range i.vectors.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		score, err := i.vectors.distanceToCode(position, queryCode)
		if err != nil {
			return nil, fmt.Errorf("core: score scalar-quantized group candidate %d: %w", position, err)
		}
		if !scoreWithinRadius(i.vectors.metric, score, options.Radius) {
			continue
		}
		value, found := options.Resolve(key)
		if !found {
			continue
		}
		accumulator.add(value, Result{Key: key, Score: score})
	}
	return accumulator.finish(options.GroupCount), nil
}

type scalarQuantizedVectors struct {
	dimension    int
	metric       Metric
	kind         Quantization
	reformer     DenseReformer
	keys         []uint64
	originals    []float32
	originalRows [][]float32
	positions    map[uint64]int
	codes        []QuantizedVector
}

// newOwnedScalarQuantizedVectors takes ownership of keys and originals.
// Callers must supply fresh storage or an immutable, privately owned snapshot.
// Public constructors copy caller-owned input before reaching this helper.
func newOwnedScalarQuantizedVectors(
	ctx context.Context,
	dimension int,
	metric Metric,
	kind Quantization,
	reformer DenseReformer,
	keys []uint64,
	originals []float32,
) (*scalarQuantizedVectors, error) {
	return newScalarQuantizedVectorStorage(ctx, dimension, metric, kind, reformer, keys, originals, nil)
}

func newScalarQuantizedVectorStorage(ctx context.Context, dimension int, metric Metric, kind Quantization, reformer DenseReformer, keys []uint64, originals []float32, rows [][]float32) (*scalarQuantizedVectors, error) {
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized index context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if !metric.Valid() {
		return nil, errors.New("core: invalid scalar-quantized index metric")
	}
	if !kind.valid() {
		return nil, ErrInvalidQuantization
	}
	if reformer != nil && reformer.Dimension() != dimension {
		return nil, fmt.Errorf("%w: reformer has %d, want %d", ErrInvalidDimension, reformer.Dimension(), dimension)
	}
	if len(keys) > maxPlatformInt()/dimension ||
		(rows == nil && len(originals) != len(keys)*dimension) ||
		(rows != nil && (len(rows) != len(keys) || len(originals) != 0)) {
		return nil, fmt.Errorf("%w: inconsistent vector storage", ErrInvalidQuantizedVector)
	}
	storage := &scalarQuantizedVectors{
		dimension:    dimension,
		metric:       metric,
		kind:         kind,
		reformer:     reformer,
		keys:         keys,
		originals:    originals,
		originalRows: rows,
		positions:    make(map[uint64]int, len(keys)),
		codes:        make([]QuantizedVector, len(keys)),
	}
	for position, key := range storage.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, duplicate := storage.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: %d", ErrDuplicateKey, key)
		}
		storage.positions[key] = position
		vector := storage.originalAt(position)
		if err := mathutil.ValidateDense(vector, dimension); err != nil {
			return nil, fmt.Errorf("core: validate scalar-quantized vector %d: %w", position, err)
		}
		transformed := vector
		if reformer != nil {
			var err error
			transformed, err = reformer.Transform(vector)
			if err != nil {
				return nil, fmt.Errorf("core: transform scalar-quantized vector %d: %w", position, err)
			}
			if len(transformed) != dimension {
				return nil, fmt.Errorf("%w: transformed vector %d has %d, want %d", ErrInvalidDimension, position, len(transformed), dimension)
			}
		}
		code, err := QuantizeVector(kind, transformed)
		if err != nil {
			return nil, fmt.Errorf("core: quantize vector %d: %w", position, err)
		}
		storage.codes[position] = code
	}
	return storage, nil
}

func (s *scalarQuantizedVectors) vector(key uint64) ([]float32, bool) {
	position, found := s.positions[key]
	if !found {
		return nil, false
	}
	return slices.Clone(s.originalAt(position)), true
}

func (s *scalarQuantizedVectors) originalAt(position int) []float32 {
	if s.originalRows != nil {
		return s.originalRows[position]
	}
	start := position * s.dimension
	return s.originals[start : start+s.dimension]
}

func (s *scalarQuantizedVectors) validateQuery(query []float32) error {
	if len(query) != s.dimension {
		return fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), s.dimension)
	}
	if err := mathutil.ValidateDense(query, s.dimension); err != nil {
		return fmt.Errorf("core: validate scalar-quantized query: %w", err)
	}
	return nil
}

func (s *scalarQuantizedVectors) quantizedQuery(query []float32) (QuantizedVector, error) {
	if err := s.validateQuery(query); err != nil {
		return QuantizedVector{}, err
	}
	transformed := query
	var err error
	if s.reformer != nil {
		transformed, err = s.reformer.Transform(query)
		if err != nil {
			return QuantizedVector{}, fmt.Errorf("core: transform scalar-quantized query: %w", err)
		}
		if len(transformed) != s.dimension {
			return QuantizedVector{}, fmt.Errorf("%w: transformed query has %d, want %d", ErrInvalidDimension, len(transformed), s.dimension)
		}
	}
	code, err := QuantizeVector(s.kind, transformed)
	if err != nil {
		return QuantizedVector{}, fmt.Errorf("core: quantize query: %w", err)
	}
	return code, nil
}

func (s *scalarQuantizedVectors) search(ctx context.Context, query []float32, options SearchOptions, positions []int) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	queryCode, err := s.quantizedQuery(query)
	if err != nil {
		return nil, err
	}
	return s.searchWithCode(ctx, queryCode, options, positions)
}

// distanceToCode scores immutable storage against a query validated by
// quantizedQuery. FP16 codes can go directly to the native half kernels.
func (s *scalarQuantizedVectors) distanceToCode(position int, query QuantizedVector) (float32, error) {
	if s.kind == QuantizationFP16 {
		return fp16CodeDistance(s.metric, s.codes[position].codes, query.codes), nil
	}
	return QuantizedDistance(s.metric, s.codes[position], query)
}

func (s *scalarQuantizedVectors) searchWithCode(ctx context.Context, queryCode QuantizedVector, options SearchOptions, positions []int) ([]Result, error) {
	accepted := make([]Result, 0, min(options.TopK, len(positions)))
	for _, position := range positions {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if position < 0 || position >= len(s.keys) {
			return nil, fmt.Errorf("%w: candidate position %d", ErrInvalidQuantizedVector, position)
		}
		key := s.keys[position]
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		score, err := s.distanceToCode(position, queryCode)
		if err != nil {
			return nil, fmt.Errorf("core: score scalar-quantized candidate %d: %w", position, err)
		}
		if scoreWithinRadius(s.metric, score, options.Radius) {
			accepted = append(accepted, Result{Key: key, Score: score})
		}
	}
	return MergeSearchResults(s.metric, options.TopK, accepted), nil
}

var (
	_ DenseProvider      = (*ScalarQuantizedFlatIndex)(nil)
	_ DenseSearcher      = (*ScalarQuantizedFlatIndex)(nil)
	_ DenseQuerySearcher = (*ScalarQuantizedFlatIndex)(nil)
	_ DenseGroupSearcher = (*ScalarQuantizedFlatIndex)(nil)
)
