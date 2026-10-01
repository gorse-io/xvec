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
	"sync"

	mmap "github.com/blevesearch/mmap-go"

	"github.com/gorse-io/xvec/internal/ailego/container"
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

// NewScalarQuantizedFlatIndexWithBorrowedVectors builds codes while referencing
// immutable original rows for refinement. The caller must keep vector contents
// immutable for the lifetime of the index; keys and slice headers are copied.
func NewScalarQuantizedFlatIndexWithBorrowedVectors(ctx context.Context, dimension int, metric Metric, kind Quantization, reformer DenseReformer, candidates []Candidate) (*ScalarQuantizedFlatIndex, error) {
	keys := make([]uint64, len(candidates))
	rows := make([][]float32, len(candidates))
	for position, candidate := range candidates {
		keys[position], rows[position] = candidate.Key, candidate.Vector
	}
	storage, err := newScalarQuantizedVectorStorage(ctx, dimension, metric, kind, reformer, keys, nil, rows)
	if err != nil {
		return nil, err
	}
	return &ScalarQuantizedFlatIndex{vectors: storage}, nil
}

// DenseVectorReader decodes an immutable original vector by position into the
// supplied destination. It must be safe for concurrent reads and remain valid
// for the lifetime of the index. It must not retain the destination.
type DenseVectorReader interface {
	ReadVector(position int, destination []float32) error
}

// NewScalarQuantizedFlatIndexWithVectorReader retains encoded originals through
// a reader and builds codes using bounded decoding/rotation work buffers.
func NewScalarQuantizedFlatIndexWithVectorReader(ctx context.Context, dimension int, metric Metric, kind Quantization, reformer DenseReformer, keys []uint64, reader DenseVectorReader) (*ScalarQuantizedFlatIndex, error) {
	if reader == nil {
		return nil, errors.New("core: nil dense vector reader")
	}
	storage, err := newScalarQuantizedVectorStorageWithReader(ctx, dimension, metric, kind, reformer, slices.Clone(keys), nil, nil, reader)
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
		if err := i.vectors.lockCodes(); err != nil {
			return nil, err
		}
		defer i.vectors.codeMu.RUnlock()

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
	return i.searchWithOptions(ctx, query, options, nil, false)
}

// SearchKeysWithOptions scans only the supplied document keys. Missing keys
// (including nullable vectors) are ignored; duplicates are scored once.
func (i *ScalarQuantizedFlatIndex) SearchKeysWithOptions(ctx context.Context, query []float32, keys []uint64, options SearchOptions) ([]Result, error) {
	return i.searchWithOptions(ctx, query, options, keys, true)
}

func (i *ScalarQuantizedFlatIndex) searchWithOptions(ctx context.Context, query []float32, options SearchOptions, keys []uint64, byKeys bool) ([]Result, error) {
	if i == nil || i.vectors == nil {
		return nil, errors.New("core: nil scalar-quantized Flat index")
	}
	if err := i.vectors.lockCodes(); err != nil {
		return nil, err
	}
	defer i.vectors.codeMu.RUnlock()

	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized search context")
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
	scoreAt := i.vectors.codeScorer(queryCode)
	// Scan codes directly and retain only top-k. Materializing every position
	// and score makes query scratch space grow with the entire collection.
	k := min(options.TopK, len(i.vectors.keys))
	metric := i.vectors.metric
	heap := container.NewHeapWithCapacity(k, func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return metric.Better(right.Score, left.Score)
	})
	visit := func(position int, key uint64) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if options.Filter != nil && !options.Filter(key) {
			return nil
		}
		score, err := scoreAt(position)
		if err != nil {
			return fmt.Errorf("core: score scalar-quantized candidate %d: %w", position, err)
		}
		retainDenseResult(heap, k, metric, options.Radius, Result{Key: key, Score: score})
		return nil
	}
	if byKeys {
		seen := make(map[uint64]struct{}, len(keys))
		for _, key := range keys {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if _, duplicate := seen[key]; duplicate {
				continue
			}
			seen[key] = struct{}{}
			if position, found := i.vectors.positions[key]; found {
				if err := visit(position, key); err != nil {
					return nil, err
				}
			}
		}
	} else {
		for position, key := range i.vectors.keys {
			if err := visit(position, key); err != nil {
				return nil, err
			}
		}
	}
	return MergeSearchResults(metric, k, heap.Values()), nil
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
	if err := i.vectors.lockCodes(); err != nil {
		return nil, err
	}
	defer i.vectors.codeMu.RUnlock()

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
	scoreAt := i.vectors.codeScorer(queryCode)
	accumulator := newGroupAccumulator(i.vectors.metric, options.TopKPerGroup)
	for position, key := range i.vectors.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		score, err := scoreAt(position)
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
	codeMu       sync.RWMutex
	mappedCodes  mmap.MMap
	closed       bool
	lazyFP16     bool // Flat searches encode into query-local scratch until graph traversal needs codes.
	fp16Cache    *fp16CodeCache
	dimension    int
	metric       Metric
	kind         Quantization
	reformer     DenseReformer
	keys         []uint64
	originals    []float32
	originalRows [][]float32
	reader       DenseVectorReader
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
	return newScalarQuantizedVectorStorageWithReader(ctx, dimension, metric, kind, reformer, keys, originals, rows, nil)
}

func newScalarQuantizedVectorStorageWithReader(ctx context.Context, dimension int, metric Metric, kind Quantization, reformer DenseReformer, keys []uint64, originals []float32, rows [][]float32, reader DenseVectorReader) (*scalarQuantizedVectors, error) {
	return newScalarQuantizedVectorStorageWithCodes(ctx, dimension, metric, kind, reformer, keys, originals, rows, reader, nil)
}

func newScalarQuantizedVectorStorageWithCodes(ctx context.Context, dimension int, metric Metric, kind Quantization, reformer DenseReformer, keys []uint64, originals []float32, rows [][]float32, reader DenseVectorReader, fp16Codes []byte) (*scalarQuantizedVectors, error) {
	return newScalarQuantizedVectorStorageWithCodesMode(ctx, dimension, metric, kind, reformer, keys, originals, rows, reader, fp16Codes, false)
}

func newScalarQuantizedVectorStorageWithCodesMode(ctx context.Context, dimension int, metric Metric, kind Quantization, reformer DenseReformer, keys []uint64, originals []float32, rows [][]float32, reader DenseVectorReader, fp16Codes []byte, lazyFP16 bool) (*scalarQuantizedVectors, error) {
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
	if lazyFP16 && (kind != QuantizationFP16 || reader == nil || fp16Codes != nil) {
		return nil, fmt.Errorf("%w: lazy FP16 requires immutable reader storage", ErrInvalidQuantizedVector)
	}
	if reformer != nil && reformer.Dimension() != dimension {
		return nil, fmt.Errorf("%w: reformer has %d, want %d", ErrInvalidDimension, reformer.Dimension(), dimension)
	}
	if len(keys) > maxPlatformInt()/dimension ||
		(reader == nil && rows == nil && len(originals) != len(keys)*dimension) ||
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
		reader:       reader,
		positions:    make(map[uint64]int, len(keys)),
		lazyFP16:     lazyFP16,
	}
	if !lazyFP16 {
		storage.codes = make([]QuantizedVector, len(keys))
	} else {
		storage.fp16Cache = newFP16CodeCache(dimension, len(keys), 16<<20)
	}
	// Keep FP16 codes in one arena rather than one heap object per vector.
	// Each row has a bounded capacity, so appending cannot overwrite another.
	if kind == QuantizationFP16 {
		if len(keys) > maxPlatformInt()/(dimension*2) {
			return nil, fmt.Errorf("%w: scalar codes exceed platform capacity", ErrInvalidQuantizedVector)
		}
		if lazyFP16 {
			// Validate all rows at open, including overflow after reforming, with
			// one reusable row instead of retaining a collection-sized arena.
			fp16Codes = make([]byte, dimension*2)
		} else if fp16Codes == nil {
			fp16Codes = make([]byte, len(keys)*dimension*2)
		} else if len(fp16Codes) != len(keys)*dimension*2 {
			return nil, fmt.Errorf("%w: inconsistent FP16 code storage", ErrInvalidQuantizedVector)
		}
	}
	var decoded, transformedBuffer []float32
	if reader != nil {
		decoded = make([]float32, dimension)
	}
	into, reuseTransform := reformer.(interface {
		TransformInto([]float32, []float32) error
	})
	if reuseTransform {
		transformedBuffer = make([]float32, dimension)
	}
	for position, key := range storage.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, duplicate := storage.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: %d", ErrDuplicateKey, key)
		}
		storage.positions[key] = position
		var vector []float32
		if reader != nil {
			if err := reader.ReadVector(position, decoded); err != nil {
				return nil, fmt.Errorf("core: read scalar-quantized vector %d: %w", position, err)
			}
			vector = decoded
		} else {
			vector = storage.originalAt(position)
		}
		if err := mathutil.ValidateDense(vector, dimension); err != nil {
			return nil, fmt.Errorf("core: validate scalar-quantized vector %d: %w", position, err)
		}
		transformed := vector
		if reformer != nil {
			var err error
			if reuseTransform {
				err = into.TransformInto(vector, transformedBuffer)
				transformed = transformedBuffer
			} else {
				transformed, err = reformer.Transform(vector)
			}
			if err != nil {
				return nil, fmt.Errorf("core: transform scalar-quantized vector %d: %w", position, err)
			}
			if len(transformed) != dimension {
				return nil, fmt.Errorf("%w: transformed vector %d has %d, want %d", ErrInvalidDimension, position, len(transformed), dimension)
			}
		}
		var code QuantizedVector
		var err error
		if kind == QuantizationFP16 {
			if err = mathutil.ValidateDense(transformed, dimension); err == nil {
				start, end := position*dimension*2, (position+1)*dimension*2
				if lazyFP16 {
					start, end = 0, dimension*2
				}
				code, err = quantizeFP16Into(transformed, fp16Codes[start:end:end])
			}
		} else {
			code, err = QuantizeVector(kind, transformed)
		}
		if err != nil {
			return nil, fmt.Errorf("core: quantize vector %d: %w", position, err)
		}
		if !lazyFP16 {
			storage.codes[position] = code
		}
	}
	return storage, nil
}

func (s *scalarQuantizedVectors) vector(key uint64) ([]float32, bool) {
	position, found := s.positions[key]
	if !found {
		return nil, false
	}
	if s.reader != nil {
		vector := make([]float32, s.dimension)
		if err := s.reader.ReadVector(position, vector); err != nil {
			return nil, false
		}
		return vector, true
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

// distanceToCode scores immutable storage against a query validated by
// quantizedQuery. FP16 codes can go directly to the native half kernels.
func (s *scalarQuantizedVectors) distanceToCode(position int, query QuantizedVector) (float32, error) {
	if s.kind == QuantizationFP16 {
		return fp16CodeDistance(s.metric, s.codes[position].codes, query.codes), nil
	}
	return QuantizedDistance(s.metric, s.codes[position], query)
}

// codeScorer is called while codeMu is read-locked. Scratch belongs to this
// search, so concurrent sparse scans share only immutable entries in a bounded cache.
func (s *scalarQuantizedVectors) codeScorer(query QuantizedVector) func(int) (float32, error) {
	if !s.lazyFP16 || s.codes != nil {
		return func(position int) (float32, error) { return s.distanceToCode(position, query) }
	}
	decoded := make([]float32, s.dimension)
	codes := make([]byte, s.dimension*2)
	var transformedBuffer []float32
	into, reusable := s.reformer.(interface {
		TransformInto([]float32, []float32) error
	})
	if reusable {
		transformedBuffer = make([]float32, s.dimension)
	}
	return func(position int) (float32, error) {
		if cached := s.fp16Cache.get(position); cached != nil {
			return fp16CodeDistance(s.metric, cached, query.codes), nil
		}
		if err := s.reader.ReadVector(position, decoded); err != nil {
			return 0, err
		}
		transformed := decoded
		if s.reformer != nil {
			var err error
			if reusable {
				err = into.TransformInto(decoded, transformedBuffer)
				transformed = transformedBuffer
			} else {
				transformed, err = s.reformer.Transform(decoded)
			}
			if err != nil {
				return 0, err
			}
			if len(transformed) != s.dimension {
				return 0, ErrInvalidDimension
			}
		}
		if _, err := quantizeFP16Into(transformed, codes); err != nil {
			return 0, err
		}
		cached := s.fp16Cache.put(position, codes)
		return fp16CodeDistance(s.metric, cached, query.codes), nil
	}
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
