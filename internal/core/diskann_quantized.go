// Copyright 2026-present the zvec-go project
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
)

// ScalarQuantizedDiskANNIndex owns original vectors for refinement, scalar
// codes for public first-stage scoring, and a DiskANN graph/PQ representation
// built from the decoded scalar vectors. DiskANN's PQ remains an independent
// traversal encoding controlled by DiskANNBuildOptions.PQChunks.
type ScalarQuantizedDiskANNIndex struct {
	base    *DiskANNIndex
	vectors *scalarQuantizedVectors
}

// NewScalarQuantizedDiskANNIndex builds an immutable DiskANN index after
// applying an optional reformer and FP16/INT8/INT4 scalar quantization. The
// candidates' unmodified vectors remain available through Vector.
func NewScalarQuantizedDiskANNIndex(
	ctx context.Context,
	dimension int,
	options DiskANNBuildOptions,
	kind Quantization,
	reformer DenseReformer,
	candidates []Candidate,
) (*ScalarQuantizedDiskANNIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized DiskANN context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	keys := make([]uint64, len(candidates))
	var originals []float32
	for position, candidate := range candidates {
		if len(candidate.Vector) != dimension {
			return nil, fmt.Errorf("%w: candidate %d has %d, want %d", ErrInvalidDimension, position, len(candidate.Vector), dimension)
		}
		keys[position] = candidate.Key
		originals = append(originals, candidate.Vector...)
	}
	vectors, err := newScalarQuantizedVectors(ctx, dimension, options.Metric, kind, reformer, keys, originals)
	if err != nil {
		return nil, err
	}
	builder, err := NewDiskANNBuilder(dimension, options)
	if err != nil {
		return nil, err
	}
	for position, code := range vectors.codes {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		decoded, err := code.Decode()
		if err != nil {
			return nil, fmt.Errorf("core: decode scalar-quantized DiskANN vector %d: %w", position, err)
		}
		if err := builder.Add(ctx, keys[position], decoded); err != nil {
			return nil, fmt.Errorf("core: add scalar-quantized DiskANN vector %d: %w", position, err)
		}
	}
	base, err := builder.Build(ctx)
	if err != nil {
		return nil, fmt.Errorf("core: build scalar-quantized DiskANN index: %w", err)
	}
	return &ScalarQuantizedDiskANNIndex{base: base, vectors: vectors}, nil
}

// Save persists the DiskANN graph and its traversal representation. Original
// vectors remain in the collection segment and are supplied again on open.
func (i *ScalarQuantizedDiskANNIndex) Save(ctx context.Context, path string) error {
	if i == nil || i.base == nil {
		return errors.New("core: nil scalar-quantized DiskANN index")
	}
	return i.base.Save(ctx, path)
}

// OpenScalarQuantizedDiskANNIndex reopens a persisted DiskANN graph and
// restores public scalar-code scoring from the collection-owned originals.
func OpenScalarQuantizedDiskANNIndex(
	ctx context.Context,
	path string,
	cacheCapacity, workers int,
	kind Quantization,
	reformer DenseReformer,
	candidates []Candidate,
) (*ScalarQuantizedDiskANNIndex, error) {
	return OpenScalarQuantizedDiskANNIndexWithMmap(ctx, path, cacheCapacity, workers, kind, reformer, candidates, false)
}

// OpenScalarQuantizedDiskANNIndexWithMmap reopens a persisted graph through
// the selected random-access reader and restores scalar-code scoring.
func OpenScalarQuantizedDiskANNIndexWithMmap(
	ctx context.Context,
	path string,
	cacheCapacity, workers int,
	kind Quantization,
	reformer DenseReformer,
	candidates []Candidate,
	useMmap bool,
) (*ScalarQuantizedDiskANNIndex, error) {
	base, err := OpenDiskANNIndexWithMmap(ctx, path, cacheCapacity, workers, useMmap)
	if err != nil {
		return nil, err
	}
	keys := make([]uint64, len(candidates))
	originals := make([]float32, 0, len(candidates)*max(1, base.Dimension()))
	for position, candidate := range candidates {
		if len(candidate.Vector) != base.Dimension() {
			_ = base.Close()
			return nil, fmt.Errorf("%w: candidate %d has %d, want %d", ErrInvalidDimension, position, len(candidate.Vector), base.Dimension())
		}
		keys[position] = candidate.Key
		originals = append(originals, candidate.Vector...)
	}
	if len(keys) != base.Len() {
		_ = base.Close()
		return nil, fmt.Errorf("%w: artifact has %d vectors, collection has %d", ErrInvalidQuantizedVector, base.Len(), len(keys))
	}
	vectors, err := newScalarQuantizedVectors(ctx, base.Dimension(), base.Metric(), kind, reformer, keys, originals)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	for _, key := range keys {
		if _, found := base.Vector(key); !found {
			_ = base.Close()
			return nil, fmt.Errorf("%w: artifact is missing key %d", ErrInvalidQuantizedVector, key)
		}
	}
	return &ScalarQuantizedDiskANNIndex{base: base, vectors: vectors}, nil
}

func (i *ScalarQuantizedDiskANNIndex) Dimension() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.dimension
}

func (i *ScalarQuantizedDiskANNIndex) Metric() Metric {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.metric
}

func (i *ScalarQuantizedDiskANNIndex) Len() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return len(i.vectors.keys)
}

func (i *ScalarQuantizedDiskANNIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil || i.vectors == nil {
		return nil, false
	}
	return i.vectors.vector(key)
}

func (i *ScalarQuantizedDiskANNIndex) BuildOptions() DiskANNBuildOptions {
	if i == nil || i.base == nil {
		return DiskANNBuildOptions{}
	}
	return i.base.BuildOptions()
}

func (i *ScalarQuantizedDiskANNIndex) PQChunks() int {
	if i == nil || i.base == nil {
		return 0
	}
	return i.base.PQChunks()
}

func (i *ScalarQuantizedDiskANNIndex) CacheStats() DiskANNCacheStats {
	if i == nil || i.base == nil {
		return DiskANNCacheStats{}
	}
	return i.base.CacheStats()
}

func (i *ScalarQuantizedDiskANNIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.search(ctx, query, DiskANNSearchOptions{
		SearchOptions: SearchOptions{TopK: k}, ListSize: DefaultDiskANNQueryList,
	}, false)
}

func (i *ScalarQuantizedDiskANNIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.search(ctx, query, DiskANNSearchOptions{
		SearchOptions: options, ListSize: DefaultDiskANNQueryList,
	}, true)
}

// SearchDiskANN traverses the graph using DiskANN's internal PQ over decoded
// scalar vectors, then ranks the visited candidates with the public scalar
// quantization kernel.
func (i *ScalarQuantizedDiskANNIndex) SearchDiskANN(ctx context.Context, query []float32, options DiskANNSearchOptions) ([]Result, error) {
	return i.search(ctx, query, options, true)
}

func (i *ScalarQuantizedDiskANNIndex) search(
	ctx context.Context,
	query []float32,
	options DiskANNSearchOptions,
	requirePositiveTopK bool,
) ([]Result, error) {
	if i == nil || i.base == nil || i.vectors == nil {
		return nil, errors.New("core: nil scalar-quantized DiskANN index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized DiskANN search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.ListSize <= 0 || uint64(options.ListSize) > math.MaxUint32 {
		return nil, ErrInvalidDiskANNListSize
	}
	if requirePositiveTopK {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else {
		if options.TopK < 0 {
			return nil, errors.New("core: negative scalar-quantized DiskANN top-k")
		}
		if options.Radius < 0 || math.IsNaN(float64(options.Radius)) || math.IsInf(float64(options.Radius), 0) {
			return nil, ErrInvalidRadius
		}
	}
	queryCode, err := i.vectors.quantizedQuery(query)
	if err != nil {
		return nil, err
	}
	if options.TopK == 0 || len(i.vectors.keys) == 0 {
		return []Result{}, nil
	}
	decodedQuery, err := queryCode.Decode()
	if err != nil {
		return nil, fmt.Errorf("core: decode scalar-quantized DiskANN query: %w", err)
	}
	candidateCount := len(i.vectors.keys)
	if !options.Linear {
		candidateCount = min(candidateCount, max(options.TopK, options.ListSize))
	}
	baseOptions := options
	baseOptions.TopK = candidateCount
	baseOptions.Radius = 0
	baseOptions.Filter = nil
	candidates, err := i.base.SearchDiskANN(ctx, decodedQuery, baseOptions)
	if err != nil {
		return nil, fmt.Errorf("core: search scalar-quantized DiskANN candidates: %w", err)
	}
	positions := make([]int, 0, len(candidates))
	for _, candidate := range candidates {
		position, found := i.vectors.positions[candidate.Key]
		if !found {
			return nil, fmt.Errorf("%w: DiskANN returned unknown key %d", ErrInvalidQuantizedVector, candidate.Key)
		}
		positions = append(positions, position)
	}
	return i.vectors.searchWithCode(ctx, queryCode, options.SearchOptions, positions)
}

// WarmCache delegates to the immutable DiskANN node cache.
func (i *ScalarQuantizedDiskANNIndex) WarmCache(ctx context.Context, count int) (int, error) {
	if i == nil || i.base == nil {
		return 0, errors.New("core: nil scalar-quantized DiskANN index")
	}
	return i.base.WarmCache(ctx, count)
}

// Close releases the underlying DiskANN node artifact. It is idempotent.
func (i *ScalarQuantizedDiskANNIndex) Close() error {
	if i == nil || i.base == nil {
		return nil
	}
	return i.base.Close()
}

var (
	_ DenseProvider      = (*ScalarQuantizedDiskANNIndex)(nil)
	_ DenseSearcher      = (*ScalarQuantizedDiskANNIndex)(nil)
	_ DenseQuerySearcher = (*ScalarQuantizedDiskANNIndex)(nil)
)
