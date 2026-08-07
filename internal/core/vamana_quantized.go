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
	"math"
)

// ScalarQuantizedVamanaIndex owns an immutable Vamana topology, original
// vectors for refinement, and scalar codes used for traversal and ranking.
type ScalarQuantizedVamanaIndex struct {
	base    *VamanaIndex
	vectors *scalarQuantizedVectors
}

// NewScalarQuantizedVamanaIndex snapshots base and quantizes every vector
// after applying the optional dimension-preserving reformer.
func NewScalarQuantizedVamanaIndex(
	ctx context.Context,
	base *VamanaIndex,
	kind Quantization,
	reformer DenseReformer,
) (*ScalarQuantizedVamanaIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized Vamana context")
	}
	if base == nil {
		return nil, errors.New("core: nil Vamana source index")
	}
	base.mu.RLock()
	snapshot, err := cloneVamanaIndex(ctx, base)
	base.mu.RUnlock()
	if err != nil {
		return nil, err
	}
	vectors, err := newScalarQuantizedVectors(
		ctx, snapshot.dimension, snapshot.options.Metric, kind, reformer, snapshot.keys, snapshot.vectors,
	)
	if err != nil {
		return nil, err
	}
	return &ScalarQuantizedVamanaIndex{base: snapshot, vectors: vectors}, nil
}

// Save persists the immutable Vamana topology and original vectors. Scalar
// codes are reconstructed deterministically when reopened.
func (i *ScalarQuantizedVamanaIndex) Save(ctx context.Context, path string) error {
	if i == nil || i.base == nil {
		return errors.New("core: nil scalar-quantized Vamana index")
	}
	return i.base.Save(ctx, path)
}

// OpenScalarQuantizedVamanaIndex reopens a persisted topology and restores
// scalar-code scoring.
func OpenScalarQuantizedVamanaIndex(ctx context.Context, path string, kind Quantization, reformer DenseReformer) (*ScalarQuantizedVamanaIndex, error) {
	base, err := OpenVamanaIndex(ctx, path)
	if err != nil {
		return nil, err
	}
	return NewScalarQuantizedVamanaIndex(ctx, base, kind, reformer)
}

func (i *ScalarQuantizedVamanaIndex) Dimension() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.dimension
}

func (i *ScalarQuantizedVamanaIndex) Metric() Metric {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.metric
}

func (i *ScalarQuantizedVamanaIndex) Len() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return len(i.vectors.keys)
}

func (i *ScalarQuantizedVamanaIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil || i.vectors == nil {
		return nil, false
	}
	return i.vectors.vector(key)
}

func (i *ScalarQuantizedVamanaIndex) BuildOptions() VamanaBuildOptions {
	if i == nil || i.base == nil {
		return VamanaBuildOptions{}
	}
	return i.base.options
}

func (i *ScalarQuantizedVamanaIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.search(ctx, query, VamanaSearchOptions{
		SearchOptions: SearchOptions{TopK: k}, EFSearch: DefaultVamanaEFSearch,
		PrefetchOffset: DefaultVamanaPrefetchOffset,
	}, false)
}

func (i *ScalarQuantizedVamanaIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.search(ctx, query, VamanaSearchOptions{
		SearchOptions: options, EFSearch: DefaultVamanaEFSearch,
		PrefetchOffset: DefaultVamanaPrefetchOffset,
	}, true)
}

func (i *ScalarQuantizedVamanaIndex) SearchVamana(ctx context.Context, query []float32, options VamanaSearchOptions) ([]Result, error) {
	return i.search(ctx, query, options, true)
}

func (i *ScalarQuantizedVamanaIndex) search(
	ctx context.Context,
	query []float32,
	options VamanaSearchOptions,
	requirePositiveTopK bool,
) ([]Result, error) {
	if i == nil || i.base == nil || i.vectors == nil {
		return nil, errors.New("core: nil scalar-quantized Vamana index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized Vamana search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.EFSearch <= 0 || options.EFSearch > MaxVamanaEFSearch {
		return nil, ErrInvalidVamanaEF
	}
	if requirePositiveTopK {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else {
		if options.TopK < 0 {
			return nil, errors.New("core: negative scalar-quantized Vamana top-k")
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
	if len(i.vectors.keys) <= DefaultVamanaBruteForceThreshold {
		positions := make([]int, len(i.vectors.keys))
		for position := range positions {
			positions[position] = position
		}
		return i.vectors.searchWithCode(ctx, queryCode, options.SearchOptions, positions)
	}
	scoreAt := func(position int) (float32, error) {
		return QuantizedDistance(i.vectors.metric, i.vectors.codes[position], queryCode)
	}
	prefetch := func(neighbors []int) {
		prefetchQuantizedHNSWNeighbors(i.vectors.codes, neighbors, options.PrefetchOffset, options.PrefetchLines)
	}
	return searchVamanaGraph(ctx, i.vectors.metric, i.vectors.keys, i.base.neighbors, i.base.entryPoint, options, scoreAt, prefetch)
}

var (
	_ DenseProvider      = (*ScalarQuantizedVamanaIndex)(nil)
	_ DenseSearcher      = (*ScalarQuantizedVamanaIndex)(nil)
	_ DenseQuerySearcher = (*ScalarQuantizedVamanaIndex)(nil)
)
