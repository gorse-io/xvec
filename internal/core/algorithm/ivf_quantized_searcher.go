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

// ScalarQuantizedIVFIndex owns a stable IVF snapshot, scalar codes for list
// scoring, and the original vectors used by an optional exact refiner.
type ScalarQuantizedIVFIndex struct {
	base    *IVFIndex
	vectors *scalarQuantizedVectors
}

// NewScalarQuantizedIVFIndex snapshots base and scalar-quantizes its vectors.
func NewScalarQuantizedIVFIndex(
	ctx context.Context,
	base *IVFIndex,
	kind Quantization,
	reformer DenseReformer,
) (*ScalarQuantizedIVFIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized IVF context")
	}
	if base == nil {
		return nil, errors.New("core: nil IVF source index")
	}
	snapshot, err := base.persistenceSnapshot(ctx)
	if err != nil {
		return nil, err
	}
	vectors, err := newScalarQuantizedVectors(
		ctx, snapshot.dimension, snapshot.options.Metric, kind, reformer, snapshot.keys, snapshot.vectors,
	)
	if err != nil {
		return nil, err
	}
	return &ScalarQuantizedIVFIndex{base: snapshot, vectors: vectors}, nil
}

// Save persists the immutable IVF topology and original vectors. Scalar codes
// are reconstructed deterministically when reopened.
func (i *ScalarQuantizedIVFIndex) Save(ctx context.Context, path string) error {
	if i == nil || i.base == nil {
		return errors.New("core: nil scalar-quantized IVF index")
	}
	return i.base.Save(ctx, path)
}

// OpenScalarQuantizedIVFIndex reopens a persisted IVF topology and restores
// scalar-code scoring.
func OpenScalarQuantizedIVFIndex(ctx context.Context, path string, kind Quantization, reformer DenseReformer) (*ScalarQuantizedIVFIndex, error) {
	base, err := OpenIVFIndex(ctx, path)
	if err != nil {
		return nil, err
	}
	return NewScalarQuantizedIVFIndex(ctx, base, kind, reformer)
}

func (i *ScalarQuantizedIVFIndex) Dimension() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.dimension
}

func (i *ScalarQuantizedIVFIndex) Metric() Metric {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.metric
}

func (i *ScalarQuantizedIVFIndex) Len() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return len(i.vectors.keys)
}

func (i *ScalarQuantizedIVFIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil || i.vectors == nil {
		return nil, false
	}
	return i.vectors.vector(key)
}

func (i *ScalarQuantizedIVFIndex) NList() int {
	if i == nil || i.base == nil {
		return 0
	}
	return len(i.base.lists)
}

func (i *ScalarQuantizedIVFIndex) BuildOptions() IVFBuildOptions {
	if i == nil || i.base == nil {
		return IVFBuildOptions{}
	}
	return i.base.options
}

func (i *ScalarQuantizedIVFIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.search(ctx, query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: k}, NProbe: DefaultIVFNProbe,
	}, false)
}

func (i *ScalarQuantizedIVFIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.search(ctx, query, IVFSearchOptions{
		SearchOptions: options, NProbe: DefaultIVFNProbe,
	}, true)
}

// SearchIVF selects centroids with the original metric and scores vectors in
// the selected lists using scalar codes.
func (i *ScalarQuantizedIVFIndex) SearchIVF(ctx context.Context, query []float32, options IVFSearchOptions) ([]Result, error) {
	return i.search(ctx, query, options, true)
}

func (i *ScalarQuantizedIVFIndex) search(
	ctx context.Context,
	query []float32,
	options IVFSearchOptions,
	requirePositiveTopK bool,
) ([]Result, error) {
	if i == nil || i.base == nil || i.vectors == nil {
		return nil, errors.New("core: nil scalar-quantized IVF index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized IVF search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.NProbe <= 0 {
		return nil, ErrInvalidIVFNProbe
	}
	if requirePositiveTopK {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else {
		if options.TopK < 0 {
			return nil, errors.New("core: negative scalar-quantized IVF top-k")
		}
		if options.Radius < 0 {
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
	queryMagnitude, err := i.base.cosineQueryMagnitude(query)
	if err != nil {
		return nil, fmt.Errorf("core: prepare scalar-quantized IVF query: %w", err)
	}
	lists, err := i.base.probedListsLocked(ctx, query, queryMagnitude, options.NProbe)
	if err != nil {
		return nil, err
	}
	count := 0
	for _, list := range lists {
		count += len(i.base.lists[list].positions)
	}
	positions := make([]int, 0, count)
	for _, list := range lists {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		positions = append(positions, i.base.lists[list].positions...)
	}
	return i.vectors.searchWithCode(ctx, queryCode, options.SearchOptions, positions)
}

var (
	_ DenseProvider      = (*ScalarQuantizedIVFIndex)(nil)
	_ DenseSearcher      = (*ScalarQuantizedIVFIndex)(nil)
	_ DenseQuerySearcher = (*ScalarQuantizedIVFIndex)(nil)
)
