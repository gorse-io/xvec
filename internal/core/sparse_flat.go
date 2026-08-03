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
	"slices"
	"sync"

	"github.com/gorse-io/zvec/internal/ailego"
)

// SparseVector is a canonical sparse FP32 vector. Indices must be strictly
// increasing and Values must be finite.
type SparseVector struct {
	Indices []uint32
	Values  []float32
}

// SparseProvider exposes cloned sparse vectors by document key.
type SparseProvider interface {
	Len() int
	SparseVector(key uint64) (SparseVector, bool)
}

// SparseSearcher is the common sparse exact/ANN search contract.
type SparseSearcher interface {
	SearchSparse(ctx context.Context, query SparseVector, k int) ([]Result, error)
}

// SparseStreamer accepts incremental canonical sparse vectors.
type SparseStreamer interface {
	AddSparse(ctx context.Context, key uint64, vector SparseVector) error
}

// SparseIndex combines sparse provider, search, and streaming capabilities.
type SparseIndex interface {
	SparseProvider
	SparseSearcher
	SparseStreamer
}

// SparseBuilder collects sparse vectors and transfers its index on Build.
type SparseBuilder interface {
	AddSparse(ctx context.Context, key uint64, vector SparseVector) error
	Build(ctx context.Context) (SparseIndex, error)
}

// SparseFlatIndex stores vectors in compressed sparse row form and performs an
// exact inner-product scan. Sparse Flat supports only IP, matching the public
// schema constraints.
type SparseFlatIndex struct {
	mu        sync.RWMutex
	keys      []uint64
	offsets   []int
	indices   []uint32
	values    []float32
	positions map[uint64]int
}

// NewSparseFlatIndex constructs an empty sparse IP index.
func NewSparseFlatIndex(metric Metric) (*SparseFlatIndex, error) {
	if metric != MetricIP {
		return nil, fmt.Errorf("core: sparse Flat supports only inner product, got metric %d", metric)
	}
	return &SparseFlatIndex{
		offsets: []int{0}, positions: make(map[uint64]int),
	}, nil
}

// Len returns the number of indexed sparse vectors.
func (i *SparseFlatIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// AddSparse validates, clones, and appends one canonical sparse vector.
func (i *SparseFlatIndex) AddSparse(ctx context.Context, key uint64, vector SparseVector) error {
	if i == nil {
		return errors.New("core: nil sparse Flat index")
	}
	if ctx == nil {
		return errors.New("core: nil sparse Flat add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := ailego.SparseInnerProduct(vector.Indices, vector.Values, nil, nil); err != nil {
		return fmt.Errorf("core: validate sparse Flat vector: %w", err)
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, exists := i.positions[key]; exists {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	i.positions[key] = len(i.keys)
	i.keys = append(i.keys, key)
	i.indices = append(i.indices, vector.Indices...)
	i.values = append(i.values, vector.Values...)
	i.offsets = append(i.offsets, len(i.indices))
	return nil
}

// SparseVector returns an independent vector copy by key.
func (i *SparseFlatIndex) SparseVector(key uint64) (SparseVector, bool) {
	if i == nil {
		return SparseVector{}, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return SparseVector{}, false
	}
	start, end := i.offsets[position], i.offsets[position+1]
	return SparseVector{
		Indices: slices.Clone(i.indices[start:end]),
		Values:  slices.Clone(i.values[start:end]),
	}, true
}

// SearchSparse evaluates exact inner products and returns highest scores first,
// breaking equal scores by ascending key.
func (i *SparseFlatIndex) SearchSparse(ctx context.Context, query SparseVector, k int) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil sparse Flat index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse Flat search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if k < 0 {
		return nil, errors.New("core: negative top-k")
	}
	if _, err := ailego.SparseInnerProduct(query.Indices, query.Values, nil, nil); err != nil {
		return nil, fmt.Errorf("core: validate sparse Flat query: %w", err)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if k == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}
	if k > len(i.keys) {
		k = len(i.keys)
	}
	worstFirst := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return left.Score < right.Score
	}
	heap := ailego.NewHeap(worstFirst)
	for position, key := range i.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		start, end := i.offsets[position], i.offsets[position+1]
		score, err := ailego.SparseInnerProduct(
			i.indices[start:end], i.values[start:end], query.Indices, query.Values,
		)
		if err != nil {
			return nil, fmt.Errorf("core: score sparse candidate %d (key %d): %w", position, key, err)
		}
		result := Result{Key: key, Score: score}
		if heap.Len() < k {
			heap.Push(result)
			continue
		}
		worst, _ := heap.Peek()
		if resultBetter(MetricIP, result, worst) {
			heap.Replace(result)
		}
	}
	results := heap.Values()
	slices.SortFunc(results, func(left, right Result) int {
		if resultBetter(MetricIP, left, right) {
			return -1
		}
		if resultBetter(MetricIP, right, left) {
			return 1
		}
		return 0
	})
	return results, nil
}

// SparseFlatIndexBuilder is a one-shot sparse Flat builder.
type SparseFlatIndexBuilder struct {
	mu    sync.Mutex
	index *SparseFlatIndex
	built bool
}

// NewSparseFlatBuilder constructs a sparse IP builder.
func NewSparseFlatBuilder(metric Metric) (*SparseFlatIndexBuilder, error) {
	index, err := NewSparseFlatIndex(metric)
	if err != nil {
		return nil, err
	}
	return &SparseFlatIndexBuilder{index: index}, nil
}

// AddSparse appends while the builder is open.
func (b *SparseFlatIndexBuilder) AddSparse(ctx context.Context, key uint64, vector SparseVector) error {
	if b == nil {
		return errors.New("core: nil sparse Flat builder")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return ErrBuilderClosed
	}
	return b.index.AddSparse(ctx, key, vector)
}

// Build closes the builder and returns its streamable sparse index.
func (b *SparseFlatIndexBuilder) Build(ctx context.Context) (SparseIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil sparse Flat builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse Flat build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	b.built = true
	return b.index, nil
}

var (
	_ SparseProvider = (*SparseFlatIndex)(nil)
	_ SparseSearcher = (*SparseFlatIndex)(nil)
	_ SparseStreamer = (*SparseFlatIndex)(nil)
	_ SparseIndex    = (*SparseFlatIndex)(nil)
	_ SparseBuilder  = (*SparseFlatIndexBuilder)(nil)
)
