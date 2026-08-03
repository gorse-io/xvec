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

var (
	ErrInvalidDimension = errors.New("core: invalid vector dimension")
	ErrDuplicateKey     = errors.New("core: duplicate vector key")
	ErrBuilderClosed    = errors.New("core: builder is closed")
)

// DenseProvider exposes vectors without prescribing their storage layout.
// Vector returns an independent copy so callers cannot mutate an index.
type DenseProvider interface {
	Dimension() int
	Len() int
	Vector(key uint64) ([]float32, bool)
}

// DenseSearcher is the common exact/ANN search contract.
type DenseSearcher interface {
	Search(ctx context.Context, query []float32, k int) ([]Result, error)
}

// DenseStreamer accepts incremental vectors after an index is built.
type DenseStreamer interface {
	Add(ctx context.Context, key uint64, vector []float32) error
}

// DenseIndex is the common runtime contract implemented by exact and ANN
// indexes that retain original dense vectors.
type DenseIndex interface {
	DenseProvider
	DenseSearcher
	DenseStreamer
}

// DenseBuilder collects vectors and transfers its index on Build.
type DenseBuilder interface {
	Add(ctx context.Context, key uint64, vector []float32) error
	Build(ctx context.Context) (DenseIndex, error)
}

// DenseFlatIndex stores FP32 vectors contiguously and scans every vector for
// exact search. Adds are serialized; any number of searches may run together.
type DenseFlatIndex struct {
	mu        sync.RWMutex
	dimension int
	metric    Metric
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
}

// NewDenseFlatIndex constructs an empty exact index.
func NewDenseFlatIndex(dimension int, metric Metric) (*DenseFlatIndex, error) {
	if dimension <= 0 {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if !metric.valid() {
		return nil, errors.New("core: invalid metric")
	}
	return &DenseFlatIndex{
		dimension: dimension,
		metric:    metric,
		positions: make(map[uint64]int),
	}, nil
}

// Dimension returns the fixed vector dimension.
func (i *DenseFlatIndex) Dimension() int {
	if i == nil {
		return 0
	}
	return i.dimension
}

// Metric returns the index score metric.
func (i *DenseFlatIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.metric
}

// Len returns the number of indexed vectors.
func (i *DenseFlatIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// Add clones and appends one finite vector. Keys are unique for the lifetime
// of an index so deterministic tie-breaking remains unambiguous.
func (i *DenseFlatIndex) Add(ctx context.Context, key uint64, vector []float32) error {
	if i == nil {
		return errors.New("core: nil dense Flat index")
	}
	if ctx == nil {
		return errors.New("core: nil dense Flat add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(vector) != i.dimension {
		return fmt.Errorf("%w: got %d, want %d", ErrInvalidDimension, len(vector), i.dimension)
	}
	if _, err := ailego.L2Squared(vector, vector); err != nil {
		return fmt.Errorf("core: validate dense Flat vector: %w", err)
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
	i.vectors = append(i.vectors, vector...)
	return nil
}

// Vector returns a cloned vector by key.
func (i *DenseFlatIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil {
		return nil, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return nil, false
	}
	start := position * i.dimension
	return slices.Clone(i.vectors[start : start+i.dimension]), true
}

// Search performs an exact top-k scan. It holds a read lock so the contiguous
// vector storage cannot move while candidate slices are being scored.
func (i *DenseFlatIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil dense Flat index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil dense Flat search context")
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return topKCandidates(ctx, i.metric, query, k, len(i.keys), func(position int) Candidate {
		start := position * i.dimension
		return Candidate{Key: i.keys[position], Vector: i.vectors[start : start+i.dimension]}
	})
}

// DenseFlatIndexBuilder is a one-shot builder. The built index remains
// streamable through DenseFlatIndex.Add.
type DenseFlatIndexBuilder struct {
	mu    sync.Mutex
	index *DenseFlatIndex
	built bool
}

// NewDenseFlatBuilder constructs a builder for an exact dense index.
func NewDenseFlatBuilder(dimension int, metric Metric) (*DenseFlatIndexBuilder, error) {
	index, err := NewDenseFlatIndex(dimension, metric)
	if err != nil {
		return nil, err
	}
	return &DenseFlatIndexBuilder{index: index}, nil
}

// Add appends a vector while the builder is open.
func (b *DenseFlatIndexBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil dense Flat builder")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return ErrBuilderClosed
	}
	return b.index.Add(ctx, key, vector)
}

// Build closes the builder and returns its index. An empty index is valid.
func (b *DenseFlatIndexBuilder) Build(ctx context.Context) (DenseIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil dense Flat builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil dense Flat build context")
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

func (m Metric) valid() bool {
	return m >= MetricL2 && m <= MetricMIPSL2
}

var (
	_ DenseProvider = (*DenseFlatIndex)(nil)
	_ DenseSearcher = (*DenseFlatIndex)(nil)
	_ DenseStreamer = (*DenseFlatIndex)(nil)
	_ DenseIndex    = (*DenseFlatIndex)(nil)
	_ DenseBuilder  = (*DenseFlatIndexBuilder)(nil)
)
