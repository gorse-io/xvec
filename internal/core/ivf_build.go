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
	"slices"
	"sync"
)

const (
	DefaultIVFNList       = 1024
	DefaultIVFNIterations = 10
)

var (
	ErrInvalidIVFOptions = errors.New("core: invalid IVF build options")
	ErrInvalidIVFList    = errors.New("core: invalid IVF list")
)

// IVFBuildOptions configures centroid training and deterministic list
// assignment. Quantization is layered onto the built layout separately.
type IVFBuildOptions struct {
	Metric      Metric
	NList       int
	NIterations int
	Tolerance   float64
	Workers     int
	Seed        uint64
}

// DefaultIVFBuildOptions returns the public baseline defaults.
func DefaultIVFBuildOptions(metric Metric) IVFBuildOptions {
	return IVFBuildOptions{
		Metric:      metric,
		NList:       DefaultIVFNList,
		NIterations: DefaultIVFNIterations,
		Tolerance:   DefaultKMeansTolerance,
	}
}

// Validate checks IVF build invariants.
func (o IVFBuildOptions) Validate() error {
	if !o.Metric.valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidIVFOptions)
	}
	if o.NList <= 0 {
		return fmt.Errorf("%w: NList must be positive", ErrInvalidIVFOptions)
	}
	if o.NIterations <= 0 {
		return fmt.Errorf("%w: NIterations must be positive", ErrInvalidIVFOptions)
	}
	if o.Tolerance < 0 || math.IsNaN(o.Tolerance) || math.IsInf(o.Tolerance, 0) {
		return fmt.Errorf("%w: Tolerance must be finite and non-negative", ErrInvalidIVFOptions)
	}
	return nil
}

// IVFBuilder collects original vectors and builds a one-shot IVF layout. The
// resulting index supports concurrent search and incremental streaming.
type IVFBuilder struct {
	mu        sync.Mutex
	dimension int
	options   IVFBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

// NewIVFBuilder constructs an empty IVF builder.
func NewIVFBuilder(dimension int, options IVFBuildOptions) (*IVFBuilder, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &IVFBuilder{
		dimension: dimension,
		options:   options,
		positions: make(map[uint64]int),
	}, nil
}

// Add clones one finite original vector. Keys remain unique for the builder's
// lifetime.
func (b *IVFBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil IVF builder")
	}
	if ctx == nil {
		return errors.New("core: nil IVF add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate IVF vector: %w", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if b.built {
		return ErrBuilderClosed
	}
	if _, exists := b.positions[key]; exists {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	b.positions[key] = len(b.keys)
	b.keys = append(b.keys, key)
	b.vectors = append(b.vectors, vector...)
	return nil
}

// Build trains at most NList centroids, assigns each vector to its best
// centroid, and transfers builder-owned storage into an immutable index. An
// empty builder produces a valid empty layout without invoking k-means.
func (b *IVFBuilder) Build(ctx context.Context) (*IVFIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil IVF builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	index := &IVFIndex{
		dimension: b.dimension,
		options:   b.options,
		keys:      b.keys,
		vectors:   b.vectors,
		positions: b.positions,
	}
	if len(b.keys) != 0 {
		training := make([][]float32, len(b.keys))
		for position := range b.keys {
			if position&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			start := position * b.dimension
			training[position] = b.vectors[start : start+b.dimension]
		}
		kmeans := DefaultKMeansOptions(b.options.NList, b.options.Metric)
		kmeans.MaxIterations = b.options.NIterations
		kmeans.Tolerance = b.options.Tolerance
		kmeans.Workers = b.options.Workers
		kmeans.Seed = b.options.Seed
		model, err := TrainKMeans(ctx, training, kmeans)
		if err != nil {
			return nil, fmt.Errorf("core: train IVF centroids: %w", err)
		}
		labels, _, err := model.Classify(ctx, training, b.options.Workers)
		if err != nil {
			return nil, fmt.Errorf("core: assign IVF lists: %w", err)
		}
		index.model = model
		index.lists = make([]ivfList, model.Len())
		index.listForPosition = make([]int, len(labels))
		for position, label := range labels {
			if position&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			index.lists[label].positions = append(index.lists[label].positions, position)
			index.listForPosition[position] = label
		}
	}

	b.built = true
	b.keys = nil
	b.vectors = nil
	b.positions = nil
	return index, nil
}

type ivfList struct{ positions []int }

// IVFIndex is the streamable output of IVF construction. It retains original
// vectors for exact refinement and stores list membership by vector position.
type IVFIndex struct {
	mu              sync.RWMutex
	dimension       int
	options         IVFBuildOptions
	model           *KMeansModel
	keys            []uint64
	vectors         []float32
	positions       map[uint64]int
	lists           []ivfList
	listForPosition []int
}

// Dimension returns the fixed vector dimension.
func (i *IVFIndex) Dimension() int {
	if i == nil {
		return 0
	}
	return i.dimension
}

// Metric returns the configured metric.
func (i *IVFIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.options.Metric
}

// Len returns the number of built vectors.
func (i *IVFIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// NList returns the effective number of trained centroids. It is zero for an
// empty index and never exceeds the vector count or configured NList. Duplicate
// samples can still leave an empty assigned list.
func (i *IVFIndex) NList() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.lists)
}

// BuildOptions returns the value-semantic construction settings.
func (i *IVFIndex) BuildOptions() IVFBuildOptions {
	if i == nil {
		return IVFBuildOptions{}
	}
	return i.options
}

// Vector returns a cloned original vector by key.
func (i *IVFIndex) Vector(key uint64) ([]float32, bool) {
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

// Centroids returns a deep copy of trained or online-bootstrapped centroids.
func (i *IVFIndex) Centroids() [][]float32 {
	if i == nil {
		return [][]float32{}
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.model == nil {
		return [][]float32{}
	}
	return i.model.Centroids()
}

// TrainingCost returns the current list-assignment objective. Empty indexes
// return zero.
func (i *IVFIndex) TrainingCost() float64 {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.model == nil {
		return 0
	}
	return i.model.Cost()
}

// TrainingIterations returns completed k-means rounds.
func (i *IVFIndex) TrainingIterations() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.model == nil {
		return 0
	}
	return i.model.Iterations()
}

// TrainingConverged reports whether centroid training stopped on tolerance.
func (i *IVFIndex) TrainingConverged() bool {
	if i == nil {
		return false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.model != nil && i.model.converged
}

// List returns original candidate clones in stable builder insertion order.
func (i *IVFIndex) List(list int) ([]Candidate, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if list < 0 || list >= len(i.lists) {
		return nil, fmt.Errorf("%w: %d", ErrInvalidIVFList, list)
	}
	positions := i.lists[list].positions
	result := make([]Candidate, len(positions))
	for index, position := range positions {
		start := position * i.dimension
		result[index] = Candidate{
			Key:    i.keys[position],
			Vector: slices.Clone(i.vectors[start : start+i.dimension]),
		}
	}
	return result, nil
}

// ListForKey returns the current list containing key.
func (i *IVFIndex) ListForKey(key uint64) (int, bool) {
	if i == nil {
		return 0, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found || position >= len(i.listForPosition) {
		return 0, false
	}
	return i.listForPosition[position], true
}

var _ DenseProvider = (*IVFIndex)(nil)
