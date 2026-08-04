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
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
)

const (
	DefaultDiskANNMaxDegree    = 100
	DefaultDiskANNBuildList    = 50
	DefaultDiskANNQueryList    = 300
	DefaultDiskANNCacheNodes   = 1024
	DefaultDiskANNMaxOcclusion = 750
)

var (
	ErrInvalidDiskANNOptions = errors.New("core: invalid DiskANN options")
	ErrDiskANNKeyNotFound    = errors.New("core: DiskANN key not found")
	ErrDiskANNClosed         = errors.New("core: DiskANN index is closed")
	ErrDiskANNCapacity       = errors.New("core: DiskANN index capacity exceeded")
)

// DiskANNBuildOptions configures graph construction, product quantization,
// random-read concurrency, and the demand cache.
type DiskANNBuildOptions struct {
	Metric        Metric
	MaxDegree     int
	ListSize      int
	PQChunks      int
	Workers       int
	CacheCapacity int
}

// DefaultDiskANNBuildOptions returns the pinned public construction defaults.
func DefaultDiskANNBuildOptions(metric Metric) DiskANNBuildOptions {
	return DiskANNBuildOptions{
		Metric: metric, MaxDegree: DefaultDiskANNMaxDegree,
		ListSize: DefaultDiskANNBuildList, CacheCapacity: DefaultDiskANNCacheNodes,
	}
}

// Validate checks invariants that do not depend on vector dimension.
func (o DiskANNBuildOptions) Validate() error {
	if !o.Metric.valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidDiskANNOptions)
	}
	if o.MaxDegree <= 0 || o.MaxDegree > MaxVamanaDegree {
		return fmt.Errorf("%w: MaxDegree must be in [1,%d]", ErrInvalidDiskANNOptions, MaxVamanaDegree)
	}
	if o.ListSize <= 0 || uint64(o.ListSize) > math.MaxUint32 {
		return fmt.Errorf("%w: ListSize must fit uint32", ErrInvalidDiskANNOptions)
	}
	if o.PQChunks < 0 {
		return fmt.Errorf("%w: PQChunks cannot be negative", ErrInvalidDiskANNOptions)
	}
	if o.Workers < 0 || o.CacheCapacity < 0 {
		return fmt.Errorf("%w: Workers and CacheCapacity cannot be negative", ErrInvalidDiskANNOptions)
	}
	return nil
}

// DiskANNBuilder collects original vectors for one immutable disk graph.
type DiskANNBuilder struct {
	mu        sync.Mutex
	dimension int
	options   DiskANNBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

func NewDiskANNBuilder(dimension int, options DiskANNBuildOptions) (*DiskANNBuilder, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if options.PQChunks > dimension {
		return nil, fmt.Errorf("%w: PQChunks cannot exceed dimension", ErrInvalidDiskANNOptions)
	}
	return &DiskANNBuilder{dimension: dimension, options: options, positions: make(map[uint64]int)}, nil
}

// Add validates and clones one unique original vector.
func (b *DiskANNBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil DiskANN builder")
	}
	if ctx == nil {
		return errors.New("core: nil DiskANN add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate DiskANN vector: %w", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if b.built {
		return ErrBuilderClosed
	}
	if _, found := b.positions[key]; found {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	if uint64(len(b.keys)) >= math.MaxUint32 || len(b.vectors) > maxPlatformInt()-b.dimension {
		return ErrDiskANNCapacity
	}
	b.positions[key] = len(b.keys)
	b.keys = append(b.keys, key)
	b.vectors = append(b.vectors, vector...)
	return nil
}

// Build constructs a Vamana topology, trains PQ traversal codes, and converts
// the graph to the native sector layout. The returned in-memory ReaderAt index
// has identical search semantics to an index reopened from disk.
func (b *DiskANNBuilder) Build(ctx context.Context) (*DiskANNIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil DiskANN builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}

	graphOptions := DefaultVamanaBuildOptions(b.options.Metric)
	graphOptions.MaxDegree = b.options.MaxDegree
	graphOptions.SearchListSize = max(b.options.ListSize, b.options.MaxDegree)
	graphOptions.MaxOcclusionSize = DefaultDiskANNMaxOcclusion
	graphBuilder, err := NewVamanaBuilder(b.dimension, graphOptions)
	if err != nil {
		return nil, err
	}
	vectors := make([][]float32, len(b.keys))
	for position, key := range b.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		start := position * b.dimension
		vectors[position] = b.vectors[start : start+b.dimension]
		if err := graphBuilder.Add(ctx, key, vectors[position]); err != nil {
			return nil, fmt.Errorf("core: add DiskANN graph vector %d: %w", position, err)
		}
	}
	graph, err := graphBuilder.Build(ctx)
	if err != nil {
		return nil, fmt.Errorf("core: build DiskANN graph: %w", err)
	}

	layout, err := NewDiskANNLayout(b.options.Metric, len(b.keys), b.dimension, b.options.MaxDegree)
	if err != nil {
		return nil, err
	}
	nodes := make([]DiskANNNode, len(b.keys))
	for position := range nodes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		nodes[position] = DiskANNNode{ID: uint32(position), Vector: slices.Clone(vectors[position])}
		nodes[position].Neighbors = make([]uint32, len(graph.neighbors[position]))
		for offset, neighbor := range graph.neighbors[position] {
			nodes[position].Neighbors[offset] = uint32(neighbor)
		}
	}
	nodeArtifact, err := encodeDiskANNNodeFile(ctx, layout, nodes)
	if err != nil {
		return nil, err
	}
	nodeReader, err := OpenDiskANNNodeReader(
		ctx, bytes.NewReader(nodeArtifact), int64(len(nodeArtifact)), b.options.CacheCapacity, b.options.Workers,
	)
	if err != nil {
		return nil, err
	}

	index := &DiskANNIndex{
		dimension: b.dimension, metric: b.options.Metric, options: b.options,
		keys: slices.Clone(b.keys), positions: cloneUint64Positions(b.positions), entryPoint: graph.entryPoint,
		nodes: nodeReader,
	}
	index.traversalMetric = diskANNTraversalMetric(b.options.Metric)
	if len(vectors) != 0 {
		prepared, traversalMetric, err := prepareDiskANNPQVectors(ctx, vectors, b.options.Metric)
		if err != nil {
			return nil, err
		}
		pqOptions := DefaultPQOptions(traversalMetric)
		pqOptions.Chunks, pqOptions.Workers = b.options.PQChunks, b.options.Workers
		model, err := TrainPQ(ctx, prepared, pqOptions)
		if err != nil {
			return nil, fmt.Errorf("core: train DiskANN PQ: %w", err)
		}
		codes, err := model.EncodeBatch(ctx, prepared, b.options.Workers)
		if err != nil {
			return nil, fmt.Errorf("core: encode DiskANN PQ: %w", err)
		}
		index.pq, index.traversalMetric = model, traversalMetric
		index.codes = make([]byte, len(codes)*model.Chunks())
		for position, code := range codes {
			copy(index.codes[position*model.Chunks():], code.codes)
		}
		if b.options.Metric == MetricMIPSL2 {
			index.codeNorms, err = diskANNPQCodeNorms(ctx, model, index.codes, len(codes))
			if err != nil {
				return nil, err
			}
		}
	}
	if err := validateDiskANNIndex(ctx, index); err != nil {
		return nil, err
	}
	b.built = true
	b.keys, b.vectors, b.positions = nil, nil, nil
	return index, nil
}

func prepareDiskANNPQVectors(ctx context.Context, vectors [][]float32, metric Metric) ([][]float32, Metric, error) {
	prepared := make([][]float32, len(vectors))
	traversalMetric := diskANNTraversalMetric(metric)
	for position, vector := range vectors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, err
			}
		}
		prepared[position] = slices.Clone(vector)
		if metric == MetricCosine {
			normalizeRaBitQVector(prepared[position])
		}
	}
	return prepared, traversalMetric, nil
}

func diskANNTraversalMetric(metric Metric) Metric {
	if metric == MetricCosine || metric == MetricMIPSL2 {
		return MetricL2
	}
	return metric
}

func prepareDiskANNPQQuery(query []float32, metric Metric) []float32 {
	prepared := slices.Clone(query)
	if metric == MetricCosine {
		normalizeRaBitQVector(prepared)
	}
	return prepared
}

func diskANNPQCodeNorms(ctx context.Context, model *PQModel, raw []byte, count int) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN PQ norm context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model == nil || model.Chunks() == 0 || count < 0 ||
		uint64(count)*uint64(model.Chunks()) > uint64(maxPlatformInt()) ||
		len(raw) != int(uint64(count)*uint64(model.Chunks())) {
		return nil, ErrInvalidPQModel
	}
	norms := make([]float32, count)
	for position := range norms {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		var norm float64
		code := raw[position*model.Chunks() : (position+1)*model.Chunks()]
		for chunk, centroid := range code {
			start, end := model.chunkOffsets[chunk], model.chunkOffsets[chunk+1]
			pivot := int(centroid)*model.dimension + start
			for component, value := range model.pivots[pivot : pivot+end-start] {
				if component&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				norm += float64(value) * float64(value)
			}
		}
		norms[position] = float32(norm)
	}
	return norms, nil
}
