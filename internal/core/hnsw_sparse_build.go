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

// ErrSparseHNSWCapacity reports that another node or coordinate cannot be
// represented by the current platform-native CSR layout.
var ErrSparseHNSWCapacity = errors.New("core: sparse HNSW index capacity exceeded")

// DefaultSparseHNSWBuildOptions returns the pinned HNSW construction defaults
// with the only metric supported by sparse vectors.
func DefaultSparseHNSWBuildOptions() HNSWBuildOptions {
	return DefaultHNSWBuildOptions(MetricIP)
}

// SparseHNSWBuilder collects canonical sparse vectors and constructs one
// deterministic graph.
type SparseHNSWBuilder struct {
	mu        sync.Mutex
	options   HNSWBuildOptions
	keys      []uint64
	offsets   []int
	indices   []uint32
	values    []float32
	positions map[uint64]int
	built     bool
}

// NewSparseHNSWBuilder constructs an empty one-shot sparse HNSW builder.
func NewSparseHNSWBuilder(options HNSWBuildOptions) (*SparseHNSWBuilder, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if options.Metric != MetricIP {
		return nil, fmt.Errorf("%w: sparse HNSW supports only inner product", ErrInvalidHNSWOptions)
	}
	return &SparseHNSWBuilder{
		options:   options,
		offsets:   []int{0},
		positions: make(map[uint64]int),
	}, nil
}

// AddSparse validates and clones one unique canonical vector while the
// builder remains open.
func (b *SparseHNSWBuilder) AddSparse(ctx context.Context, key uint64, vector SparseVector) error {
	if b == nil {
		return errors.New("core: nil sparse HNSW builder")
	}
	if ctx == nil {
		return errors.New("core: nil sparse HNSW add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := ailego.SparseInnerProduct(vector.Indices, vector.Values, nil, nil); err != nil {
		return fmt.Errorf("core: validate sparse HNSW vector: %w", err)
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
	if len(b.keys) == maxPlatformInt() || len(vector.Indices) > maxPlatformInt()-len(b.indices) {
		return ErrSparseHNSWCapacity
	}
	b.positions[key] = len(b.keys)
	b.keys = append(b.keys, key)
	b.indices = append(b.indices, vector.Indices...)
	b.values = append(b.values, vector.Values...)
	b.offsets = append(b.offsets, len(b.indices))
	return nil
}

// Build assigns deterministic levels, inserts nodes in input order, and
// transfers the builder-owned CSR vectors to the resulting graph.
func (b *SparseHNSWBuilder) Build(ctx context.Context) (*SparseHNSWIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil sparse HNSW builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW build context")
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

	index := &SparseHNSWIndex{
		options:    b.options,
		keys:       b.keys,
		offsets:    b.offsets,
		indices:    b.indices,
		values:     b.values,
		positions:  b.positions,
		entryPoint: -1,
		maxLevel:   -1,
		levels:     make([]int, len(b.keys)),
		neighbors:  make([][][]int, len(b.keys)),
	}
	random := splitMix64{state: b.options.Seed}
	for position := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		level := sampleHNSWLevel(&random, b.options.M)
		index.levels[position] = level
		index.neighbors[position] = make([][]int, level+1)
	}
	index.levelRNGState = random.state
	for position := range index.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := index.insertBuiltNode(ctx, position); err != nil {
			return nil, fmt.Errorf("core: construct sparse HNSW node %d: %w", position, err)
		}
	}

	b.built = true
	b.keys = nil
	b.offsets = nil
	b.indices = nil
	b.values = nil
	b.positions = nil
	return index, nil
}

// SparseHNSWIndex stores canonical FP32 sparse vectors in CSR form and a
// bounded multi-layer proximity graph. Readers share one immutable generation
// while additions publish a complete copy-on-write generation.
type SparseHNSWIndex struct {
	streamMu      sync.Mutex
	mu            sync.RWMutex
	options       HNSWBuildOptions
	keys          []uint64
	offsets       []int
	indices       []uint32
	values        []float32
	positions     map[uint64]int
	levels        []int
	neighbors     [][][]int
	entryPoint    int
	maxLevel      int
	levelRNGState uint64
}

// Metric returns inner product, the only supported sparse HNSW metric.
func (i *SparseHNSWIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.options.Metric
}

// Len returns the number of graph nodes.
func (i *SparseHNSWIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// BuildOptions returns the value-semantic construction settings.
func (i *SparseHNSWIndex) BuildOptions() HNSWBuildOptions {
	if i == nil {
		return HNSWBuildOptions{}
	}
	return i.options
}

// SparseVector returns a cloned canonical vector by key.
func (i *SparseHNSWIndex) SparseVector(key uint64) (SparseVector, bool) {
	if i == nil {
		return SparseVector{}, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return SparseVector{}, false
	}
	return i.sparseVectorAt(position).clone(), true
}

// MaxLevel returns the highest occupied graph level, or -1 for an empty graph.
func (i *SparseHNSWIndex) MaxLevel() int {
	if i == nil {
		return -1
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.maxLevel
}

// EntryPoint returns the current top-layer entry key.
func (i *SparseHNSWIndex) EntryPoint() (uint64, bool) {
	if i == nil {
		return 0, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.entryPoint < 0 {
		return 0, false
	}
	return i.keys[i.entryPoint], true
}

// Level returns a node's maximum graph level.
func (i *SparseHNSWIndex) Level(key uint64) (int, bool) {
	if i == nil {
		return 0, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return 0, false
	}
	return i.levels[position], true
}

// Neighbors returns cloned neighbor keys in deterministic selection order.
func (i *SparseHNSWIndex) Neighbors(key uint64, level int) ([]uint64, error) {
	if i == nil {
		return nil, errors.New("core: nil sparse HNSW index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return nil, fmt.Errorf("%w: %d", ErrHNSWKeyNotFound, key)
	}
	if level < 0 || level > i.levels[position] {
		return nil, fmt.Errorf("%w: key %d has maximum level %d, got %d", ErrInvalidHNSWLevel, key, i.levels[position], level)
	}
	positions := i.neighbors[position][level]
	result := make([]uint64, len(positions))
	for offset, neighbor := range positions {
		result[offset] = i.keys[neighbor]
	}
	return result, nil
}

func (i *SparseHNSWIndex) insertBuiltNode(ctx context.Context, position int) error {
	level := i.levels[position]
	if i.entryPoint < 0 {
		i.entryPoint = position
		i.maxLevel = level
		return nil
	}
	entry := i.entryPoint
	query := i.sparseVectorAt(position)
	for currentLevel := i.maxLevel; currentLevel > level; currentLevel-- {
		nearest, err := i.searchLayer(ctx, query, []int{entry}, 1, currentLevel)
		if err != nil {
			return err
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	for currentLevel := min(level, i.maxLevel); currentLevel >= 0; currentLevel-- {
		candidates, err := i.searchLayer(ctx, query, []int{entry}, i.options.EFConstruction, currentLevel)
		if err != nil {
			return err
		}
		selected, err := i.selectNeighbors(ctx, position, candidates, i.maxDegree(currentLevel))
		if err != nil {
			return err
		}
		i.neighbors[position][currentLevel] = selected
		for _, neighbor := range selected {
			if err := i.addReverseEdge(ctx, neighbor, position, currentLevel); err != nil {
				return err
			}
		}
		if len(candidates) != 0 {
			entry = candidates[0].position
		}
	}
	if level > i.maxLevel {
		i.entryPoint = position
		i.maxLevel = level
	}
	return nil
}

func (i *SparseHNSWIndex) searchLayer(ctx context.Context, query SparseVector, entries []int, ef, level int) ([]hnswScoredNode, error) {
	limit := min(ef, len(i.keys))
	if limit <= 0 {
		return []hnswScoredNode{}, nil
	}
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(MetricIP, left, right) }
	worse := func(left, right hnswScoredNode) bool { return hnswNodeBetter(MetricIP, right, left) }
	candidates := ailego.NewHeap(better)
	results := ailego.NewHeap(worse)
	visited := make([]bool, len(i.keys))
	for _, entry := range entries {
		if entry < 0 || entry >= len(i.keys) || i.levels[entry] < level || visited[entry] {
			continue
		}
		score, err := sparseHNSWScore(query, i.sparseVectorAt(entry))
		if err != nil {
			return nil, err
		}
		node := hnswScoredNode{position: entry, score: score}
		visited[entry] = true
		candidates.Push(node)
		results.Push(node)
	}
	for candidates.Len() != 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current, _ := candidates.Pop()
		worst, hasWorst := results.Peek()
		if results.Len() >= limit && hasWorst && hnswNodeBetter(MetricIP, worst, current) {
			break
		}
		for _, neighbor := range i.neighbors[current.position][level] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := sparseHNSWScore(query, i.sparseVectorAt(neighbor))
			if err != nil {
				return nil, err
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = results.Peek()
			if results.Len() < limit || !hasWorst || hnswNodeBetter(MetricIP, node, worst) {
				candidates.Push(node)
				results.Push(node)
				if results.Len() > limit {
					_, _ = results.Pop()
				}
			}
		}
	}
	result := results.Values()
	slices.SortFunc(result, func(left, right hnswScoredNode) int {
		if hnswNodeBetter(MetricIP, left, right) {
			return -1
		}
		if hnswNodeBetter(MetricIP, right, left) {
			return 1
		}
		return 0
	})
	return result, nil
}

func (i *SparseHNSWIndex) selectNeighbors(ctx context.Context, owner int, candidates []hnswScoredNode, limit int) ([]int, error) {
	selected := make([]int, 0, min(limit, len(candidates)))
	for candidateIndex, candidate := range candidates {
		if candidateIndex&63 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if candidate.position == owner {
			continue
		}
		good := true
		for _, accepted := range selected {
			between, err := sparseHNSWScore(i.sparseVectorAt(candidate.position), i.sparseVectorAt(accepted))
			if err != nil {
				return nil, err
			}
			if !MetricIP.Better(candidate.score, between) {
				good = false
				break
			}
		}
		if good {
			selected = append(selected, candidate.position)
			if len(selected) == limit {
				break
			}
		}
	}
	return selected, nil
}

func (i *SparseHNSWIndex) addReverseEdge(ctx context.Context, owner, neighbor, level int) error {
	current := i.neighbors[owner][level]
	for _, existing := range current {
		if existing == neighbor {
			return nil
		}
	}
	if len(current) < i.maxDegree(level) {
		i.neighbors[owner][level] = append(current, neighbor)
		return nil
	}
	candidates := make([]hnswScoredNode, 0, len(current)+1)
	for _, position := range append(slices.Clone(current), neighbor) {
		score, err := sparseHNSWScore(i.sparseVectorAt(owner), i.sparseVectorAt(position))
		if err != nil {
			return err
		}
		candidates = append(candidates, hnswScoredNode{position: position, score: score})
	}
	slices.SortFunc(candidates, func(left, right hnswScoredNode) int {
		if hnswNodeBetter(MetricIP, left, right) {
			return -1
		}
		if hnswNodeBetter(MetricIP, right, left) {
			return 1
		}
		return 0
	})
	selected, err := i.selectNeighbors(ctx, owner, candidates, i.maxDegree(level))
	if err != nil {
		return err
	}
	i.neighbors[owner][level] = selected
	return nil
}

func (i *SparseHNSWIndex) maxDegree(level int) int {
	if level == 0 {
		return i.options.M * 2
	}
	return i.options.M
}

func (i *SparseHNSWIndex) sparseVectorAt(position int) SparseVector {
	start, end := i.offsets[position], i.offsets[position+1]
	return SparseVector{Indices: i.indices[start:end], Values: i.values[start:end]}
}

func (v SparseVector) clone() SparseVector {
	return SparseVector{Indices: slices.Clone(v.Indices), Values: slices.Clone(v.Values)}
}

func sparseHNSWScore(left, right SparseVector) (float32, error) {
	return ailego.SparseInnerProduct(left.Indices, left.Values, right.Indices, right.Values)
}

var _ SparseProvider = (*SparseHNSWIndex)(nil)
