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

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	DefaultHNSWM              = 50
	DefaultHNSWEFConstruction = 500
	MaxHNSWLevel              = 14
	MaxHNSWM                  = 32767
)

var (
	ErrInvalidHNSWOptions = errors.New("core: invalid HNSW build options")
	ErrInvalidHNSWLevel   = errors.New("core: invalid HNSW level")
	ErrHNSWKeyNotFound    = errors.New("core: HNSW key not found")
	ErrHNSWCapacity       = errors.New("core: HNSW index capacity exceeded")
)

// HNSWBuildOptions configures deterministic dense graph construction. Level
// sampling is reproducible for a fixed Seed and insertion order.
type HNSWBuildOptions struct {
	Metric         Metric
	M              int
	EFConstruction int
	Seed           uint64
}

// DefaultHNSWBuildOptions returns the pinned public construction defaults.
func DefaultHNSWBuildOptions(metric Metric) HNSWBuildOptions {
	return HNSWBuildOptions{
		Metric:         metric,
		M:              DefaultHNSWM,
		EFConstruction: DefaultHNSWEFConstruction,
	}
}

// Validate checks graph degree and construction-search invariants.
func (o HNSWBuildOptions) Validate() error {
	if !o.Metric.valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidHNSWOptions)
	}
	if o.M <= 0 || o.M > MaxHNSWM {
		return fmt.Errorf("%w: M must be in [1,%d]", ErrInvalidHNSWOptions, MaxHNSWM)
	}
	if o.EFConstruction < o.M {
		return fmt.Errorf("%w: EFConstruction must be at least M", ErrInvalidHNSWOptions)
	}
	return nil
}

// HNSWBuilder collects dense originals and constructs one deterministic graph.
type HNSWBuilder struct {
	mu        sync.Mutex
	dimension int
	options   HNSWBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

// NewHNSWBuilder constructs an empty one-shot dense HNSW builder.
func NewHNSWBuilder(dimension int, options HNSWBuildOptions) (*HNSWBuilder, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &HNSWBuilder{
		dimension: dimension,
		options:   options,
		positions: make(map[uint64]int),
	}, nil
}

// Add validates and clones one unique vector while the builder is open.
func (b *HNSWBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil HNSW builder")
	}
	if ctx == nil {
		return errors.New("core: nil HNSW add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate HNSW vector: %w", err)
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
	if len(b.vectors) > maxPlatformInt()-b.dimension {
		return ErrHNSWCapacity
	}
	b.positions[key] = len(b.keys)
	b.keys = append(b.keys, key)
	b.vectors = append(b.vectors, vector...)
	return nil
}

// Build assigns deterministic levels, inserts nodes in input order, and
// transfers builder-owned original storage to the resulting graph.
func (b *HNSWBuilder) Build(ctx context.Context) (*HNSWIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil HNSW builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil HNSW build context")
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

	index := &HNSWIndex{
		dimension:  b.dimension,
		options:    b.options,
		keys:       b.keys,
		vectors:    b.vectors,
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
			return nil, fmt.Errorf("core: construct HNSW node %d: %w", position, err)
		}
	}

	b.built = true
	b.keys = nil
	b.vectors = nil
	b.positions = nil
	return index, nil
}

// HNSWIndex stores original FP32 vectors and a bounded multi-layer proximity
// graph. Readers share one immutable generation while additions publish a
// complete copy-on-write generation.
type HNSWIndex struct {
	streamMu      sync.Mutex
	mu            sync.RWMutex
	dimension     int
	options       HNSWBuildOptions
	keys          []uint64
	vectors       []float32
	positions     map[uint64]int
	levels        []int
	neighbors     [][][]int
	entryPoint    int
	maxLevel      int
	levelRNGState uint64
}

// Dimension returns the fixed dense vector dimension.
func (i *HNSWIndex) Dimension() int {
	if i == nil {
		return 0
	}
	return i.dimension
}

// Metric returns the graph construction metric.
func (i *HNSWIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.options.Metric
}

// Len returns the number of graph nodes.
func (i *HNSWIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// BuildOptions returns the value-semantic construction settings.
func (i *HNSWIndex) BuildOptions() HNSWBuildOptions {
	if i == nil {
		return HNSWBuildOptions{}
	}
	return i.options
}

// Vector returns a cloned original vector by key.
func (i *HNSWIndex) Vector(key uint64) ([]float32, bool) {
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

// MaxLevel returns the highest occupied level, or -1 for an empty graph.
func (i *HNSWIndex) MaxLevel() int {
	if i == nil {
		return -1
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.maxLevel
}

// EntryPoint returns the current top-layer entry key.
func (i *HNSWIndex) EntryPoint() (uint64, bool) {
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
func (i *HNSWIndex) Level(key uint64) (int, bool) {
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
func (i *HNSWIndex) Neighbors(key uint64, level int) ([]uint64, error) {
	if i == nil {
		return nil, errors.New("core: nil HNSW index")
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

func (i *HNSWIndex) insertBuiltNode(ctx context.Context, position int) error {
	level := i.levels[position]
	if i.entryPoint < 0 {
		i.entryPoint = position
		i.maxLevel = level
		return nil
	}
	entry := i.entryPoint
	query := i.vectorAt(position)
	for currentLevel := i.maxLevel; currentLevel > level; currentLevel-- {
		nearest, err := i.searchHNSWLayer(ctx, query, []int{entry}, 1, currentLevel)
		if err != nil {
			return err
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	for currentLevel := min(level, i.maxLevel); currentLevel >= 0; currentLevel-- {
		candidates, err := i.searchHNSWLayer(ctx, query, []int{entry}, i.options.EFConstruction, currentLevel)
		if err != nil {
			return err
		}
		selected, err := i.selectHNSWNeighbors(ctx, position, candidates, i.maxDegree(currentLevel))
		if err != nil {
			return err
		}
		i.neighbors[position][currentLevel] = selected
		for _, neighbor := range selected {
			if err := i.addHNSWReverseEdge(ctx, neighbor, position, currentLevel); err != nil {
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

type hnswScoredNode struct {
	position int
	score    float32
}

func (i *HNSWIndex) searchHNSWLayer(ctx context.Context, query []float32, entries []int, ef, level int) ([]hnswScoredNode, error) {
	limit := min(ef, len(i.keys))
	if limit <= 0 {
		return []hnswScoredNode{}, nil
	}
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(i.options.Metric, left, right) }
	worse := func(left, right hnswScoredNode) bool { return hnswNodeBetter(i.options.Metric, right, left) }
	candidates := ailego.NewHeap(better)
	results := ailego.NewHeap(worse)
	visited := make([]bool, len(i.keys))
	for _, entry := range entries {
		if entry < 0 || entry >= len(i.keys) || i.levels[entry] < level || visited[entry] {
			continue
		}
		score, err := i.options.Metric.Compute(query, i.vectorAt(entry))
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
		if results.Len() >= limit && hasWorst && hnswNodeBetter(i.options.Metric, worst, current) {
			break
		}
		for _, neighbor := range i.neighbors[current.position][level] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := i.options.Metric.Compute(query, i.vectorAt(neighbor))
			if err != nil {
				return nil, err
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = results.Peek()
			if results.Len() < limit || !hasWorst || hnswNodeBetter(i.options.Metric, node, worst) {
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
		if hnswNodeBetter(i.options.Metric, left, right) {
			return -1
		}
		if hnswNodeBetter(i.options.Metric, right, left) {
			return 1
		}
		return 0
	})
	return result, nil
}

func (i *HNSWIndex) selectHNSWNeighbors(ctx context.Context, owner int, candidates []hnswScoredNode, limit int) ([]int, error) {
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
			between, err := i.options.Metric.Compute(i.vectorAt(candidate.position), i.vectorAt(accepted))
			if err != nil {
				return nil, err
			}
			if !i.options.Metric.Better(candidate.score, between) {
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

func (i *HNSWIndex) addHNSWReverseEdge(ctx context.Context, owner, neighbor, level int) error {
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
		score, err := i.options.Metric.Compute(i.vectorAt(owner), i.vectorAt(position))
		if err != nil {
			return err
		}
		candidates = append(candidates, hnswScoredNode{position: position, score: score})
	}
	slices.SortFunc(candidates, func(left, right hnswScoredNode) int {
		if hnswNodeBetter(i.options.Metric, left, right) {
			return -1
		}
		if hnswNodeBetter(i.options.Metric, right, left) {
			return 1
		}
		return 0
	})
	selected, err := i.selectHNSWNeighbors(ctx, owner, candidates, i.maxDegree(level))
	if err != nil {
		return err
	}
	i.neighbors[owner][level] = selected
	return nil
}

func (i *HNSWIndex) maxDegree(level int) int {
	if level == 0 {
		return i.options.M * 2
	}
	return i.options.M
}

func (i *HNSWIndex) vectorAt(position int) []float32 {
	start := position * i.dimension
	return i.vectors[start : start+i.dimension]
}

func hnswNodeBetter(metric Metric, left, right hnswScoredNode) bool {
	if left.score == right.score {
		return left.position < right.position
	}
	return metric.Better(left.score, right.score)
}

func sampleHNSWLevel(random *splitMix64, m int) int {
	// Use an open interval to avoid log(0), then cap to the baseline's fifteen
	// graph layers. M=1 uses scale two so its accepted public edge case still
	// has a finite hierarchy.
	const denominator = float64(uint64(1)<<53) + 1
	uniform := (float64(random.next()>>11) + 1) / denominator
	level := int(-math.Log(uniform) / math.Log(float64(max(2, m))))
	return min(level, MaxHNSWLevel)
}

var _ DenseProvider = (*HNSWIndex)(nil)
