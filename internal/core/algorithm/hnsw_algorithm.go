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
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/ailego/io"
	"github.com/gorse-io/xvec/internal/ailego/math"
)

const (
	DefaultHNSWM              = 50
	DefaultHNSWEFConstruction = 500
	MaxHNSWLevel              = 14
	MaxHNSWM                  = 32767
)

var (
	ErrInvalidHNSWOptions = errors.New("core: invalid HNSW build options")
	ErrInvalidHNSWWorkers = errors.New("core: HNSW workers must be positive")
	ErrInvalidHNSWLevel   = errors.New("core: invalid HNSW level")
	ErrHNSWKeyNotFound    = errors.New("core: HNSW key not found")
	ErrHNSWCapacity       = errors.New("core: HNSW index capacity exceeded")
)

// HNSWBuildOptions configures dense graph construction. Level sampling is
// reproducible for a fixed Seed; Build uses deterministic input-order insertion.
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
	if !o.Metric.Valid() {
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

// Build assigns deterministic levels, inserts nodes in input order on one
// worker, and transfers builder-owned original storage to the resulting graph.
func (b *HNSWBuilder) Build(ctx context.Context) (*HNSWIndex, error) {
	return b.build(ctx, 1)
}

// BuildWithWorkers constructs the graph with up to workers concurrent node
// insertions. A single worker is bit-for-bit deterministic; multiple workers
// preserve graph invariants but topology may vary with goroutine scheduling.
func (b *HNSWBuilder) BuildWithWorkers(ctx context.Context, workers int) (*HNSWIndex, error) {
	return b.build(ctx, workers)
}

func (b *HNSWBuilder) build(ctx context.Context, workers int) (*HNSWIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil HNSW builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil HNSW build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if workers <= 0 {
		return nil, ErrInvalidHNSWWorkers
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	distance, err := b.options.Metric.PrevalidatedDistance()
	if err != nil {
		return nil, err
	}

	index := &HNSWIndex{
		dimension:  b.dimension,
		options:    b.options,
		distance:   distance,
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
	if workers == 1 {
		for position := range index.keys {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if err := index.insertBuiltNode(ctx, position); err != nil {
				return nil, fmt.Errorf("core: construct HNSW node %d: %w", position, err)
			}
		}
	} else {
		entryPoint, maxLevel, err := buildParallelHNSW(ctx, workers, index.options, index.levels, index.neighbors,
			func(left, right int) (float32, error) {
				return index.computeDistance(index.vectorAt(left), index.vectorAt(right))
			})
		if err != nil {
			return nil, fmt.Errorf("core: construct HNSW: %w", err)
		}
		index.entryPoint = entryPoint
		index.maxLevel = maxLevel
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
	distance      mathutil.DenseDistance
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
	visited := acquireHNSWVisited(len(i.keys))
	defer releaseHNSWVisited(visited)
	level := i.levels[position]
	if i.entryPoint < 0 {
		i.entryPoint = position
		i.maxLevel = level
		return nil
	}
	entry := i.entryPoint
	query := i.vectorAt(position)
	for currentLevel := i.maxLevel; currentLevel > level; currentLevel-- {
		nearest, err := i.searchHNSWLayer(ctx, query, []int{entry}, 1, currentLevel, visited)
		if err != nil {
			return err
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	for currentLevel := min(level, i.maxLevel); currentLevel >= 0; currentLevel-- {
		candidates, err := i.searchHNSWLayer(ctx, query, []int{entry}, i.options.EFConstruction, currentLevel, visited)
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

func (i *HNSWIndex) searchHNSWLayer(ctx context.Context, query []float32, entries []int, ef, level int, visited *hnswVisited) ([]hnswScoredNode, error) {
	limit := min(ef, len(i.keys))
	if limit <= 0 {
		return []hnswScoredNode{}, nil
	}
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(i.options.Metric, left, right) }
	worse := func(left, right hnswScoredNode) bool { return hnswNodeBetter(i.options.Metric, right, left) }
	candidates := container.NewHeap(better)
	results := container.NewHeap(worse)
	visited.reset(len(i.keys))
	for _, entry := range entries {
		if entry < 0 || entry >= len(i.keys) || i.levels[entry] < level || visited.seen(entry) {
			continue
		}
		score, err := i.computeDistance(query, i.vectorAt(entry))
		if err != nil {
			return nil, err
		}
		node := hnswScoredNode{position: entry, score: score}
		visited.mark(entry)
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
			if visited.seen(neighbor) {
				continue
			}
			visited.mark(neighbor)
			score, err := i.computeDistance(query, i.vectorAt(neighbor))
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
			between, err := i.computeDistance(i.vectorAt(candidate.position), i.vectorAt(accepted))
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
		score, err := i.computeDistance(i.vectorAt(owner), i.vectorAt(position))
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

func (i *HNSWIndex) computeDistance(left, right []float32) (float32, error) {
	distance := i.distance
	if distance == nil {
		// Keep package-local literal fixtures usable while production indexes
		// always install the scorer at build or open time.
		var err error
		distance, err = i.options.Metric.PrevalidatedDistance()
		if err != nil {
			return 0, err
		}
	}
	return distance(left, right)
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

const (
	DefaultHNSWEFSearch            = 300
	MaxHNSWEFSearch                = 2048
	DefaultHNSWBruteForceThreshold = 1000
	DefaultHNSWPrefetchOffset      = 8
	MaxHNSWPrefetchLines           = 256
)

var ErrInvalidHNSWEF = errors.New("core: HNSW EF must be in [1, 2048]")

// HNSWSearchOptions combines common result controls with the level-zero
// exploration width.
type HNSWSearchOptions struct {
	SearchOptions
	EF             int
	PrefetchOffset uint32
	PrefetchLines  uint32
}

// Validate checks top-k, radius, and graph exploration invariants.
func (o HNSWSearchOptions) Validate() error {
	if err := o.SearchOptions.Validate(); err != nil {
		return err
	}
	if o.EF <= 0 || o.EF > MaxHNSWEFSearch {
		return ErrInvalidHNSWEF
	}
	return nil
}

// Search uses the pinned default EF. A zero top-k returns an empty result for
// consistency with the common DenseSearcher contract.
func (i *HNSWIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.searchHNSW(ctx, query, HNSWSearchOptions{
		SearchOptions:  SearchOptions{TopK: k},
		EF:             DefaultHNSWEFSearch,
		PrefetchOffset: DefaultHNSWPrefetchOffset,
	}, false)
}

// SearchWithOptions applies common filter and radius controls with default EF.
func (i *HNSWIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.searchHNSW(ctx, query, HNSWSearchOptions{
		SearchOptions:  options,
		EF:             DefaultHNSWEFSearch,
		PrefetchOffset: DefaultHNSWPrefetchOffset,
	}, true)
}

// SearchHNSW executes a metric-aware hierarchical graph query with explicit
// EF. EF smaller than TopK is raised to TopK so the requested result count can
// be retained.
func (i *HNSWIndex) SearchHNSW(ctx context.Context, query []float32, options HNSWSearchOptions) ([]Result, error) {
	return i.searchHNSW(ctx, query, options, true)
}

// SearchHNSWGroups performs native HNSW group traversal. It retains an
// initial groupCount*topKPerGroup candidate set and expands level zero when
// those candidates do not contain enough distinct groups.
func (i *HNSWIndex) SearchHNSWGroups(
	ctx context.Context,
	query []float32,
	options HNSWGroupSearchOptions,
) ([]GroupResult, error) {
	if i == nil {
		return nil, errors.New("core: nil HNSW index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil HNSW group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if _, err := i.options.Metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate HNSW group-by query: %w", err)
	}

	i.mu.RLock()
	defer i.mu.RUnlock()
	if len(i.keys) == 0 {
		return []GroupResult{}, nil
	}
	candidateCount, err := hnswGroupCandidateCount(options.GroupByOptions)
	if err != nil {
		return nil, err
	}
	entry := i.entryPoint
	visited := acquireHNSWVisited(len(i.keys))
	defer releaseHNSWVisited(visited)
	for level := i.maxLevel; level > 0; level-- {
		nearest, err := i.searchHNSWLayer(ctx, query, []int{entry}, 1, level, visited)
		if err != nil {
			return nil, fmt.Errorf("core: descend HNSW group-by level %d: %w", level, err)
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	searchOptions := HNSWSearchOptions{
		SearchOptions: SearchOptions{
			TopK: candidateCount, Radius: options.Radius, Filter: options.Filter,
		},
		EF: options.EF, PrefetchOffset: options.PrefetchOffset, PrefetchLines: options.PrefetchLines,
	}
	initial, err := i.searchHNSWBase(ctx, query, entry, max(options.EF, candidateCount), searchOptions, visited)
	if err != nil {
		return nil, err
	}
	if len(initial) > candidateCount {
		initial = initial[:candidateCount]
	}
	scoreAt := func(position int) (float32, error) {
		return i.computeDistance(query, i.vectorAt(position))
	}
	prefetch := func(neighbors []int) {
		prefetchDenseHNSWNeighbors(i.vectors, i.dimension, neighbors, options.PrefetchOffset, options.PrefetchLines)
	}
	return expandHNSWGroups(
		ctx, i.options.Metric, i.keys, i.neighbors, initial, options.GroupByOptions,
		scoreAt, func(score float32) float32 { return score }, groupNodeBetter(i.options.Metric, i.keys), prefetch, visited,
	)
}

func (i *HNSWIndex) searchHNSW(ctx context.Context, query []float32, options HNSWSearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil HNSW index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if ctx == nil {
		return nil, errors.New("core: nil HNSW search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if options.EF <= 0 || options.EF > MaxHNSWEFSearch {
		return nil, ErrInvalidHNSWEF
	}
	if requirePositiveTopK {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else {
		if options.TopK < 0 {
			return nil, errors.New("core: negative HNSW top-k")
		}
		if options.Radius < 0 {
			return nil, ErrInvalidRadius
		}
	}
	if _, err := i.options.Metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate HNSW query: %w", err)
	}
	if options.TopK == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}
	if len(i.keys) <= DefaultHNSWBruteForceThreshold {
		return topKPrevalidatedCandidatesWithOptions(ctx, i.options.Metric, i.computeDistance, query, options.SearchOptions, len(i.keys), func(position int) Candidate {
			return Candidate{Key: i.keys[position], Vector: i.vectorAt(position)}
		})
	}

	entry := i.entryPoint
	visited := acquireHNSWVisited(len(i.keys))
	defer releaseHNSWVisited(visited)
	for level := i.maxLevel; level > 0; level-- {
		nearest, err := i.searchHNSWLayer(ctx, query, []int{entry}, 1, level, visited)
		if err != nil {
			return nil, fmt.Errorf("core: descend HNSW level %d: %w", level, err)
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	capacity := max(options.EF, options.TopK)
	candidates, err := i.searchHNSWBase(ctx, query, entry, capacity, options, visited)
	if err != nil {
		return nil, err
	}
	if len(candidates) > options.TopK {
		candidates = candidates[:options.TopK]
	}
	results := make([]Result, len(candidates))
	for index, candidate := range candidates {
		results[index] = Result{Key: i.keys[candidate.position], Score: candidate.score}
	}
	return results, nil
}

func (i *HNSWIndex) searchHNSWBase(ctx context.Context, query []float32, entry, capacity int, options HNSWSearchOptions, visited *hnswVisited) ([]hnswScoredNode, error) {
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(i.options.Metric, left, right) }
	worse := func(left, right hnswScoredNode) bool { return i.hnswResultNodeBetter(right, left) }
	frontier := container.NewHeap(better)
	accepted := container.NewHeap(worse)
	visited.reset(len(i.keys))

	score, err := i.computeDistance(query, i.vectorAt(entry))
	if err != nil {
		return nil, fmt.Errorf("core: score HNSW entry point: %w", err)
	}
	start := hnswScoredNode{position: entry, score: score}
	visited.mark(entry)
	frontier.Push(start)
	if i.acceptHNSWResult(start, options.SearchOptions) {
		accepted.Push(start)
	}

	for frontier.Len() != 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current, _ := frontier.Pop()
		worst, hasWorst := accepted.Peek()
		if accepted.Len() >= capacity && hasWorst && i.options.Metric.Better(worst.score, current.score) {
			break
		}
		neighbors := i.neighbors[current.position][0]
		prefetchDenseHNSWNeighbors(i.vectors, i.dimension, neighbors, options.PrefetchOffset, options.PrefetchLines)
		for _, neighbor := range neighbors {
			if visited.seen(neighbor) {
				continue
			}
			visited.mark(neighbor)
			score, err := i.computeDistance(query, i.vectorAt(neighbor))
			if err != nil {
				return nil, fmt.Errorf("core: score HNSW node %d: %w", neighbor, err)
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = accepted.Peek()
			if accepted.Len() < capacity || !hasWorst || !i.options.Metric.Better(worst.score, node.score) {
				frontier.Push(node)
				if i.acceptHNSWResult(node, options.SearchOptions) {
					accepted.Push(node)
					if accepted.Len() > capacity {
						_, _ = accepted.Pop()
					}
				}
			}
		}
	}
	result := accepted.Values()
	slices.SortFunc(result, func(left, right hnswScoredNode) int {
		if i.hnswResultNodeBetter(left, right) {
			return -1
		}
		if i.hnswResultNodeBetter(right, left) {
			return 1
		}
		return 0
	})
	return result, nil
}

func (i *HNSWIndex) hnswResultNodeBetter(left, right hnswScoredNode) bool {
	if left.score == right.score {
		return i.keys[left.position] < i.keys[right.position]
	}
	return i.options.Metric.Better(left.score, right.score)
}

func (i *HNSWIndex) acceptHNSWResult(node hnswScoredNode, options SearchOptions) bool {
	key := i.keys[node.position]
	return (options.Filter == nil || options.Filter(key)) && scoreWithinRadius(i.options.Metric, node.score, options.Radius)
}

var (
	_ DenseSearcher      = (*HNSWIndex)(nil)
	_ DenseQuerySearcher = (*HNSWIndex)(nil)
)

// Add incrementally inserts one unique key and finite original vector. The
// insertion is planned on a private graph generation and becomes visible in
// one commit, so cancellation never exposes a half-linked node.
func (i *HNSWIndex) Add(ctx context.Context, key uint64, vector []float32) error {
	if i == nil {
		return errors.New("core: nil HNSW index")
	}
	if ctx == nil {
		return errors.New("core: nil HNSW add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, i.dimension); err != nil {
		return fmt.Errorf("core: validate incremental HNSW vector: %w", err)
	}

	i.streamMu.Lock()
	defer i.streamMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}

	i.mu.RLock()
	if _, exists := i.positions[key]; exists {
		i.mu.RUnlock()
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	if uint64(len(i.keys)) >= math.MaxUint32 || len(i.vectors) > maxPlatformInt()-i.dimension {
		i.mu.RUnlock()
		return ErrHNSWCapacity
	}

	working, err := cloneHNSWIndex(ctx, i)
	i.mu.RUnlock()
	if err != nil {
		return err
	}
	random := splitMix64{state: working.levelRNGState}
	level := sampleHNSWLevel(&random, working.options.M)
	position := len(working.keys)
	working.keys = append(working.keys, key)
	working.vectors = append(working.vectors, vector...)
	working.positions[key] = position
	working.levels = append(working.levels, level)
	working.neighbors = append(working.neighbors, make([][]int, level+1))
	working.levelRNGState = random.state
	if err := working.insertBuiltNode(ctx, position); err != nil {
		return fmt.Errorf("core: insert incremental HNSW node %d: %w", position, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	i.mu.Lock()
	if err := ctx.Err(); err != nil {
		i.mu.Unlock()
		return err
	}
	i.keys = working.keys
	i.vectors = working.vectors
	i.positions = working.positions
	i.levels = working.levels
	i.neighbors = working.neighbors
	i.entryPoint = working.entryPoint
	i.maxLevel = working.maxLevel
	i.levelRNGState = working.levelRNGState
	i.mu.Unlock()
	return nil
}

// cloneHNSWIndex copies one complete generation. Callers that clone a live
// streamable index must hold its read or write lock for the duration.
func cloneHNSWIndex(ctx context.Context, source *HNSWIndex) (*HNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW clone context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidHNSWFile)
	}
	clone := &HNSWIndex{
		dimension:     source.dimension,
		options:       source.options,
		distance:      source.distance,
		keys:          slices.Clone(source.keys),
		vectors:       slices.Clone(source.vectors),
		positions:     make(map[uint64]int, len(source.positions)),
		levels:        slices.Clone(source.levels),
		neighbors:     make([][][]int, len(source.neighbors)),
		entryPoint:    source.entryPoint,
		maxLevel:      source.maxLevel,
		levelRNGState: source.levelRNGState,
	}
	for key, position := range source.positions {
		clone.positions[key] = position
	}
	for position, levels := range source.neighbors {
		if position&127 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		clone.neighbors[position] = make([][]int, len(levels))
		for level, neighbors := range levels {
			clone.neighbors[position][level] = slices.Clone(neighbors)
		}
	}
	return clone, nil
}

var (
	_ DenseStreamer = (*HNSWIndex)(nil)
	_ DenseIndex    = (*HNSWIndex)(nil)
)

const (
	hnswFileVersion = 1
	hnswHeaderSize  = 112
	hnswReadChunk   = 1 << 20

	hnswRecordFixedBytes = 12 // key uint64 plus maximum level uint32
	hnswLevelFixedBytes  = 4  // neighbor count uint32
)

var (
	hnswFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'H', 'N', 'S', 'W'}

	// ErrInvalidHNSWFile reports a structurally or semantically invalid native
	// Go HNSW artifact.
	ErrInvalidHNSWFile = errors.New("core: invalid HNSW file")
	// ErrHNSWChecksumMismatch distinguishes detected bit flips from other
	// format violations.
	ErrHNSWChecksumMismatch = errors.New("core: HNSW checksum mismatch")
	// ErrUnsupportedHNSWVersion reports a native Go HNSW artifact whose format
	// version is not supported by this library.
	ErrUnsupportedHNSWVersion = errors.New("core: unsupported HNSW file version")
)

// Save durably publishes the immutable graph as one checksummed native Go
// HNSW file. Replacing an existing file is atomic to concurrent openers.
func (i *HNSWIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil HNSW save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidHNSWFile)
	}
	if i == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidHNSWFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateHNSWIndex(ctx, i); err != nil {
		return err
	}
	payloadSize, err := checkedHNSWPayloadSize(i)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFileAtomicFunc(ctx, path, 0o600, func(file *os.File) error {
		return writeHNSWIndex(ctx, file, i, payloadSize)
	}); err != nil {
		return fmt.Errorf("core: save HNSW file: %w", err)
	}
	return nil
}

func writeHNSWIndex(ctx context.Context, file *os.File, index *HNSWIndex, payloadSize int) error {
	header := makeHNSWHeader(index, payloadSize, 0)
	if _, err := file.Write(header); err != nil {
		return err
	}
	writer := bufio.NewWriterSize(file, hnswReadChunk)
	node := make([]byte, 0, hnswRecordFixedBytes+index.dimension*4+(MaxHNSWLevel+1)*(hnswLevelFixedBytes+index.options.M*8))
	var payloadCRC uint32
	written := 0
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		node = node[:0]
		node = binary.LittleEndian.AppendUint64(node, key)
		node = binary.LittleEndian.AppendUint32(node, uint32(index.levels[position]))
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			node = binary.LittleEndian.AppendUint32(node, math.Float32bits(value))
		}
		for _, neighbors := range index.neighbors[position] {
			node = binary.LittleEndian.AppendUint32(node, uint32(len(neighbors)))
			for _, neighbor := range neighbors {
				node = binary.LittleEndian.AppendUint32(node, uint32(neighbor))
			}
		}
		payloadCRC = hashutil.UpdateCRC32C(payloadCRC, node)
		count, err := writer.Write(node)
		if err != nil {
			return err
		}
		if count != len(node) {
			return io.ErrShortWrite
		}
		written += count
	}
	if written != payloadSize {
		return fmt.Errorf("%w: internal payload length", ErrInvalidHNSWFile)
	}
	if err := writer.Flush(); err != nil {
		return err
	}
	header = makeHNSWHeader(index, payloadSize, payloadCRC)
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if _, err := file.Write(header); err != nil {
		return err
	}
	return nil
}

func makeHNSWHeader(index *HNSWIndex, payloadSize int, payloadCRC uint32) []byte {
	header := make([]byte, hnswHeaderSize)
	copy(header[:8], hnswFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], hnswFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], hnswHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(hnswHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(index.keys)))
	binary.LittleEndian.PutUint32(header[40:44], uint32(index.dimension))
	binary.LittleEndian.PutUint32(header[44:48], uint32(index.options.M))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.options.EFConstruction))
	header[52] = byte(index.options.Metric)
	entryPoint := uint64(math.MaxUint64)
	if index.entryPoint >= 0 {
		entryPoint = uint64(index.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[56:64], entryPoint)
	binary.LittleEndian.PutUint32(header[64:68], uint32(int32(index.maxLevel)))
	binary.LittleEndian.PutUint64(header[68:76], index.options.Seed)
	binary.LittleEndian.PutUint64(header[76:84], index.levelRNGState)
	binary.LittleEndian.PutUint32(header[84:88], payloadCRC)
	binary.LittleEndian.PutUint32(header[108:112], hashutil.CRC32C(header[:108]))
	return header
}

func (i *HNSWIndex) persistenceSnapshot(ctx context.Context) (*HNSWIndex, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidHNSWFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return cloneHNSWIndex(ctx, i)
}

// OpenHNSWIndex reads and fully verifies a native Go HNSW artifact. The
// returned graph owns all decoded memory and does not retain the source file.
func OpenHNSWIndex(ctx context.Context, path string) (*HNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidHNSWFile)
	}
	encoded, err := readHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read HNSW file: %w", err)
	}
	index, err := decodeHNSWIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open HNSW file: %w", err)
	}
	return index, nil
}

func encodeHNSWIndex(ctx context.Context, index *HNSWIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if index == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidHNSWFile)
	}
	if err := validateHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	payloadSize, err := checkedHNSWPayloadSize(index)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint64(payload, key)
		payload = binary.LittleEndian.AppendUint32(payload, uint32(index.levels[position]))
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
		for _, neighbors := range index.neighbors[position] {
			payload = binary.LittleEndian.AppendUint32(payload, uint32(len(neighbors)))
			for _, neighbor := range neighbors {
				payload = binary.LittleEndian.AppendUint32(payload, uint32(neighbor))
			}
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidHNSWFile)
	}

	header := makeHNSWHeader(index, payloadSize, hashutil.CRC32C(payload))
	return append(header, payload...), nil
}

func decodeHNSWIndex(ctx context.Context, encoded []byte) (*HNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < hnswHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidHNSWFile)
	}
	header := encoded[:hnswHeaderSize]
	if !bytes.Equal(header[:8], hnswFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidHNSWFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != hnswFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedHNSWVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != hnswHeaderSize {
		return nil, fmt.Errorf("%w: bad header size", ErrInvalidHNSWFile)
	}
	if binary.LittleEndian.Uint32(header[12:16]) != 0 ||
		!hnswAllZero(header[53:56]) ||
		!hnswAllZero(header[88:108]) {
		return nil, fmt.Errorf("%w: nonzero reserved field", ErrInvalidHNSWFile)
	}
	if got, want := hashutil.CRC32C(header[:108]), binary.LittleEndian.Uint32(header[108:112]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrHNSWChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-hnswHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidHNSWFile)
	}
	payload := encoded[hnswHeaderSize:]
	if got, want := hashutil.CRC32C(payload), binary.LittleEndian.Uint32(header[84:88]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrHNSWChecksumMismatch, got, want)
	}

	count64 := binary.LittleEndian.Uint64(header[32:40])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[40:44]))
	if count64 > uint64(math.MaxUint32) || count64 > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: node count exceeds format capacity", ErrInvalidHNSWFile)
	}
	if dimension64 == 0 || dimension64 > MaxRotationDimension {
		return nil, fmt.Errorf("%w: invalid dimension %d", ErrInvalidHNSWFile, dimension64)
	}
	count, dimension := int(count64), int(dimension64)
	minimumSize, err := checkedHNSWMinimumPayloadSize(dimension, count)
	if err != nil || minimumSize > len(payload) {
		return nil, fmt.Errorf("%w: invalid payload length", ErrInvalidHNSWFile)
	}
	options, err := decodeHNSWOptions(header)
	if err != nil {
		return nil, err
	}
	entry64 := binary.LittleEndian.Uint64(header[56:64])
	maxLevel := int(int32(binary.LittleEndian.Uint32(header[64:68])))
	if maxLevel < -1 || maxLevel > MaxHNSWLevel {
		return nil, fmt.Errorf("%w: invalid maximum level", ErrInvalidHNSWFile)
	}
	entryPoint := -1
	if entry64 != math.MaxUint64 {
		if entry64 >= count64 {
			return nil, fmt.Errorf("%w: entry point out of range", ErrInvalidHNSWFile)
		}
		entryPoint = int(entry64)
	}
	if (count == 0 && (entryPoint != -1 || maxLevel != -1)) ||
		(count != 0 && (entryPoint < 0 || maxLevel < 0)) {
		return nil, fmt.Errorf("%w: inconsistent entry point", ErrInvalidHNSWFile)
	}
	if count > maxPlatformInt()/dimension {
		return nil, fmt.Errorf("%w: vector storage exceeds platform capacity", ErrInvalidHNSWFile)
	}
	distance, err := options.Metric.PrevalidatedDistance()
	if err != nil {
		return nil, fmt.Errorf("%w: invalid metric", ErrInvalidHNSWFile)
	}

	index := &HNSWIndex{
		dimension:     dimension,
		options:       options,
		distance:      distance,
		keys:          make([]uint64, count),
		vectors:       make([]float32, count*dimension),
		positions:     make(map[uint64]int, count),
		levels:        make([]int, count),
		neighbors:     make([][][]int, count),
		entryPoint:    entryPoint,
		maxLevel:      maxLevel,
		levelRNGState: binary.LittleEndian.Uint64(header[76:84]),
	}
	offset := 0
	for position := 0; position < count; position++ {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		vectorBytes := dimension * 4
		if !hnswPayloadAvailable(payload, offset, hnswRecordFixedBytes+vectorBytes+hnswLevelFixedBytes) {
			return nil, fmt.Errorf("%w: truncated node %d", ErrInvalidHNSWFile, position)
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, duplicate := index.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key %d", ErrInvalidHNSWFile, key)
		}
		index.keys[position] = key
		index.positions[key] = position
		level := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if level > MaxHNSWLevel {
			return nil, fmt.Errorf("%w: invalid node level %d", ErrInvalidHNSWFile, level)
		}
		index.levels[position] = int(level)
		index.neighbors[position] = make([][]int, int(level)+1)
		start := position * dimension
		for component := 0; component < dimension; component++ {
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: non-finite vector", ErrInvalidHNSWFile)
			}
			index.vectors[start+component] = value
		}
		for currentLevel := 0; currentLevel <= int(level); currentLevel++ {
			if !hnswPayloadAvailable(payload, offset, hnswLevelFixedBytes) {
				return nil, fmt.Errorf("%w: truncated neighbors", ErrInvalidHNSWFile)
			}
			degree64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			degreeLimit := options.M
			if currentLevel == 0 {
				degreeLimit *= 2
			}
			if degree64 > uint64(degreeLimit) || degree64 > count64 {
				return nil, fmt.Errorf("%w: node %d level %d degree out of range", ErrInvalidHNSWFile, position, currentLevel)
			}
			degree := int(degree64)
			if !hnswPayloadAvailable(payload, offset, degree*4) {
				return nil, fmt.Errorf("%w: truncated neighbor positions", ErrInvalidHNSWFile)
			}
			var neighbors []int
			if degree != 0 {
				neighbors = make([]int, degree)
			}
			seen := make(map[int]struct{}, degree)
			for neighborIndex := range neighbors {
				neighbor64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
				offset += 4
				if neighbor64 >= count64 || neighbor64 == uint64(position) {
					return nil, fmt.Errorf("%w: invalid neighbor reference", ErrInvalidHNSWFile)
				}
				neighbor := int(neighbor64)
				if _, duplicate := seen[neighbor]; duplicate {
					return nil, fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidHNSWFile)
				}
				seen[neighbor] = struct{}{}
				neighbors[neighborIndex] = neighbor
			}
			index.neighbors[position][currentLevel] = neighbors
		}
	}
	if offset != len(payload) {
		return nil, fmt.Errorf("%w: trailing payload data", ErrInvalidHNSWFile)
	}
	if err := validateHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	return index, nil
}

func decodeHNSWOptions(header []byte) (HNSWBuildOptions, error) {
	m64 := uint64(binary.LittleEndian.Uint32(header[44:48]))
	ef64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	if m64 > uint64(maxPlatformInt()) || ef64 > uint64(maxPlatformInt()) {
		return HNSWBuildOptions{}, fmt.Errorf("%w: options exceed platform capacity", ErrInvalidHNSWFile)
	}
	options := HNSWBuildOptions{
		Metric:         Metric(header[52]),
		M:              int(m64),
		EFConstruction: int(ef64),
		Seed:           binary.LittleEndian.Uint64(header[68:76]),
	}
	if err := options.Validate(); err != nil {
		return HNSWBuildOptions{}, fmt.Errorf("%w: %v", ErrInvalidHNSWFile, err)
	}
	return options, nil
}

func validateHNSWIndex(ctx context.Context, index *HNSWIndex) error {
	if ctx == nil {
		return errors.New("core: nil HNSW validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidHNSWFile)
	}
	if index.dimension <= 0 || index.dimension > MaxRotationDimension {
		return fmt.Errorf("%w: invalid dimension", ErrInvalidHNSWFile)
	}
	if err := index.options.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidHNSWFile, err)
	}
	if index.options.M > math.MaxUint32 || index.options.EFConstruction > math.MaxUint32 {
		return fmt.Errorf("%w: options exceed format capacity", ErrInvalidHNSWFile)
	}
	count := len(index.keys)
	if uint64(count) > math.MaxUint32 || count > maxPlatformInt()/index.dimension ||
		len(index.vectors) != count*index.dimension || len(index.positions) != count ||
		len(index.levels) != count || len(index.neighbors) != count {
		return fmt.Errorf("%w: inconsistent graph storage", ErrInvalidHNSWFile)
	}
	if count == 0 {
		if index.entryPoint != -1 || index.maxLevel != -1 {
			return fmt.Errorf("%w: inconsistent empty graph", ErrInvalidHNSWFile)
		}
		return nil
	}
	if index.entryPoint < 0 || index.entryPoint >= count || index.maxLevel < 0 || index.maxLevel > MaxHNSWLevel {
		return fmt.Errorf("%w: invalid graph entry point", ErrInvalidHNSWFile)
	}
	derivedMaxLevel := -1
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if mapped, found := index.positions[key]; !found || mapped != position {
			return fmt.Errorf("%w: inconsistent key map", ErrInvalidHNSWFile)
		}
		level := index.levels[position]
		if level < 0 || level > MaxHNSWLevel || len(index.neighbors[position]) != level+1 {
			return fmt.Errorf("%w: invalid node level storage", ErrInvalidHNSWFile)
		}
		derivedMaxLevel = max(derivedMaxLevel, level)
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			if !finiteFloat32(value) {
				return fmt.Errorf("%w: non-finite vector", ErrInvalidHNSWFile)
			}
		}
		for currentLevel, neighbors := range index.neighbors[position] {
			degreeLimit := index.options.M
			if currentLevel == 0 {
				degreeLimit *= 2
			}
			if len(neighbors) > degreeLimit {
				return fmt.Errorf("%w: node degree exceeds limit", ErrInvalidHNSWFile)
			}
			seen := make(map[int]struct{}, len(neighbors))
			for _, neighbor := range neighbors {
				if neighbor < 0 || neighbor >= count || neighbor == position || index.levels[neighbor] < currentLevel {
					return fmt.Errorf("%w: invalid neighbor reference", ErrInvalidHNSWFile)
				}
				if _, duplicate := seen[neighbor]; duplicate {
					return fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidHNSWFile)
				}
				seen[neighbor] = struct{}{}
			}
		}
	}
	if derivedMaxLevel != index.maxLevel || index.levels[index.entryPoint] != index.maxLevel {
		return fmt.Errorf("%w: inconsistent maximum level", ErrInvalidHNSWFile)
	}
	return nil
}

func checkedHNSWPayloadSize(index *HNSWIndex) (int, error) {
	minimum, err := checkedHNSWMinimumPayloadSize(index.dimension, len(index.keys))
	if err != nil {
		return 0, err
	}
	total := uint64(minimum)
	for position, level := range index.levels {
		// The minimum already includes one level-count field for every node.
		total += uint64(level) * hnswLevelFixedBytes
		for _, neighbors := range index.neighbors[position] {
			total += uint64(len(neighbors)) * 4
			if total > uint64(maxPlatformInt()-hnswHeaderSize) {
				return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidHNSWFile)
			}
		}
	}
	return int(total), nil
}

func checkedHNSWMinimumPayloadSize(dimension, count int) (int, error) {
	if dimension <= 0 || count < 0 {
		return 0, fmt.Errorf("%w: invalid size", ErrInvalidHNSWFile)
	}
	recordBytes := uint64(hnswRecordFixedBytes+hnswLevelFixedBytes) + uint64(dimension)*4
	total := uint64(count) * recordBytes
	if count != 0 && total/recordBytes != uint64(count) {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidHNSWFile)
	}
	if total > uint64(maxPlatformInt()-hnswHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidHNSWFile)
	}
	return int(total), nil
}

func hnswPayloadAvailable(payload []byte, offset, size int) bool {
	return offset >= 0 && size >= 0 && offset <= len(payload) && size <= len(payload)-offset
}

func readHNSWFile(ctx context.Context, path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < 0 || uint64(info.Size()) > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: file exceeds platform capacity", ErrInvalidHNSWFile)
	}
	encoded := make([]byte, int(info.Size()))
	for offset := 0; offset < len(encoded); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+hnswReadChunk, len(encoded))
		if _, err := io.ReadFull(file, encoded[offset:end]); err != nil {
			return nil, err
		}
		offset = end
	}
	return encoded, nil
}

func hnswAllZero(data []byte) bool {
	for _, value := range data {
		if value != 0 {
			return false
		}
	}
	return true
}
