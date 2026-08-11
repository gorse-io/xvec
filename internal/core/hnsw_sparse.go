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
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
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

// Build assigns deterministic levels, inserts nodes in input order on one
// worker, and transfers the builder-owned CSR vectors to the resulting graph.
func (b *SparseHNSWBuilder) Build(ctx context.Context) (*SparseHNSWIndex, error) {
	return b.build(ctx, 1)
}

// BuildWithWorkers constructs the graph with up to workers concurrent node
// insertions. A single worker is bit-for-bit deterministic; multiple workers
// preserve graph invariants but topology may vary with goroutine scheduling.
func (b *SparseHNSWBuilder) BuildWithWorkers(ctx context.Context, workers int) (*SparseHNSWIndex, error) {
	return b.build(ctx, workers)
}

func (b *SparseHNSWBuilder) build(ctx context.Context, workers int) (*SparseHNSWIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil sparse HNSW builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW build context")
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
	if workers == 1 {
		for position := range index.keys {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if err := index.insertBuiltNode(ctx, position); err != nil {
				return nil, fmt.Errorf("core: construct sparse HNSW node %d: %w", position, err)
			}
		}
	} else {
		entryPoint, maxLevel, err := buildParallelHNSW(ctx, workers, index.options, index.levels, index.neighbors,
			func(left, right int) (float32, error) {
				return sparseHNSWScore(index.sparseVectorAt(left), index.sparseVectorAt(right))
			})
		if err != nil {
			return nil, fmt.Errorf("core: construct sparse HNSW: %w", err)
		}
		index.entryPoint = entryPoint
		index.maxLevel = maxLevel
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

// SearchSparse uses the pinned default EF. A zero top-k returns an empty
// result for consistency with the common SparseSearcher contract.
func (i *SparseHNSWIndex) SearchSparse(ctx context.Context, query SparseVector, k int) ([]Result, error) {
	return i.searchSparseHNSW(ctx, query, HNSWSearchOptions{
		SearchOptions:  SearchOptions{TopK: k},
		EF:             DefaultHNSWEFSearch,
		PrefetchOffset: DefaultHNSWPrefetchOffset,
	}, false)
}

// SearchSparseWithOptions applies common filter and radius controls with the
// pinned default EF.
func (i *SparseHNSWIndex) SearchSparseWithOptions(ctx context.Context, query SparseVector, options SearchOptions) ([]Result, error) {
	return i.searchSparseHNSW(ctx, query, HNSWSearchOptions{
		SearchOptions:  options,
		EF:             DefaultHNSWEFSearch,
		PrefetchOffset: DefaultHNSWPrefetchOffset,
	}, true)
}

// SearchSparseHNSW executes an inner-product hierarchical graph query with an
// explicit EF. EF smaller than TopK is raised to TopK for candidate retention.
func (i *SparseHNSWIndex) SearchSparseHNSW(ctx context.Context, query SparseVector, options HNSWSearchOptions) ([]Result, error) {
	return i.searchSparseHNSW(ctx, query, options, true)
}

// SearchSparseHNSWGroups performs native sparse HNSW group traversal and
// expands level zero when the initial candidates do not cover enough groups.
func (i *SparseHNSWIndex) SearchSparseHNSWGroups(
	ctx context.Context,
	query SparseVector,
	options HNSWGroupSearchOptions,
) ([]GroupResult, error) {
	if i == nil {
		return nil, errors.New("core: nil sparse HNSW index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if _, err := sparseHNSWScore(query, SparseVector{}); err != nil {
		return nil, fmt.Errorf("core: validate sparse HNSW group-by query: %w", err)
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
	for level := i.maxLevel; level > 0; level-- {
		nearest, err := i.searchLayer(ctx, query, []int{entry}, 1, level)
		if err != nil {
			return nil, fmt.Errorf("core: descend sparse HNSW group-by level %d: %w", level, err)
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
	initial, err := i.searchBase(ctx, query, entry, max(options.EF, candidateCount), searchOptions)
	if err != nil {
		return nil, err
	}
	if len(initial) > candidateCount {
		initial = initial[:candidateCount]
	}
	scoreAt := func(position int) (float32, error) {
		return sparseHNSWScore(query, i.sparseVectorAt(position))
	}
	prefetch := func(neighbors []int) {
		prefetchSparseHNSWNeighbors(i.offsets, i.indices, i.values, neighbors, options.PrefetchOffset, options.PrefetchLines)
	}
	return expandHNSWGroups(
		ctx, MetricIP, i.keys, i.neighbors, initial, options.GroupByOptions,
		scoreAt, func(score float32) float32 { return score }, groupNodeBetter(MetricIP, i.keys), prefetch,
	)
}

func (i *SparseHNSWIndex) searchSparseHNSW(ctx context.Context, query SparseVector, options HNSWSearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil sparse HNSW index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
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
			return nil, errors.New("core: negative sparse HNSW top-k")
		}
		if options.Radius < 0 {
			return nil, ErrInvalidRadius
		}
	}
	if _, err := sparseHNSWScore(query, SparseVector{}); err != nil {
		return nil, fmt.Errorf("core: validate sparse HNSW query: %w", err)
	}
	if options.TopK == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}
	if len(i.keys) <= DefaultHNSWBruteForceThreshold {
		return topKSparseCandidatesWithOptions(ctx, query, options.SearchOptions, i.keys, i.offsets, i.indices, i.values, requirePositiveTopK)
	}

	entry := i.entryPoint
	for level := i.maxLevel; level > 0; level-- {
		nearest, err := i.searchLayer(ctx, query, []int{entry}, 1, level)
		if err != nil {
			return nil, fmt.Errorf("core: descend sparse HNSW level %d: %w", level, err)
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	capacity := max(options.EF, options.TopK)
	candidates, err := i.searchBase(ctx, query, entry, capacity, options)
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

func (i *SparseHNSWIndex) searchBase(ctx context.Context, query SparseVector, entry, capacity int, options HNSWSearchOptions) ([]hnswScoredNode, error) {
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(MetricIP, left, right) }
	worse := func(left, right hnswScoredNode) bool { return i.resultNodeBetter(right, left) }
	frontier := ailego.NewHeap(better)
	accepted := ailego.NewHeap(worse)
	visited := make([]bool, len(i.keys))

	score, err := sparseHNSWScore(query, i.sparseVectorAt(entry))
	if err != nil {
		return nil, fmt.Errorf("core: score sparse HNSW entry point: %w", err)
	}
	start := hnswScoredNode{position: entry, score: score}
	visited[entry] = true
	frontier.Push(start)
	if i.acceptResult(start, options.SearchOptions) {
		accepted.Push(start)
	}

	for frontier.Len() != 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current, _ := frontier.Pop()
		worst, hasWorst := accepted.Peek()
		if accepted.Len() >= capacity && hasWorst && worst.score > current.score {
			break
		}
		neighbors := i.neighbors[current.position][0]
		prefetchSparseHNSWNeighbors(i.offsets, i.indices, i.values, neighbors, options.PrefetchOffset, options.PrefetchLines)
		for _, neighbor := range neighbors {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := sparseHNSWScore(query, i.sparseVectorAt(neighbor))
			if err != nil {
				return nil, fmt.Errorf("core: score sparse HNSW node %d: %w", neighbor, err)
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = accepted.Peek()
			if accepted.Len() < capacity || !hasWorst || node.score >= worst.score {
				frontier.Push(node)
				if i.acceptResult(node, options.SearchOptions) {
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
		if i.resultNodeBetter(left, right) {
			return -1
		}
		if i.resultNodeBetter(right, left) {
			return 1
		}
		return 0
	})
	return result, nil
}

func (i *SparseHNSWIndex) resultNodeBetter(left, right hnswScoredNode) bool {
	if left.score == right.score {
		return i.keys[left.position] < i.keys[right.position]
	}
	return left.score > right.score
}

func (i *SparseHNSWIndex) acceptResult(node hnswScoredNode, options SearchOptions) bool {
	key := i.keys[node.position]
	return (options.Filter == nil || options.Filter(key)) && scoreWithinRadius(MetricIP, node.score, options.Radius)
}

func topKSparseCandidatesWithOptions(
	ctx context.Context,
	query SparseVector,
	options SearchOptions,
	keys []uint64,
	offsets []int,
	indices []uint32,
	values []float32,
	requirePositiveTopK bool,
) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse top-k context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if requirePositiveTopK {
		if err := options.Validate(); err != nil {
			return nil, err
		}
	} else if options.TopK < 0 {
		return nil, errors.New("core: negative sparse top-k")
	}
	if _, err := sparseHNSWScore(query, SparseVector{}); err != nil {
		return nil, fmt.Errorf("core: validate sparse query: %w", err)
	}
	if options.TopK == 0 || len(keys) == 0 {
		return []Result{}, nil
	}
	k := min(options.TopK, len(keys))
	worstFirst := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return left.Score < right.Score
	}
	heap := ailego.NewHeap(worstFirst)
	for position, key := range keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		start, end := offsets[position], offsets[position+1]
		score, err := ailego.SparseInnerProduct(query.Indices, query.Values, indices[start:end], values[start:end])
		if err != nil {
			return nil, fmt.Errorf("core: score sparse candidate %d (key %d): %w", position, key, err)
		}
		if !scoreWithinRadius(MetricIP, score, options.Radius) {
			continue
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

var (
	_ SparseSearcher      = (*SparseHNSWIndex)(nil)
	_ SparseQuerySearcher = (*SparseHNSWIndex)(nil)
)

// AddSparse incrementally inserts one unique key and canonical sparse vector.
// The insertion is planned on a private graph generation and becomes visible
// in one commit, so cancellation never exposes partial CSR or topology state.
func (i *SparseHNSWIndex) AddSparse(ctx context.Context, key uint64, vector SparseVector) error {
	if i == nil {
		return errors.New("core: nil sparse HNSW index")
	}
	if ctx == nil {
		return errors.New("core: nil sparse HNSW add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := ailego.SparseInnerProduct(vector.Indices, vector.Values, nil, nil); err != nil {
		return fmt.Errorf("core: validate incremental sparse HNSW vector: %w", err)
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
	if len(i.keys) == maxPlatformInt() || uint64(len(i.keys)) >= math.MaxUint32 ||
		len(vector.Indices) > maxPlatformInt()-len(i.indices) {
		i.mu.RUnlock()
		return ErrSparseHNSWCapacity
	}
	working, err := cloneSparseHNSWIndex(ctx, i)
	i.mu.RUnlock()
	if err != nil {
		return err
	}

	random := splitMix64{state: working.levelRNGState}
	level := sampleHNSWLevel(&random, working.options.M)
	position := len(working.keys)
	working.keys = append(working.keys, key)
	working.indices = append(working.indices, vector.Indices...)
	working.values = append(working.values, vector.Values...)
	working.offsets = append(working.offsets, len(working.indices))
	working.positions[key] = position
	working.levels = append(working.levels, level)
	working.neighbors = append(working.neighbors, make([][]int, level+1))
	working.levelRNGState = random.state
	if err := working.insertBuiltNode(ctx, position); err != nil {
		return fmt.Errorf("core: insert incremental sparse HNSW node %d: %w", position, err)
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
	i.offsets = working.offsets
	i.indices = working.indices
	i.values = working.values
	i.positions = working.positions
	i.levels = working.levels
	i.neighbors = working.neighbors
	i.entryPoint = working.entryPoint
	i.maxLevel = working.maxLevel
	i.levelRNGState = working.levelRNGState
	i.mu.Unlock()
	return nil
}

// cloneSparseHNSWIndex copies one complete generation. Callers that clone a
// live streamable index must hold its read or write lock for the duration.
func cloneSparseHNSWIndex(ctx context.Context, source *SparseHNSWIndex) (*SparseHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW clone context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidSparseHNSWFile)
	}
	clone := &SparseHNSWIndex{
		options:       source.options,
		keys:          slices.Clone(source.keys),
		offsets:       slices.Clone(source.offsets),
		indices:       slices.Clone(source.indices),
		values:        slices.Clone(source.values),
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
	_ SparseStreamer = (*SparseHNSWIndex)(nil)
	_ SparseIndex    = (*SparseHNSWIndex)(nil)
)

const (
	sparseHNSWFileVersion = 1
	sparseHNSWHeaderSize  = 112
	sparseHNSWReadChunk   = 1 << 20

	sparseHNSWRecordFixedBytes = 16 // key, maximum level, and nonzero count
	sparseHNSWLevelFixedBytes  = 4  // neighbor count
	sparseHNSWElementBytes     = 8  // coordinate and FP32 value
)

var (
	sparseHNSWFileMagic = [8]byte{'Z', 'V', 'S', 'P', 'H', 'N', 'S', 'W'}

	// ErrInvalidSparseHNSWFile reports a structurally or semantically invalid
	// native Go sparse HNSW artifact.
	ErrInvalidSparseHNSWFile = errors.New("core: invalid sparse HNSW file")
	// ErrSparseHNSWChecksumMismatch distinguishes bit flips from other format
	// violations.
	ErrSparseHNSWChecksumMismatch = errors.New("core: sparse HNSW checksum mismatch")
	// ErrUnsupportedSparseHNSWVersion reports an unsupported native Go sparse
	// HNSW format version.
	ErrUnsupportedSparseHNSWVersion = errors.New("core: unsupported sparse HNSW file version")
)

// Save durably publishes one complete graph snapshot as a checksummed native
// Go sparse HNSW file.
func (i *SparseHNSWIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil sparse HNSW save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidSparseHNSWFile)
	}
	snapshot, err := i.persistenceSnapshot(ctx)
	if err != nil {
		return err
	}
	encoded, err := encodeSparseHNSWIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save sparse HNSW file: %w", err)
	}
	return nil
}

func (i *SparseHNSWIndex) persistenceSnapshot(ctx context.Context) (*SparseHNSWIndex, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidSparseHNSWFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return cloneSparseHNSWIndex(ctx, i)
}

// OpenSparseHNSWIndex reads and fully verifies a native Go sparse HNSW
// artifact. The returned graph owns its decoded memory.
func OpenSparseHNSWIndex(ctx context.Context, path string) (*SparseHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidSparseHNSWFile)
	}
	encoded, err := readSparseHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read sparse HNSW file: %w", err)
	}
	index, err := decodeSparseHNSWIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open sparse HNSW file: %w", err)
	}
	return index, nil
}

func encodeSparseHNSWIndex(ctx context.Context, index *SparseHNSWIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateSparseHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	payloadSize, err := checkedSparseHNSWPayloadSize(index)
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
		start, end := index.offsets[position], index.offsets[position+1]
		payload = binary.LittleEndian.AppendUint32(payload, uint32(end-start))
		for element := start; element < end; element++ {
			payload = binary.LittleEndian.AppendUint32(payload, index.indices[element])
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(index.values[element]))
		}
		for _, neighbors := range index.neighbors[position] {
			payload = binary.LittleEndian.AppendUint32(payload, uint32(len(neighbors)))
			for _, neighbor := range neighbors {
				payload = binary.LittleEndian.AppendUint32(payload, uint32(neighbor))
			}
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidSparseHNSWFile)
	}

	header := make([]byte, sparseHNSWHeaderSize)
	copy(header[:8], sparseHNSWFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], sparseHNSWFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], sparseHNSWHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(sparseHNSWHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(index.keys)))
	binary.LittleEndian.PutUint64(header[40:48], uint64(len(index.indices)))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.options.M))
	binary.LittleEndian.PutUint32(header[52:56], uint32(index.options.EFConstruction))
	header[56] = byte(index.options.Metric)
	entryPoint := uint64(math.MaxUint64)
	if index.entryPoint >= 0 {
		entryPoint = uint64(index.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[60:68], entryPoint)
	binary.LittleEndian.PutUint32(header[68:72], uint32(int32(index.maxLevel)))
	binary.LittleEndian.PutUint64(header[72:80], index.options.Seed)
	binary.LittleEndian.PutUint64(header[80:88], index.levelRNGState)
	binary.LittleEndian.PutUint32(header[88:92], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[108:112], ailego.CRC32C(header[:108]))
	return append(header, payload...), nil
}

func decodeSparseHNSWIndex(ctx context.Context, encoded []byte) (*SparseHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse HNSW decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < sparseHNSWHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidSparseHNSWFile)
	}
	header := encoded[:sparseHNSWHeaderSize]
	if !bytes.Equal(header[:8], sparseHNSWFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidSparseHNSWFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != sparseHNSWFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedSparseHNSWVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != sparseHNSWHeaderSize {
		return nil, fmt.Errorf("%w: bad header size", ErrInvalidSparseHNSWFile)
	}
	if binary.LittleEndian.Uint32(header[12:16]) != 0 || !hnswAllZero(header[57:60]) || !hnswAllZero(header[92:108]) {
		return nil, fmt.Errorf("%w: nonzero reserved field", ErrInvalidSparseHNSWFile)
	}
	if got, want := ailego.CRC32C(header[:108]), binary.LittleEndian.Uint32(header[108:112]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrSparseHNSWChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-sparseHNSWHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidSparseHNSWFile)
	}
	payload := encoded[sparseHNSWHeaderSize:]
	if got, want := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[88:92]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrSparseHNSWChecksumMismatch, got, want)
	}

	count64 := binary.LittleEndian.Uint64(header[32:40])
	elements64 := binary.LittleEndian.Uint64(header[40:48])
	if count64 > math.MaxUint32 || count64 > uint64(maxPlatformInt()) || elements64 > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: counts exceed format capacity", ErrInvalidSparseHNSWFile)
	}
	count, elements := int(count64), int(elements64)
	minimum, err := checkedSparseHNSWMinimumPayloadSize(count, elements)
	if err != nil || minimum > len(payload) {
		return nil, fmt.Errorf("%w: invalid payload length", ErrInvalidSparseHNSWFile)
	}
	options, err := decodeSparseHNSWOptions(header)
	if err != nil {
		return nil, err
	}
	entry64 := binary.LittleEndian.Uint64(header[60:68])
	maxLevel := int(int32(binary.LittleEndian.Uint32(header[68:72])))
	if maxLevel < -1 || maxLevel > MaxHNSWLevel {
		return nil, fmt.Errorf("%w: invalid maximum level", ErrInvalidSparseHNSWFile)
	}
	entryPoint := -1
	if entry64 != math.MaxUint64 {
		if entry64 >= count64 {
			return nil, fmt.Errorf("%w: entry point out of range", ErrInvalidSparseHNSWFile)
		}
		entryPoint = int(entry64)
	}
	if (count == 0 && (entryPoint != -1 || maxLevel != -1 || elements != 0)) ||
		(count != 0 && (entryPoint < 0 || maxLevel < 0)) {
		return nil, fmt.Errorf("%w: inconsistent entry point", ErrInvalidSparseHNSWFile)
	}

	index := &SparseHNSWIndex{
		options:       options,
		keys:          make([]uint64, count),
		offsets:       make([]int, count+1),
		indices:       make([]uint32, 0, elements),
		values:        make([]float32, 0, elements),
		positions:     make(map[uint64]int, count),
		levels:        make([]int, count),
		neighbors:     make([][][]int, count),
		entryPoint:    entryPoint,
		maxLevel:      maxLevel,
		levelRNGState: binary.LittleEndian.Uint64(header[80:88]),
	}
	offset := 0
	for position := 0; position < count; position++ {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if !sparseHNSWPayloadAvailable(payload, offset, sparseHNSWRecordFixedBytes+sparseHNSWLevelFixedBytes) {
			return nil, fmt.Errorf("%w: truncated node %d", ErrInvalidSparseHNSWFile, position)
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, duplicate := index.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key %d", ErrInvalidSparseHNSWFile, key)
		}
		index.keys[position] = key
		index.positions[key] = position
		level64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if level64 > MaxHNSWLevel {
			return nil, fmt.Errorf("%w: invalid node level", ErrInvalidSparseHNSWFile)
		}
		level := int(level64)
		index.levels[position] = level
		index.neighbors[position] = make([][]int, level+1)
		nonzero64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if nonzero64 > elements64-uint64(len(index.indices)) || nonzero64 > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: invalid nonzero count", ErrInvalidSparseHNSWFile)
		}
		nonzero := int(nonzero64)
		if !sparseHNSWPayloadAvailable(payload, offset, nonzero*sparseHNSWElementBytes+sparseHNSWLevelFixedBytes) {
			return nil, fmt.Errorf("%w: truncated sparse vector", ErrInvalidSparseHNSWFile)
		}
		var previous uint32
		for element := 0; element < nonzero; element++ {
			coordinate := binary.LittleEndian.Uint32(payload[offset : offset+4])
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset+4 : offset+8]))
			offset += sparseHNSWElementBytes
			if (element != 0 && coordinate <= previous) || !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: invalid sparse vector", ErrInvalidSparseHNSWFile)
			}
			previous = coordinate
			index.indices = append(index.indices, coordinate)
			index.values = append(index.values, value)
		}
		index.offsets[position+1] = len(index.indices)
		for currentLevel := 0; currentLevel <= level; currentLevel++ {
			if !sparseHNSWPayloadAvailable(payload, offset, sparseHNSWLevelFixedBytes) {
				return nil, fmt.Errorf("%w: truncated neighbors", ErrInvalidSparseHNSWFile)
			}
			degree64 := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			limit := options.M
			if currentLevel == 0 {
				limit *= 2
			}
			if degree64 > uint64(limit) || degree64 > count64 {
				return nil, fmt.Errorf("%w: node degree out of range", ErrInvalidSparseHNSWFile)
			}
			degree := int(degree64)
			if !sparseHNSWPayloadAvailable(payload, offset, degree*4) {
				return nil, fmt.Errorf("%w: truncated neighbor positions", ErrInvalidSparseHNSWFile)
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
					return nil, fmt.Errorf("%w: invalid neighbor reference", ErrInvalidSparseHNSWFile)
				}
				neighbor := int(neighbor64)
				if _, duplicate := seen[neighbor]; duplicate {
					return nil, fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidSparseHNSWFile)
				}
				seen[neighbor] = struct{}{}
				neighbors[neighborIndex] = neighbor
			}
			index.neighbors[position][currentLevel] = neighbors
		}
	}
	if len(index.indices) != elements || offset != len(payload) {
		return nil, fmt.Errorf("%w: inconsistent payload contents", ErrInvalidSparseHNSWFile)
	}
	if err := validateSparseHNSWIndex(ctx, index); err != nil {
		return nil, err
	}
	return index, nil
}

func decodeSparseHNSWOptions(header []byte) (HNSWBuildOptions, error) {
	m64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	ef64 := uint64(binary.LittleEndian.Uint32(header[52:56]))
	if m64 > uint64(maxPlatformInt()) || ef64 > uint64(maxPlatformInt()) {
		return HNSWBuildOptions{}, fmt.Errorf("%w: options exceed platform capacity", ErrInvalidSparseHNSWFile)
	}
	options := HNSWBuildOptions{
		Metric:         Metric(header[56]),
		M:              int(m64),
		EFConstruction: int(ef64),
		Seed:           binary.LittleEndian.Uint64(header[72:80]),
	}
	if err := options.Validate(); err != nil || options.Metric != MetricIP {
		return HNSWBuildOptions{}, fmt.Errorf("%w: invalid build options", ErrInvalidSparseHNSWFile)
	}
	return options, nil
}

func validateSparseHNSWIndex(ctx context.Context, index *SparseHNSWIndex) error {
	if ctx == nil {
		return errors.New("core: nil sparse HNSW validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidSparseHNSWFile)
	}
	if err := index.options.Validate(); err != nil || index.options.Metric != MetricIP {
		return fmt.Errorf("%w: invalid build options", ErrInvalidSparseHNSWFile)
	}
	if index.options.M > math.MaxUint32 || index.options.EFConstruction > math.MaxUint32 {
		return fmt.Errorf("%w: options exceed format capacity", ErrInvalidSparseHNSWFile)
	}
	count := len(index.keys)
	if uint64(count) > math.MaxUint32 || len(index.offsets) != count+1 || len(index.indices) != len(index.values) ||
		len(index.positions) != count || len(index.levels) != count || len(index.neighbors) != count ||
		index.offsets[0] != 0 || index.offsets[count] != len(index.indices) {
		return fmt.Errorf("%w: inconsistent graph storage", ErrInvalidSparseHNSWFile)
	}
	if count == 0 {
		if index.entryPoint != -1 || index.maxLevel != -1 || len(index.indices) != 0 {
			return fmt.Errorf("%w: inconsistent empty graph", ErrInvalidSparseHNSWFile)
		}
		return nil
	}
	if index.entryPoint < 0 || index.entryPoint >= count || index.maxLevel < 0 || index.maxLevel > MaxHNSWLevel {
		return fmt.Errorf("%w: invalid graph entry point", ErrInvalidSparseHNSWFile)
	}
	derivedMax := -1
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		start, end := index.offsets[position], index.offsets[position+1]
		if mapped, found := index.positions[key]; !found || mapped != position ||
			start < 0 || end < start || end > len(index.indices) || uint64(end-start) > math.MaxUint32 {
			return fmt.Errorf("%w: invalid key or CSR offsets", ErrInvalidSparseHNSWFile)
		}
		vector := index.sparseVectorAt(position)
		if _, err := sparseHNSWScore(vector, SparseVector{}); err != nil {
			return fmt.Errorf("%w: invalid sparse vector", ErrInvalidSparseHNSWFile)
		}
		level := index.levels[position]
		if level < 0 || level > MaxHNSWLevel || len(index.neighbors[position]) != level+1 {
			return fmt.Errorf("%w: invalid node level storage", ErrInvalidSparseHNSWFile)
		}
		derivedMax = max(derivedMax, level)
		for currentLevel, neighbors := range index.neighbors[position] {
			limit := index.options.M
			if currentLevel == 0 {
				limit *= 2
			}
			if len(neighbors) > limit {
				return fmt.Errorf("%w: node degree exceeds limit", ErrInvalidSparseHNSWFile)
			}
			seen := make(map[int]struct{}, len(neighbors))
			for _, neighbor := range neighbors {
				if neighbor < 0 || neighbor >= count || neighbor == position || index.levels[neighbor] < currentLevel {
					return fmt.Errorf("%w: invalid neighbor reference", ErrInvalidSparseHNSWFile)
				}
				if _, duplicate := seen[neighbor]; duplicate {
					return fmt.Errorf("%w: duplicate neighbor reference", ErrInvalidSparseHNSWFile)
				}
				seen[neighbor] = struct{}{}
			}
		}
	}
	if derivedMax != index.maxLevel || index.levels[index.entryPoint] != index.maxLevel {
		return fmt.Errorf("%w: inconsistent maximum level", ErrInvalidSparseHNSWFile)
	}
	return nil
}

func checkedSparseHNSWPayloadSize(index *SparseHNSWIndex) (int, error) {
	minimum, err := checkedSparseHNSWMinimumPayloadSize(len(index.keys), len(index.indices))
	if err != nil {
		return 0, err
	}
	total := uint64(minimum)
	for position, level := range index.levels {
		total += uint64(level) * sparseHNSWLevelFixedBytes
		for _, neighbors := range index.neighbors[position] {
			total += uint64(len(neighbors)) * 4
			if total > uint64(maxPlatformInt()-sparseHNSWHeaderSize) {
				return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidSparseHNSWFile)
			}
		}
	}
	return int(total), nil
}

func checkedSparseHNSWMinimumPayloadSize(count, elements int) (int, error) {
	if count < 0 || elements < 0 {
		return 0, fmt.Errorf("%w: invalid size", ErrInvalidSparseHNSWFile)
	}
	base := uint64(count) * (sparseHNSWRecordFixedBytes + sparseHNSWLevelFixedBytes)
	values := uint64(elements) * sparseHNSWElementBytes
	if (count != 0 && base/(sparseHNSWRecordFixedBytes+sparseHNSWLevelFixedBytes) != uint64(count)) ||
		(elements != 0 && values/sparseHNSWElementBytes != uint64(elements)) || base > math.MaxUint64-values {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidSparseHNSWFile)
	}
	total := base + values
	if total > uint64(maxPlatformInt()-sparseHNSWHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidSparseHNSWFile)
	}
	return int(total), nil
}

func sparseHNSWPayloadAvailable(payload []byte, offset, size int) bool {
	return offset >= 0 && size >= 0 && offset <= len(payload) && size <= len(payload)-offset
}

func readSparseHNSWFile(ctx context.Context, path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < 0 || uint64(info.Size()) > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: file exceeds platform capacity", ErrInvalidSparseHNSWFile)
	}
	encoded := make([]byte, int(info.Size()))
	for offset := 0; offset < len(encoded); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+sparseHNSWReadChunk, len(encoded))
		if _, err := io.ReadFull(file, encoded[offset:end]); err != nil {
			return nil, err
		}
		offset = end
	}
	return encoded, nil
}
