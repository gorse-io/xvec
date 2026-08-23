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
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/ailego/io"
	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
)

const (
	DefaultVamanaMaxDegree        = 64
	DefaultVamanaSearchListSize   = 100
	DefaultVamanaMaxOcclusionSize = 750
	DefaultVamanaAlpha            = float32(1.2)
	defaultVamanaBuildWorkers     = 8
	MaxVamanaDegree               = 65_535
)

var (
	ErrInvalidVamanaOptions = errors.New("core: invalid Vamana build options")
	ErrVamanaKeyNotFound    = errors.New("core: Vamana key not found")
	ErrVamanaCapacity       = errors.New("core: Vamana index capacity exceeded")
)

// VamanaBuildOptions configures deterministic single-layer graph construction.
type VamanaBuildOptions struct {
	Metric           Metric
	MaxDegree        int
	SearchListSize   int
	Alpha            float32
	MaxOcclusionSize int
	SaturateGraph    bool
}

// DefaultVamanaBuildOptions returns the pinned public construction defaults.
func DefaultVamanaBuildOptions(metric Metric) VamanaBuildOptions {
	return VamanaBuildOptions{
		Metric: metric, MaxDegree: DefaultVamanaMaxDegree,
		SearchListSize: DefaultVamanaSearchListSize,
		Alpha:          DefaultVamanaAlpha, MaxOcclusionSize: DefaultVamanaMaxOcclusionSize,
	}
}

// Validate checks graph degree, construction width, and RobustPrune settings.
func (o VamanaBuildOptions) Validate() error {
	if !o.Metric.Valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidVamanaOptions)
	}
	if o.MaxDegree <= 0 || o.MaxDegree > MaxVamanaDegree {
		return fmt.Errorf("%w: MaxDegree must be in [1,%d]", ErrInvalidVamanaOptions, MaxVamanaDegree)
	}
	if o.SearchListSize <= 0 {
		return fmt.Errorf("%w: SearchListSize must be positive", ErrInvalidVamanaOptions)
	}
	if uint64(o.SearchListSize) > math.MaxUint32 {
		return fmt.Errorf("%w: SearchListSize exceeds format capacity", ErrInvalidVamanaOptions)
	}
	if o.MaxOcclusionSize <= 0 || uint64(o.MaxOcclusionSize) > math.MaxUint32 {
		return fmt.Errorf("%w: MaxOcclusionSize must be positive", ErrInvalidVamanaOptions)
	}
	if math.IsNaN(float64(o.Alpha)) || math.IsInf(float64(o.Alpha), 0) || o.Alpha < 1 {
		return fmt.Errorf("%w: Alpha must be finite and at least 1", ErrInvalidVamanaOptions)
	}
	return nil
}

// VamanaBuilder collects original vectors for one deterministic graph build.
type VamanaBuilder struct {
	mu        sync.Mutex
	dimension int
	options   VamanaBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

// NewVamanaBuilder constructs an empty one-shot builder.
func NewVamanaBuilder(dimension int, options VamanaBuildOptions) (*VamanaBuilder, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &VamanaBuilder{dimension: dimension, options: options, positions: make(map[uint64]int)}, nil
}

// newBorrowedVamanaBuilder constructs an internal one-shot builder over
// storage owned by its caller. The caller must keep keys, vectors, and
// positions immutable until build returns.
func newBorrowedVamanaBuilder(
	dimension int,
	options VamanaBuildOptions,
	keys []uint64,
	vectors []float32,
	positions map[uint64]int,
) (*VamanaBuilder, error) {
	builder, err := NewVamanaBuilder(dimension, options)
	if err != nil {
		return nil, err
	}
	if len(keys) > maxPlatformInt()/dimension || len(vectors) != len(keys)*dimension || len(positions) != len(keys) {
		return nil, errors.New("core: inconsistent borrowed Vamana storage")
	}
	builder.keys, builder.vectors, builder.positions = keys, vectors, positions
	return builder, nil
}

// Add validates and clones one unique vector while the builder is open.
func (b *VamanaBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil Vamana builder")
	}
	if ctx == nil {
		return errors.New("core: nil Vamana add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate Vamana vector: %w", err)
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
	if len(b.vectors) > maxPlatformInt()-b.dimension {
		return ErrVamanaCapacity
	}
	b.positions[key] = len(b.keys)
	b.keys = append(b.keys, key)
	b.vectors = append(b.vectors, vector...)
	return nil
}

// Build inserts vectors in input order, applies RobustPrune and reverse-link
// updates, then selects the persisted-search medoid entry point.
func (b *VamanaBuilder) Build(ctx context.Context) (*VamanaIndex, error) {
	return b.build(ctx, 1)
}

// build constructs a deterministic graph with batched parallel candidate
// discovery. Each batch reads an immutable graph generation, then publishes
// forward and reverse edges in input order so scheduling cannot affect output.
func (b *VamanaBuilder) build(ctx context.Context, workers int) (*VamanaIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil Vamana builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil Vamana build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	distance, err := b.options.Metric.PrevalidatedDistance()
	if err != nil {
		return nil, err
	}
	index := &VamanaIndex{
		dimension: b.dimension, options: b.options,
		distance: distance,
		keys:     b.keys, vectors: b.vectors, positions: b.positions,
		neighbors: make([][]int, len(b.keys)), neighborDistances: make([][]float32, len(b.keys)),
		entryPoint: -1,
	}
	workers = vamanaBuildWorkers(workers, len(index.keys))
	for batchStart := 0; batchStart < len(index.keys); batchStart += workers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		batchEnd := min(batchStart+workers, len(index.keys))
		computeStart := batchStart
		if computeStart == 0 {
			index.entryPoint = 0
			computeStart++
			if computeStart == batchEnd {
				continue
			}
		}
		selected := make([][]vamanaDistanceNode, batchEnd-computeStart)
		err := parallel.ParallelFor(ctx, len(selected), workers, func(ctx context.Context, offset int) error {
			position := computeStart + offset
			candidates, err := index.searchBuildCandidates(
				ctx, index.vectorAt(position), index.entryPoint, index.options.SearchListSize,
			)
			if err != nil {
				return err
			}
			selected[offset], err = index.robustPrune(ctx, position, candidates)
			return err
		})
		if err != nil {
			return nil, fmt.Errorf("core: construct Vamana batch at node %d: %w", computeStart, err)
		}
		for offset, neighbors := range selected {
			position := computeStart + offset
			index.setVamanaNeighbors(position, neighbors)
			for _, neighbor := range neighbors {
				if err := index.addReverseEdge(ctx, position, neighbor.position, neighbor.distance); err != nil {
					return nil, fmt.Errorf("core: construct Vamana node %d: %w", position, err)
				}
			}
		}
	}
	entry, err := index.calculateMedoid(ctx)
	if err != nil {
		return nil, err
	}
	index.entryPoint = entry
	if err := validateVamanaIndex(ctx, index); err != nil {
		return nil, err
	}
	b.built = true
	b.keys = nil
	b.vectors = nil
	b.positions = nil
	return index, nil
}

func vamanaBuildWorkers(workers, count int) int {
	if workers <= 0 {
		// The worker count also defines the immutable graph-generation batch.
		// Keep the default fixed so identical inputs do not produce different
		// graph topology and recall on machines with different CPU counts.
		workers = defaultVamanaBuildWorkers
	}
	return max(1, min(workers, max(1, count)))
}

// VamanaIndex stores original FP32 vectors and one bounded directed graph.
// Readers share an immutable generation while Add publishes copy-on-write.
type VamanaIndex struct {
	streamMu          sync.Mutex
	mu                sync.RWMutex
	pruneScratch      sync.Pool
	dimension         int
	options           VamanaBuildOptions
	distance          mathutil.DenseDistance
	keys              []uint64
	vectors           []float32
	positions         map[uint64]int
	neighbors         [][]int
	neighborDistances [][]float32
	entryPoint        int
}

func (i *VamanaIndex) Dimension() int {
	if i == nil {
		return 0
	}
	return i.dimension
}

func (i *VamanaIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.options.Metric
}

func (i *VamanaIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

func (i *VamanaIndex) BuildOptions() VamanaBuildOptions {
	if i == nil {
		return VamanaBuildOptions{}
	}
	return i.options
}

func (i *VamanaIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil {
		return nil, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return nil, false
	}
	return slices.Clone(i.vectorAt(position)), true
}

func (i *VamanaIndex) EntryPoint() (uint64, bool) {
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

// Neighbors returns cloned outbound neighbor keys in prune-selection order.
func (i *VamanaIndex) Neighbors(key uint64) ([]uint64, error) {
	if i == nil {
		return nil, errors.New("core: nil Vamana index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return nil, fmt.Errorf("%w: %d", ErrVamanaKeyNotFound, key)
	}
	result := make([]uint64, len(i.neighbors[position]))
	for offset, neighbor := range i.neighbors[position] {
		result[offset] = i.keys[neighbor]
	}
	return result, nil
}

func (i *VamanaIndex) insertNode(ctx context.Context, position int) error {
	if i.entryPoint < 0 {
		i.entryPoint = position
		return nil
	}
	candidates, err := i.searchBuildCandidates(ctx, i.vectorAt(position), i.entryPoint, i.options.SearchListSize)
	if err != nil {
		return err
	}
	selected, err := i.robustPrune(ctx, position, candidates)
	if err != nil {
		return err
	}
	i.setVamanaNeighbors(position, selected)
	for _, neighbor := range selected {
		if err := i.addReverseEdge(ctx, position, neighbor.position, neighbor.distance); err != nil {
			return err
		}
	}
	return nil
}

type vamanaDistanceNode struct {
	position int
	distance float32
}

type vamanaPruneScratch struct {
	deduplicated []vamanaDistanceNode
	seen         map[int]struct{}
	factors      []float32
	consumed     []bool
}

func (i *VamanaIndex) searchBuildCandidates(ctx context.Context, query []float32, entry, capacity int) ([]vamanaDistanceNode, error) {
	limit := min(capacity, len(i.keys))
	if limit <= 0 {
		return []vamanaDistanceNode{}, nil
	}
	better := func(left, right vamanaDistanceNode) bool { return vamanaDistanceBetter(left, right) }
	worse := func(left, right vamanaDistanceNode) bool { return vamanaDistanceBetter(right, left) }
	frontier := container.NewHeap(better)
	retained := container.NewHeap(worse)
	visited := make([]bool, len(i.keys))
	distance, err := i.graphDistance(query, i.vectorAt(entry))
	if err != nil {
		return nil, err
	}
	start := vamanaDistanceNode{position: entry, distance: distance}
	visited[entry] = true
	frontier.Push(start)
	retained.Push(start)
	for frontier.Len() != 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current, _ := frontier.Pop()
		worst, hasWorst := retained.Peek()
		if retained.Len() >= limit && hasWorst && worst.distance < current.distance {
			break
		}
		for _, neighbor := range i.neighbors[current.position] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			distance, err := i.graphDistance(query, i.vectorAt(neighbor))
			if err != nil {
				return nil, err
			}
			node := vamanaDistanceNode{position: neighbor, distance: distance}
			worst, hasWorst = retained.Peek()
			if retained.Len() < limit || !hasWorst || vamanaDistanceBetter(node, worst) {
				frontier.Push(node)
				retained.Push(node)
				if retained.Len() > limit {
					_, _ = retained.Pop()
				}
			}
		}
	}
	result := retained.Values()
	slices.SortFunc(result, compareVamanaDistanceNodes)
	return result, nil
}

func (i *VamanaIndex) robustPrune(ctx context.Context, owner int, candidates []vamanaDistanceNode) ([]vamanaDistanceNode, error) {
	slices.SortFunc(candidates, compareVamanaDistanceNodes)
	value := i.pruneScratch.Get()
	var scratch *vamanaPruneScratch
	if value == nil {
		scratch = &vamanaPruneScratch{seen: make(map[int]struct{}, len(candidates))}
	} else {
		scratch = value.(*vamanaPruneScratch)
	}
	scratch.deduplicated = scratch.deduplicated[:0]
	clear(scratch.seen)
	defer i.pruneScratch.Put(scratch)
	for _, candidate := range candidates {
		if candidate.position == owner {
			continue
		}
		if _, found := scratch.seen[candidate.position]; found {
			continue
		}
		scratch.seen[candidate.position] = struct{}{}
		scratch.deduplicated = append(scratch.deduplicated, candidate)
		if len(scratch.deduplicated) == i.options.MaxOcclusionSize {
			break
		}
	}
	candidates = scratch.deduplicated
	scratch.factors = slices.Grow(scratch.factors[:0], len(candidates))[:len(candidates)]
	scratch.consumed = slices.Grow(scratch.consumed[:0], len(candidates))[:len(candidates)]
	clear(scratch.factors)
	clear(scratch.consumed)
	factors, consumed := scratch.factors, scratch.consumed
	selected := make([]vamanaDistanceNode, 0, min(i.options.MaxDegree, len(candidates)))
	for alpha := float32(1); alpha <= i.options.Alpha+1e-6 && len(selected) < i.options.MaxDegree; {
		for candidateIndex, candidate := range candidates {
			if len(selected) == i.options.MaxDegree {
				break
			}
			if consumed[candidateIndex] || factors[candidateIndex] > alpha {
				continue
			}
			consumed[candidateIndex] = true
			factors[candidateIndex] = math.MaxFloat32
			selected = append(selected, candidate)
			for next := candidateIndex + 1; next < len(candidates); next++ {
				if factors[next] > i.options.Alpha {
					continue
				}
				if next&63 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				between, err := i.graphDistance(i.vectorAt(candidate.position), i.vectorAt(candidates[next].position))
				if err != nil {
					return nil, err
				}
				if between <= 0 {
					factors[next] = math.MaxFloat32
				} else {
					factors[next] = max(factors[next], candidates[next].distance/between)
				}
			}
		}
		nextAlpha := alpha * 1.2
		if nextAlpha <= alpha {
			break
		}
		alpha = nextAlpha
	}
	if i.options.SaturateGraph && i.options.Alpha > 1 {
		for index, candidate := range candidates {
			if len(selected) == i.options.MaxDegree {
				break
			}
			if !consumed[index] {
				selected = append(selected, candidate)
			}
		}
	}
	return selected, nil
}

func (i *VamanaIndex) addReverseEdge(ctx context.Context, node, owner int, distance float32) error {
	for _, existing := range i.neighbors[owner] {
		if existing == node {
			return nil
		}
	}
	if len(i.neighbors[owner]) < i.options.MaxDegree {
		i.neighbors[owner] = append(i.neighbors[owner], node)
		i.neighborDistances[owner] = append(i.neighborDistances[owner], distance)
		return nil
	}
	candidates := make([]vamanaDistanceNode, 0, len(i.neighbors[owner])+1)
	for offset, neighbor := range i.neighbors[owner] {
		candidates = append(candidates, vamanaDistanceNode{position: neighbor, distance: i.neighborDistances[owner][offset]})
	}
	candidates = append(candidates, vamanaDistanceNode{position: node, distance: distance})
	selected, err := i.robustPrune(ctx, owner, candidates)
	if err != nil {
		return err
	}
	i.setVamanaNeighbors(owner, selected)
	return nil
}

func (i *VamanaIndex) setVamanaNeighbors(owner int, selected []vamanaDistanceNode) {
	if cap(i.neighbors[owner]) < len(selected) {
		i.neighbors[owner] = make([]int, len(selected))
	} else {
		i.neighbors[owner] = i.neighbors[owner][:len(selected)]
	}
	if cap(i.neighborDistances[owner]) < len(selected) {
		i.neighborDistances[owner] = make([]float32, len(selected))
	} else {
		i.neighborDistances[owner] = i.neighborDistances[owner][:len(selected)]
	}
	for offset, candidate := range selected {
		i.neighbors[owner][offset] = candidate.position
		i.neighborDistances[owner][offset] = candidate.distance
	}
}

func (i *VamanaIndex) graphDistance(left, right []float32) (float32, error) {
	distance := i.distance
	if distance == nil {
		var err error
		distance, err = i.options.Metric.PrevalidatedDistance()
		if err != nil {
			return 0, err
		}
	}
	score, err := distance(left, right)
	if err != nil {
		return 0, err
	}
	if i.options.Metric == MetricIP {
		return -score, nil
	}
	return score, nil
}

func (i *VamanaIndex) calculateMedoid(ctx context.Context) (int, error) {
	if len(i.keys) == 0 {
		return -1, nil
	}
	centroid := make([]float64, i.dimension)
	for position := range i.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return -1, err
			}
		}
		for component, value := range i.vectorAt(position) {
			centroid[component] += float64(value)
		}
	}
	scale := 1 / float64(len(i.keys))
	for component := range centroid {
		centroid[component] *= scale
	}
	best, bestDistance := 0, math.Inf(1)
	for position := range i.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return -1, err
			}
		}
		var distance float64
		for component, value := range i.vectorAt(position) {
			delta := float64(value) - centroid[component]
			distance += delta * delta
		}
		if distance < bestDistance {
			best, bestDistance = position, distance
		}
	}
	return best, nil
}

func (i *VamanaIndex) vectorAt(position int) []float32 {
	start := position * i.dimension
	return i.vectors[start : start+i.dimension]
}

func vamanaDistanceBetter(left, right vamanaDistanceNode) bool {
	if left.distance == right.distance {
		return left.position < right.position
	}
	return left.distance < right.distance
}

func compareVamanaDistanceNodes(left, right vamanaDistanceNode) int {
	if vamanaDistanceBetter(left, right) {
		return -1
	}
	if vamanaDistanceBetter(right, left) {
		return 1
	}
	return 0
}

func cloneVamanaIndex(ctx context.Context, source *VamanaIndex) (*VamanaIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil Vamana clone context")
	}
	if source == nil {
		return nil, errors.New("core: nil Vamana source index")
	}
	clone := &VamanaIndex{
		dimension: source.dimension, options: source.options,
		distance: source.distance,
		keys:     slices.Clone(source.keys), vectors: slices.Clone(source.vectors),
		positions: cloneUint64Positions(source.positions), entryPoint: source.entryPoint,
		neighbors: make([][]int, len(source.neighbors)), neighborDistances: make([][]float32, len(source.neighborDistances)),
	}
	for position := range clone.neighbors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		clone.neighbors[position] = slices.Clone(source.neighbors[position])
		clone.neighborDistances[position] = slices.Clone(source.neighborDistances[position])
	}
	return clone, nil
}

func validateVamanaIndex(ctx context.Context, index *VamanaIndex) error {
	if ctx == nil {
		return errors.New("core: nil Vamana validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil || index.dimension <= 0 || index.dimension > MaxRotationDimension {
		return errors.New("core: invalid Vamana index")
	}
	if err := index.options.Validate(); err != nil {
		return err
	}
	count := len(index.keys)
	if count > maxPlatformInt()/index.dimension || len(index.vectors) != count*index.dimension || len(index.positions) != count ||
		len(index.neighbors) != count || len(index.neighborDistances) != count {
		return errors.New("core: inconsistent Vamana storage")
	}
	if (count == 0 && index.entryPoint != -1) || (count > 0 && (index.entryPoint < 0 || index.entryPoint >= count)) {
		return errors.New("core: invalid Vamana entry point")
	}
	seenKeys := make(map[uint64]struct{}, count)
	for position, key := range index.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if _, found := seenKeys[key]; found || index.positions[key] != position {
			return errors.New("core: invalid Vamana key map")
		}
		seenKeys[key] = struct{}{}
		if err := validateTrainingVector(index.vectorAt(position), index.dimension); err != nil {
			return err
		}
		if len(index.neighbors[position]) > index.options.MaxDegree || len(index.neighbors[position]) != len(index.neighborDistances[position]) {
			return errors.New("core: invalid Vamana degree")
		}
		seenNeighbors := make(map[int]struct{}, len(index.neighbors[position]))
		for offset, neighbor := range index.neighbors[position] {
			if neighbor < 0 || neighbor >= count || neighbor == position {
				return errors.New("core: invalid Vamana neighbor")
			}
			if _, found := seenNeighbors[neighbor]; found {
				return errors.New("core: duplicate Vamana neighbor")
			}
			seenNeighbors[neighbor] = struct{}{}
			distance := index.neighborDistances[position][offset]
			if math.IsNaN(float64(distance)) || math.IsInf(float64(distance), 0) {
				return errors.New("core: invalid Vamana neighbor distance")
			}
		}
	}
	return nil
}

const (
	DefaultVamanaEFSearch            = 200
	MaxVamanaEFSearch                = 2048
	DefaultVamanaBruteForceThreshold = 1000
	DefaultVamanaPrefetchOffset      = 8
)

var ErrInvalidVamanaEF = errors.New("core: Vamana EF must be in [1, 2048]")

// VamanaSearchOptions combines common result controls with beam width and
// portable cache-warming hints.
type VamanaSearchOptions struct {
	SearchOptions
	EFSearch       int
	PrefetchOffset uint32
	PrefetchLines  uint32
}

func (o VamanaSearchOptions) Validate() error {
	if err := o.SearchOptions.Validate(); err != nil {
		return err
	}
	if o.EFSearch <= 0 || o.EFSearch > MaxVamanaEFSearch {
		return ErrInvalidVamanaEF
	}
	return nil
}

func (i *VamanaIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.searchVamana(ctx, query, VamanaSearchOptions{
		SearchOptions: SearchOptions{TopK: k}, EFSearch: DefaultVamanaEFSearch,
		PrefetchOffset: DefaultVamanaPrefetchOffset,
	}, false)
}

func (i *VamanaIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.searchVamana(ctx, query, VamanaSearchOptions{
		SearchOptions: options, EFSearch: DefaultVamanaEFSearch,
		PrefetchOffset: DefaultVamanaPrefetchOffset,
	}, true)
}

// SearchVamana executes one metric-aware beam search. EFSearch smaller than
// TopK is raised to TopK, matching the pinned interface behavior.
func (i *VamanaIndex) SearchVamana(ctx context.Context, query []float32, options VamanaSearchOptions) ([]Result, error) {
	return i.searchVamana(ctx, query, options, true)
}

func (i *VamanaIndex) searchVamana(ctx context.Context, query []float32, options VamanaSearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil Vamana index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil Vamana search context")
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
			return nil, errors.New("core: negative Vamana top-k")
		}
		if options.Radius < 0 || math.IsNaN(float64(options.Radius)) || math.IsInf(float64(options.Radius), 0) {
			return nil, ErrInvalidRadius
		}
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if _, err := i.options.Metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate Vamana query: %w", err)
	}

	i.mu.RLock()
	defer i.mu.RUnlock()
	if options.TopK == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}
	if len(i.keys) <= DefaultVamanaBruteForceThreshold {
		return topKCandidatesWithOptions(ctx, i.options.Metric, query, options.SearchOptions, len(i.keys), func(position int) Candidate {
			return Candidate{Key: i.keys[position], Vector: i.vectorAt(position)}
		}, requirePositiveTopK)
	}
	scoreAt := func(position int) (float32, error) {
		return i.options.Metric.Compute(query, i.vectorAt(position))
	}
	prefetch := func(neighbors []int) {
		prefetchDenseHNSWNeighbors(i.vectors, i.dimension, neighbors, options.PrefetchOffset, options.PrefetchLines)
	}
	return searchVamanaGraph(ctx, i.options.Metric, i.keys, i.neighbors, i.entryPoint, options, scoreAt, prefetch)
}

func searchVamanaGraph(
	ctx context.Context,
	metric Metric,
	keys []uint64,
	neighbors [][]int,
	entry int,
	options VamanaSearchOptions,
	scoreAt func(int) (float32, error),
	prefetch func([]int),
) ([]Result, error) {
	capacity := min(len(keys), max(options.TopK, options.EFSearch))
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(metric, left, right) }
	resultBetter := func(left, right hnswScoredNode) bool {
		if left.score == right.score {
			return keys[left.position] < keys[right.position]
		}
		return metric.Better(left.score, right.score)
	}
	worse := func(left, right hnswScoredNode) bool { return resultBetter(right, left) }
	frontier := container.NewHeap(better)
	accepted := container.NewHeap(worse)
	visited := make([]bool, len(keys))
	score, err := scoreAt(entry)
	if err != nil {
		return nil, fmt.Errorf("core: score Vamana entry point: %w", err)
	}
	start := hnswScoredNode{position: entry, score: score}
	visited[entry] = true
	frontier.Push(start)
	if acceptVamanaNode(metric, keys, start, options.SearchOptions) {
		accepted.Push(start)
	}
	for frontier.Len() != 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current, _ := frontier.Pop()
		worst, hasWorst := accepted.Peek()
		if accepted.Len() >= capacity && hasWorst && metric.Better(worst.score, current.score) {
			break
		}
		adjacent := neighbors[current.position]
		if prefetch != nil {
			prefetch(adjacent)
		}
		for _, neighbor := range adjacent {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := scoreAt(neighbor)
			if err != nil {
				return nil, fmt.Errorf("core: score Vamana node %d: %w", neighbor, err)
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = accepted.Peek()
			if accepted.Len() < capacity || !hasWorst || !metric.Better(worst.score, node.score) {
				frontier.Push(node)
				if acceptVamanaNode(metric, keys, node, options.SearchOptions) {
					accepted.Push(node)
					if accepted.Len() > capacity {
						_, _ = accepted.Pop()
					}
				}
			}
		}
	}
	resultNodes := accepted.Values()
	slices.SortFunc(resultNodes, func(left, right hnswScoredNode) int {
		if resultBetter(left, right) {
			return -1
		}
		if resultBetter(right, left) {
			return 1
		}
		return 0
	})
	if len(resultNodes) > options.TopK {
		resultNodes = resultNodes[:options.TopK]
	}
	results := make([]Result, len(resultNodes))
	for position, node := range resultNodes {
		results[position] = Result{Key: keys[node.position], Score: node.score}
	}
	return results, nil
}

func acceptVamanaNode(metric Metric, keys []uint64, node hnswScoredNode, options SearchOptions) bool {
	key := keys[node.position]
	return (options.Filter == nil || options.Filter(key)) && scoreWithinRadius(metric, node.score, options.Radius)
}

// Add streams one vector into a private graph copy and atomically publishes
// the complete topology/vector/medoid generation.
func (i *VamanaIndex) Add(ctx context.Context, key uint64, vector []float32) error {
	if i == nil {
		return errors.New("core: nil Vamana index")
	}
	if ctx == nil {
		return errors.New("core: nil Vamana add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, i.dimension); err != nil {
		return fmt.Errorf("core: validate incremental Vamana vector: %w", err)
	}
	i.streamMu.Lock()
	defer i.streamMu.Unlock()

	i.mu.RLock()
	if _, found := i.positions[key]; found {
		i.mu.RUnlock()
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	if len(i.vectors) > maxPlatformInt()-i.dimension {
		i.mu.RUnlock()
		return ErrVamanaCapacity
	}
	working, err := cloneVamanaIndex(ctx, i)
	i.mu.RUnlock()
	if err != nil {
		return err
	}
	position := len(working.keys)
	working.positions[key] = position
	working.keys = append(working.keys, key)
	working.vectors = append(working.vectors, vector...)
	working.neighbors = append(working.neighbors, nil)
	working.neighborDistances = append(working.neighborDistances, nil)
	if err := working.insertNode(ctx, position); err != nil {
		return err
	}
	entry, err := working.calculateMedoid(ctx)
	if err != nil {
		return err
	}
	working.entryPoint = entry
	if err := validateVamanaIndex(ctx, working); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	if err := ctx.Err(); err != nil {
		i.mu.Unlock()
		return err
	}
	i.keys, i.vectors, i.positions = working.keys, working.vectors, working.positions
	i.neighbors, i.neighborDistances, i.entryPoint = working.neighbors, working.neighborDistances, working.entryPoint
	i.mu.Unlock()
	return nil
}

var (
	_ DenseIndex         = (*VamanaIndex)(nil)
	_ DenseProvider      = (*VamanaIndex)(nil)
	_ DenseQuerySearcher = (*VamanaIndex)(nil)
)

const (
	vamanaFileVersion  = 1
	vamanaHeaderSize   = 128
	vamanaFlagSaturate = uint32(1)
)

var (
	vamanaFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'V', 'M', 'N', 'A'}

	ErrInvalidVamanaFile        = errors.New("core: invalid Vamana file")
	ErrVamanaChecksumMismatch   = errors.New("core: Vamana checksum mismatch")
	ErrUnsupportedVamanaVersion = errors.New("core: unsupported Vamana file version")
)

// Save atomically publishes one complete native graph generation.
func (i *VamanaIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil Vamana save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidVamanaFile)
	}
	if i == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidVamanaFile)
	}
	i.mu.RLock()
	snapshot, err := cloneVamanaIndex(ctx, i)
	i.mu.RUnlock()
	if err != nil {
		return err
	}
	encoded, err := encodeVamanaIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save Vamana file: %w", err)
	}
	return nil
}

// OpenVamanaIndex reads and verifies a native Go Vamana artifact.
func OpenVamanaIndex(ctx context.Context, path string) (*VamanaIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil Vamana open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidVamanaFile)
	}
	encoded, err := readHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read Vamana file: %w", err)
	}
	index, err := decodeVamanaIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open Vamana file: %w", err)
	}
	return index, nil
}

func encodeVamanaIndex(ctx context.Context, index *VamanaIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil Vamana encode context")
	}
	if err := validateVamanaIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidVamanaFile, err)
	}
	if err := validateVamanaFormatOptions(index.options); err != nil {
		return nil, err
	}
	if uint64(len(index.keys)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: node count exceeds format capacity", ErrInvalidVamanaFile)
	}
	edgeCount := 0
	for _, adjacent := range index.neighbors {
		if edgeCount > maxPlatformInt()-len(adjacent) {
			return nil, fmt.Errorf("%w: edge count exceeds platform capacity", ErrInvalidVamanaFile)
		}
		edgeCount += len(adjacent)
	}
	payloadSize, err := checkedVamanaPayloadSize(len(index.keys), index.dimension, edgeCount)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint64(payload, key)
	}
	for position, value := range index.vectors {
		if position&16383 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
	}
	for position, adjacent := range index.neighbors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint32(payload, uint32(len(adjacent)))
		for _, neighbor := range adjacent {
			payload = binary.LittleEndian.AppendUint32(payload, uint32(neighbor))
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidVamanaFile)
	}
	header := make([]byte, vamanaHeaderSize)
	copy(header[:8], vamanaFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], vamanaFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], vamanaHeaderSize)
	if index.options.SaturateGraph {
		binary.LittleEndian.PutUint32(header[12:16], vamanaFlagSaturate)
	}
	binary.LittleEndian.PutUint64(header[16:24], uint64(vamanaHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(index.keys)))
	binary.LittleEndian.PutUint64(header[40:48], uint64(edgeCount))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.dimension))
	header[52] = byte(index.options.Metric)
	binary.LittleEndian.PutUint32(header[56:60], uint32(index.options.MaxDegree))
	binary.LittleEndian.PutUint32(header[60:64], uint32(index.options.SearchListSize))
	binary.LittleEndian.PutUint32(header[64:68], uint32(index.options.MaxOcclusionSize))
	binary.LittleEndian.PutUint32(header[68:72], math.Float32bits(index.options.Alpha))
	entry := uint64(math.MaxUint64)
	if index.entryPoint >= 0 {
		entry = uint64(index.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[72:80], entry)
	binary.LittleEndian.PutUint32(header[80:84], hashutil.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], hashutil.CRC32C(header[:124]))
	return append(header, payload...), nil
}

func decodeVamanaIndex(ctx context.Context, encoded []byte) (*VamanaIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil Vamana decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < vamanaHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidVamanaFile)
	}
	header := encoded[:vamanaHeaderSize]
	if !bytes.Equal(header[:8], vamanaFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidVamanaFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != vamanaFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedVamanaVersion, version)
	}
	flags := binary.LittleEndian.Uint32(header[12:16])
	if binary.LittleEndian.Uint16(header[10:12]) != vamanaHeaderSize || flags&^vamanaFlagSaturate != 0 ||
		!hnswAllZero(header[53:56]) || !hnswAllZero(header[84:124]) {
		return nil, fmt.Errorf("%w: invalid header fields", ErrInvalidVamanaFile)
	}
	if got, want := hashutil.CRC32C(header[:124]), binary.LittleEndian.Uint32(header[124:128]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrVamanaChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-vamanaHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidVamanaFile)
	}
	payload := encoded[vamanaHeaderSize:]
	if got, want := hashutil.CRC32C(payload), binary.LittleEndian.Uint32(header[80:84]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrVamanaChecksumMismatch, got, want)
	}
	count64 := binary.LittleEndian.Uint64(header[32:40])
	edgeCount64 := binary.LittleEndian.Uint64(header[40:48])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	if count64 > math.MaxUint32 {
		return nil, fmt.Errorf("%w: node count exceeds format capacity", ErrInvalidVamanaFile)
	}
	for _, value := range []uint64{count64, edgeCount64, dimension64} {
		if value > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: field exceeds platform capacity", ErrInvalidVamanaFile)
		}
	}
	count, edgeCount, dimension := int(count64), int(edgeCount64), int(dimension64)
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: invalid dimension", ErrInvalidVamanaFile)
	}
	maxDegree64 := uint64(binary.LittleEndian.Uint32(header[56:60]))
	searchListSize64 := uint64(binary.LittleEndian.Uint32(header[60:64]))
	maxOcclusionSize64 := uint64(binary.LittleEndian.Uint32(header[64:68]))
	for _, value := range []uint64{maxDegree64, searchListSize64, maxOcclusionSize64} {
		if value > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: option exceeds platform capacity", ErrInvalidVamanaFile)
		}
	}
	options := VamanaBuildOptions{
		Metric: Metric(header[52]), MaxDegree: int(maxDegree64),
		SearchListSize:   int(searchListSize64),
		MaxOcclusionSize: int(maxOcclusionSize64),
		Alpha:            math.Float32frombits(binary.LittleEndian.Uint32(header[68:72])), SaturateGraph: flags&vamanaFlagSaturate != 0,
	}
	if err := options.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidVamanaFile, err)
	}
	expectedSize, err := checkedVamanaPayloadSize(count, dimension, edgeCount)
	if err != nil || expectedSize != len(payload) {
		return nil, fmt.Errorf("%w: invalid payload size", ErrInvalidVamanaFile)
	}
	entry64 := binary.LittleEndian.Uint64(header[72:80])
	entry := -1
	if count == 0 {
		if entry64 != math.MaxUint64 {
			return nil, fmt.Errorf("%w: empty graph entry point", ErrInvalidVamanaFile)
		}
	} else {
		if entry64 >= uint64(count) {
			return nil, fmt.Errorf("%w: entry point out of range", ErrInvalidVamanaFile)
		}
		entry = int(entry64)
	}
	index := &VamanaIndex{
		dimension: dimension, options: options, keys: make([]uint64, count),
		vectors: make([]float32, count*dimension), positions: make(map[uint64]int, count),
		neighbors: make([][]int, count), neighborDistances: make([][]float32, count), entryPoint: entry,
	}
	index.distance, err = options.Metric.PrevalidatedDistance()
	if err != nil {
		return nil, err
	}
	offset := 0
	for position := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, found := index.positions[key]; found {
			return nil, fmt.Errorf("%w: duplicate key", ErrInvalidVamanaFile)
		}
		index.keys[position], index.positions[key] = key, position
	}
	for position := range index.vectors {
		if position&16383 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		index.vectors[position] = math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
	}
	decodedEdges := 0
	for position := range index.neighbors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		degree := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if degree > uint64(options.MaxDegree) || degree > uint64(edgeCount-decodedEdges) || degree > uint64((len(payload)-offset)/4) {
			return nil, fmt.Errorf("%w: invalid degree", ErrInvalidVamanaFile)
		}
		adjacent := make([]int, int(degree))
		for neighborIndex := range adjacent {
			if (decodedEdges+neighborIndex)&16383 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			neighbor := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if neighbor >= uint64(count) {
				return nil, fmt.Errorf("%w: neighbor out of range", ErrInvalidVamanaFile)
			}
			adjacent[neighborIndex] = int(neighbor)
		}
		index.neighbors[position] = adjacent
		decodedEdges += len(adjacent)
	}
	if offset != len(payload) || decodedEdges != edgeCount {
		return nil, fmt.Errorf("%w: inconsistent edge payload", ErrInvalidVamanaFile)
	}
	for position, adjacent := range index.neighbors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		distances := make([]float32, len(adjacent))
		for neighborIndex, neighbor := range adjacent {
			distance, err := index.graphDistance(index.vectorAt(position), index.vectorAt(neighbor))
			if err != nil {
				return nil, fmt.Errorf("%w: neighbor distance: %v", ErrInvalidVamanaFile, err)
			}
			distances[neighborIndex] = distance
		}
		index.neighborDistances[position] = distances
	}
	if err := validateVamanaIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidVamanaFile, err)
	}
	return index, nil
}

func validateVamanaFormatOptions(options VamanaBuildOptions) error {
	for _, value := range []int{options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize} {
		if value < 0 || uint64(value) > math.MaxUint32 {
			return fmt.Errorf("%w: options exceed format capacity", ErrInvalidVamanaFile)
		}
	}
	return nil
}

func checkedVamanaPayloadSize(count, dimension, edgeCount int) (int, error) {
	if count < 0 || dimension <= 0 || edgeCount < 0 {
		return 0, fmt.Errorf("%w: invalid payload inputs", ErrInvalidVamanaFile)
	}
	perNode := uint64(12) + uint64(dimension)*4
	if uint64(edgeCount) > math.MaxUint64/4 || (perNode != 0 && uint64(count) > (math.MaxUint64-uint64(edgeCount)*4)/perNode) {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidVamanaFile)
	}
	total := uint64(count)*perNode + uint64(edgeCount)*4
	if total > uint64(maxPlatformInt()-vamanaHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidVamanaFile)
	}
	return int(total), nil
}
