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
	DefaultVamanaMaxDegree        = 64
	DefaultVamanaSearchListSize   = 100
	DefaultVamanaMaxOcclusionSize = 750
	DefaultVamanaAlpha            = float32(1.2)
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
	if !o.Metric.valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidVamanaOptions)
	}
	if o.MaxDegree <= 0 || o.MaxDegree > MaxVamanaDegree {
		return fmt.Errorf("%w: MaxDegree must be in [1,%d]", ErrInvalidVamanaOptions, MaxVamanaDegree)
	}
	if o.SearchListSize < o.MaxDegree {
		return fmt.Errorf("%w: SearchListSize must be at least MaxDegree", ErrInvalidVamanaOptions)
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
	index := &VamanaIndex{
		dimension: b.dimension, options: b.options,
		keys: b.keys, vectors: b.vectors, positions: b.positions,
		neighbors: make([][]int, len(b.keys)), neighborDistances: make([][]float32, len(b.keys)),
		entryPoint: -1,
	}
	for position := range index.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := index.insertNode(ctx, position); err != nil {
			return nil, fmt.Errorf("core: construct Vamana node %d: %w", position, err)
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

// VamanaIndex stores original FP32 vectors and one bounded directed graph.
// Readers share an immutable generation while Add publishes copy-on-write.
type VamanaIndex struct {
	streamMu          sync.Mutex
	mu                sync.RWMutex
	dimension         int
	options           VamanaBuildOptions
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

func (i *VamanaIndex) searchBuildCandidates(ctx context.Context, query []float32, entry, capacity int) ([]vamanaDistanceNode, error) {
	limit := min(capacity, len(i.keys))
	if limit <= 0 {
		return []vamanaDistanceNode{}, nil
	}
	better := func(left, right vamanaDistanceNode) bool { return vamanaDistanceBetter(left, right) }
	worse := func(left, right vamanaDistanceNode) bool { return vamanaDistanceBetter(right, left) }
	frontier := ailego.NewHeap(better)
	retained := ailego.NewHeap(worse)
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
	deduplicated := make([]vamanaDistanceNode, 0, len(candidates))
	seen := make(map[int]struct{}, len(candidates))
	for _, candidate := range candidates {
		if candidate.position == owner {
			continue
		}
		if _, found := seen[candidate.position]; found {
			continue
		}
		seen[candidate.position] = struct{}{}
		deduplicated = append(deduplicated, candidate)
		if len(deduplicated) == i.options.MaxOcclusionSize {
			break
		}
	}
	candidates = deduplicated
	factors := make([]float32, len(candidates))
	consumed := make([]bool, len(candidates))
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
		selectedSet := make(map[int]struct{}, len(selected))
		for _, candidate := range selected {
			selectedSet[candidate.position] = struct{}{}
		}
		for _, candidate := range candidates {
			if len(selected) == i.options.MaxDegree {
				break
			}
			if _, found := selectedSet[candidate.position]; !found {
				selected = append(selected, candidate)
				selectedSet[candidate.position] = struct{}{}
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
	i.neighbors[owner] = make([]int, len(selected))
	i.neighborDistances[owner] = make([]float32, len(selected))
	for offset, candidate := range selected {
		i.neighbors[owner][offset] = candidate.position
		i.neighborDistances[owner][offset] = candidate.distance
	}
}

func (i *VamanaIndex) graphDistance(left, right []float32) (float32, error) {
	score, err := i.options.Metric.Compute(left, right)
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
		keys: slices.Clone(source.keys), vectors: slices.Clone(source.vectors),
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
