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

	"github.com/gorse-io/xvec/internal/ailego"
)

var ErrInvalidHNSWRaBitQOptions = errors.New("core: invalid HNSW-RaBitQ build options")

// HNSWRaBitQBuildOptions configures original-vector graph construction and
// the RaBitQ model used for queries. Seed controls both components.
type HNSWRaBitQBuildOptions struct {
	Metric         Metric
	TotalBits      int
	Clusters       int
	SampleCount    int
	MaxIterations  int
	Workers        int
	M              int
	EFConstruction int
	Seed           uint64
}

// DefaultHNSWRaBitQBuildOptions returns the pinned public defaults.
func DefaultHNSWRaBitQBuildOptions(metric Metric) HNSWRaBitQBuildOptions {
	return HNSWRaBitQBuildOptions{
		Metric:         metric,
		TotalBits:      DefaultRaBitQTotalBits,
		Clusters:       DefaultRaBitQClusters,
		MaxIterations:  DefaultKMeansIterations,
		M:              DefaultHNSWM,
		EFConstruction: DefaultHNSWEFConstruction,
	}
}

// Validate checks graph and converter invariants that do not depend on data.
func (o HNSWRaBitQBuildOptions) Validate() error {
	rabitq := o.raBitQOptions()
	if err := rabitq.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQOptions, err)
	}
	hnsw := o.hnswOptions()
	if err := hnsw.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQOptions, err)
	}
	return nil
}

func (o HNSWRaBitQBuildOptions) raBitQOptions() RaBitQOptions {
	return RaBitQOptions{
		Metric: o.Metric, TotalBits: o.TotalBits, Clusters: o.Clusters,
		SampleCount: o.SampleCount, MaxIterations: o.MaxIterations,
		Workers: o.Workers, Seed: o.Seed ^ 0x6872777261626974,
	}
}

func (o HNSWRaBitQBuildOptions) hnswOptions() HNSWBuildOptions {
	return HNSWBuildOptions{
		Metric: o.Metric, M: o.M, EFConstruction: o.EFConstruction,
		Seed: o.Seed ^ 0x6872776772617068,
	}
}

// HNSWRaBitQBuilder owns input originals until Build publishes one index.
type HNSWRaBitQBuilder struct {
	mu        sync.Mutex
	dimension int
	options   HNSWRaBitQBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

func NewHNSWRaBitQBuilder(dimension int, options HNSWRaBitQBuildOptions) (*HNSWRaBitQBuilder, error) {
	if dimension < MinRaBitQDimension || dimension > MaxRaBitQDimension {
		return nil, fmt.Errorf("%w: dimension must be in [%d,%d]", ErrInvalidHNSWRaBitQOptions, MinRaBitQDimension, MaxRaBitQDimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &HNSWRaBitQBuilder{
		dimension: dimension, options: options, positions: make(map[uint64]int),
	}, nil
}

func (b *HNSWRaBitQBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil HNSW-RaBitQ builder")
	}
	if ctx == nil {
		return errors.New("core: nil HNSW-RaBitQ add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate HNSW-RaBitQ vector: %w", err)
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

func (b *HNSWRaBitQBuilder) Build(ctx context.Context) (*HNSWRaBitQIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}

	vectors := denseVectorViews(b.vectors, b.dimension)
	model, err := trainHNSWRaBitQModel(ctx, b.dimension, vectors, b.options.raBitQOptions())
	if err != nil {
		return nil, fmt.Errorf("core: train HNSW-RaBitQ model: %w", err)
	}
	baseBuilder := &HNSWBuilder{
		dimension: b.dimension, options: b.options.hnswOptions(),
		keys: slices.Clone(b.keys), vectors: slices.Clone(b.vectors),
		positions: cloneUint64Positions(b.positions),
	}
	base, err := baseBuilder.Build(ctx)
	if err != nil {
		return nil, fmt.Errorf("core: build HNSW-RaBitQ graph: %w", err)
	}
	codes, err := model.EncodeBatch(ctx, denseVectorViews(base.vectors, base.dimension), b.options.Workers)
	if err != nil {
		return nil, fmt.Errorf("core: encode HNSW-RaBitQ vectors: %w", err)
	}
	index := &HNSWRaBitQIndex{options: b.options, base: base, model: model, codes: codes}
	if err := validateHNSWRaBitQIndex(ctx, index); err != nil {
		return nil, err
	}
	b.built = true
	b.keys = nil
	b.vectors = nil
	b.positions = nil
	return index, nil
}

func trainHNSWRaBitQModel(ctx context.Context, dimension int, vectors [][]float32, options RaBitQOptions) (*RaBitQModel, error) {
	if len(vectors) != 0 {
		return TrainRaBitQ(ctx, vectors, options)
	}
	padded := roundUpRaBitQDimension(dimension)
	random := splitMix64{state: options.Seed ^ 0x726162697471726f}
	signs := make([]byte, 4*padded/8)
	for index := range signs {
		signs[index] = byte(random.next())
	}
	extraScale := float64(0)
	if options.TotalBits > 1 {
		var err error
		extraScale, err = trainRaBitQExtraScale(ctx, padded, options.TotalBits-1, options.Workers, options.Seed^0x7261626974717363)
		if err != nil {
			return nil, err
		}
	}
	return RestoreRaBitQModel(RaBitQModelState{
		Dimension: dimension, Metric: options.Metric, TotalBits: options.TotalBits,
		Centroids: [][]float32{make([]float32, dimension)}, RotationSigns: signs,
		ExtraScale: extraScale,
	})
}

func denseVectorViews(flat []float32, dimension int) [][]float32 {
	if dimension <= 0 || len(flat) == 0 {
		return [][]float32{}
	}
	result := make([][]float32, len(flat)/dimension)
	for index := range result {
		start := index * dimension
		result[index] = flat[start : start+dimension]
	}
	return result
}

func cloneUint64Positions(source map[uint64]int) map[uint64]int {
	result := make(map[uint64]int, len(source))
	for key, position := range source {
		result[key] = position
	}
	return result
}

// HNSWRaBitQIndex binds an original-vector HNSW graph, an immutable RaBitQ
// model, and one code per graph position. Adds publish complete generations.
type HNSWRaBitQIndex struct {
	streamMu sync.Mutex
	mu       sync.RWMutex
	options  HNSWRaBitQBuildOptions
	base     *HNSWIndex
	model    *RaBitQModel
	codes    []RaBitQCode
}

func (i *HNSWRaBitQIndex) Dimension() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.base == nil {
		return 0
	}
	return i.base.dimension
}

func (i *HNSWRaBitQIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.options.Metric
}

func (i *HNSWRaBitQIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.base == nil {
		return 0
	}
	return len(i.base.keys)
}

func (i *HNSWRaBitQIndex) BuildOptions() HNSWRaBitQBuildOptions {
	if i == nil {
		return HNSWRaBitQBuildOptions{}
	}
	return i.options
}

func (i *HNSWRaBitQIndex) ModelState() RaBitQModelState {
	if i == nil {
		return RaBitQModelState{}
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.model == nil {
		return RaBitQModelState{}
	}
	return i.model.State()
}

func (i *HNSWRaBitQIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil {
		return nil, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.base == nil {
		return nil, false
	}
	position, found := i.base.positions[key]
	if !found {
		return nil, false
	}
	return slices.Clone(i.base.vectorAt(position)), true
}

func (i *HNSWRaBitQIndex) MaxLevel() int {
	if i == nil {
		return -1
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.base == nil {
		return -1
	}
	return i.base.maxLevel
}

func (i *HNSWRaBitQIndex) EntryPoint() (uint64, bool) {
	if i == nil {
		return 0, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.base == nil || i.base.entryPoint < 0 {
		return 0, false
	}
	return i.base.keys[i.base.entryPoint], true
}

func (i *HNSWRaBitQIndex) Level(key uint64) (int, bool) {
	if i == nil {
		return 0, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.base == nil {
		return 0, false
	}
	position, found := i.base.positions[key]
	if !found {
		return 0, false
	}
	return i.base.levels[position], true
}

func (i *HNSWRaBitQIndex) Neighbors(key uint64, level int) ([]uint64, error) {
	if i == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.base == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ graph")
	}
	position, found := i.base.positions[key]
	if !found {
		return nil, fmt.Errorf("%w: %d", ErrHNSWKeyNotFound, key)
	}
	if level < 0 || level > i.base.levels[position] {
		return nil, fmt.Errorf("%w: key %d has maximum level %d, got %d", ErrInvalidHNSWLevel, key, i.base.levels[position], level)
	}
	result := make([]uint64, len(i.base.neighbors[position][level]))
	for index, neighbor := range i.base.neighbors[position][level] {
		result[index] = i.base.keys[neighbor]
	}
	return result, nil
}

// HNSWRaBitQSearchOptions configures graph exploration and optional exact
// reranking. Refine reranks up to EF approximate candidates from originals.
type HNSWRaBitQSearchOptions struct {
	SearchOptions
	EF     int
	Refine bool
	Linear bool
}

func (o HNSWRaBitQSearchOptions) Validate() error {
	if err := o.SearchOptions.Validate(); err != nil {
		return err
	}
	if o.EF <= 0 || o.EF > MaxHNSWEFSearch {
		return ErrInvalidHNSWEF
	}
	return nil
}

func (i *HNSWRaBitQIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.search(ctx, query, HNSWRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: k}, EF: DefaultHNSWEFSearch,
	}, false)
}

func (i *HNSWRaBitQIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.search(ctx, query, HNSWRaBitQSearchOptions{
		SearchOptions: options, EF: DefaultHNSWEFSearch,
	}, true)
}

func (i *HNSWRaBitQIndex) SearchHNSWRaBitQ(ctx context.Context, query []float32, options HNSWRaBitQSearchOptions) ([]Result, error) {
	return i.search(ctx, query, options, true)
}

// SearchHNSWRaBitQGroups performs native HNSW traversal with RaBitQ estimates
// and expands level zero when the initial candidates lack enough groups.
func (i *HNSWRaBitQIndex) SearchHNSWRaBitQGroups(
	ctx context.Context,
	vector []float32,
	options HNSWGroupSearchOptions,
) ([]GroupResult, error) {
	if i == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}

	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateHNSWRaBitQGeneration(i); err != nil {
		return nil, err
	}
	query, err := i.model.PrepareQuery(vector)
	if err != nil {
		return nil, fmt.Errorf("core: prepare HNSW-RaBitQ group-by query: %w", err)
	}
	if len(i.codes) == 0 {
		return []GroupResult{}, nil
	}
	candidateCount, err := hnswGroupCandidateCount(options.GroupByOptions)
	if err != nil {
		return nil, err
	}
	entry := i.base.entryPoint
	for level := i.base.maxLevel; level > 0; level-- {
		entry, err = i.searchRaBitQLayer(ctx, query, entry, level)
		if err != nil {
			return nil, fmt.Errorf("core: descend HNSW-RaBitQ group-by level %d: %w", level, err)
		}
	}
	searchOptions := SearchOptions{
		TopK: candidateCount, Radius: options.Radius, Filter: options.Filter,
	}
	initial, err := i.searchRaBitQBase(ctx, query, entry, max(options.EF, candidateCount), searchOptions)
	if err != nil {
		return nil, err
	}
	if len(initial) > candidateCount {
		initial = initial[:candidateCount]
	}
	scoreAt := func(position int) (float32, error) {
		estimate, err := query.Estimate(i.codes[position])
		if err != nil {
			return 0, err
		}
		return estimate.Distance, nil
	}
	better := func(left, right hnswScoredNode) bool {
		if left.score == right.score {
			return i.base.keys[left.position] < i.base.keys[right.position]
		}
		return left.score < right.score
	}
	return expandHNSWGroups(
		ctx, i.options.Metric, i.base.keys, i.base.neighbors, initial, options.GroupByOptions,
		scoreAt, i.publicRaBitQScore, better, nil,
	)
}

// SearchGroups performs an exact scan over the immutable RaBitQ codes and
// groups the resulting public approximation scores.
func (i *HNSWRaBitQIndex) SearchGroups(
	ctx context.Context,
	vector []float32,
	options GroupByOptions,
) ([]GroupResult, error) {
	if i == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateHNSWRaBitQGeneration(i); err != nil {
		return nil, err
	}
	query, err := i.model.PrepareQuery(vector)
	if err != nil {
		return nil, fmt.Errorf("core: prepare HNSW-RaBitQ group-by query: %w", err)
	}
	accumulator := newGroupAccumulator(i.options.Metric, options.TopKPerGroup)
	for position, code := range i.codes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := i.base.keys[position]
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		estimate, err := query.Estimate(code)
		if err != nil {
			return nil, fmt.Errorf("core: score HNSW-RaBitQ group candidate %d: %w", position, err)
		}
		score := i.publicRaBitQScore(estimate.Distance)
		if !scoreWithinRadius(i.options.Metric, score, options.Radius) {
			continue
		}
		value, found := options.Resolve(key)
		if !found {
			continue
		}
		accumulator.add(value, Result{Key: key, Score: score})
	}
	return accumulator.finish(options.GroupCount), nil
}

func (i *HNSWRaBitQIndex) search(ctx context.Context, vector []float32, options HNSWRaBitQSearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ search context")
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
			return nil, errors.New("core: negative HNSW-RaBitQ top-k")
		}
		if options.Radius < 0 || math.IsNaN(float64(options.Radius)) || math.IsInf(float64(options.Radius), 0) {
			return nil, ErrInvalidRadius
		}
	}

	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateHNSWRaBitQGeneration(i); err != nil {
		return nil, err
	}
	query, err := i.model.PrepareQuery(vector)
	if err != nil {
		return nil, fmt.Errorf("core: prepare HNSW-RaBitQ query: %w", err)
	}
	if options.TopK == 0 || len(i.codes) == 0 {
		return []Result{}, nil
	}
	if !options.Refine {
		return i.searchPrepared(ctx, query, options.SearchOptions, options.EF, options.Linear)
	}
	candidateCount := min(len(i.codes), max(options.TopK, options.EF))
	if options.Linear {
		candidateCount = len(i.codes)
	}
	candidates, err := i.searchPrepared(ctx, query, SearchOptions{TopK: candidateCount, Filter: options.Filter}, options.EF, options.Linear)
	if err != nil {
		return nil, err
	}
	return i.refinePrepared(ctx, vector, candidates, options.SearchOptions)
}

func (i *HNSWRaBitQIndex) searchPrepared(ctx context.Context, query *RaBitQQuery, options SearchOptions, ef int, linear bool) ([]Result, error) {
	if linear || len(i.codes) <= DefaultHNSWBruteForceThreshold {
		return i.scanRaBitQCodes(ctx, query, options)
	}
	entry := i.base.entryPoint
	for level := i.base.maxLevel; level > 0; level-- {
		nearest, err := i.searchRaBitQLayer(ctx, query, entry, level)
		if err != nil {
			return nil, fmt.Errorf("core: descend HNSW-RaBitQ level %d: %w", level, err)
		}
		entry = nearest
	}
	nodes, err := i.searchRaBitQBase(ctx, query, entry, max(ef, options.TopK), options)
	if err != nil {
		return nil, err
	}
	if len(nodes) > options.TopK {
		nodes = nodes[:options.TopK]
	}
	return i.raBitQResults(nodes), nil
}

func (i *HNSWRaBitQIndex) scanRaBitQCodes(ctx context.Context, query *RaBitQQuery, options SearchOptions) ([]Result, error) {
	worse := func(left, right hnswScoredNode) bool { return i.raBitQNodeBetter(right, left) }
	selected := ailego.NewHeap(worse)
	for position, code := range i.codes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		estimate, err := query.Estimate(code)
		if err != nil {
			return nil, err
		}
		node := hnswScoredNode{position: position, score: estimate.Distance}
		if !i.acceptRaBitQNode(node, options) {
			continue
		}
		if selected.Len() < options.TopK {
			selected.Push(node)
			continue
		}
		worst, _ := selected.Peek()
		if i.raBitQNodeBetter(node, worst) {
			selected.Replace(node)
		}
	}
	return i.sortedRaBitQResults(selected.Values()), nil
}

func (i *HNSWRaBitQIndex) searchRaBitQLayer(ctx context.Context, query *RaBitQQuery, entry, level int) (int, error) {
	current := entry
	estimate, err := query.EstimateCoarse(i.codes[current])
	if err != nil {
		return 0, err
	}
	currentDistance := estimate.Distance
	for {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		changed := false
		for _, neighbor := range i.base.neighbors[current][level] {
			candidate, err := query.EstimateCoarse(i.codes[neighbor])
			if err != nil {
				return 0, err
			}
			if candidate.Distance < currentDistance || (candidate.Distance == currentDistance && neighbor < current) {
				current, currentDistance, changed = neighbor, candidate.Distance, true
			}
		}
		if !changed {
			return current, nil
		}
	}
}

func (i *HNSWRaBitQIndex) searchRaBitQBase(ctx context.Context, query *RaBitQQuery, entry, capacity int, options SearchOptions) ([]hnswScoredNode, error) {
	better := func(left, right hnswScoredNode) bool { return i.raBitQNodeBetter(left, right) }
	worse := func(left, right hnswScoredNode) bool { return i.raBitQNodeBetter(right, left) }
	frontier := ailego.NewHeap(better)
	accepted := ailego.NewHeap(worse)
	visited := make([]bool, len(i.codes))
	estimate, err := query.Estimate(i.codes[entry])
	if err != nil {
		return nil, err
	}
	start := hnswScoredNode{position: entry, score: estimate.Distance}
	visited[entry] = true
	frontier.Push(start)
	if i.acceptRaBitQNode(start, options) {
		accepted.Push(start)
	}
	for frontier.Len() != 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current, _ := frontier.Pop()
		worst, hasWorst := accepted.Peek()
		if accepted.Len() >= capacity && hasWorst && i.raBitQNodeBetter(worst, current) {
			break
		}
		for _, neighbor := range i.base.neighbors[current.position][0] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			coarse, err := query.EstimateCoarse(i.codes[neighbor])
			if err != nil {
				return nil, err
			}
			worst, hasWorst = accepted.Peek()
			if accepted.Len() >= capacity && hasWorst && coarse.LowerBound >= worst.score {
				continue
			}
			full, err := query.Estimate(i.codes[neighbor])
			if err != nil {
				return nil, err
			}
			node := hnswScoredNode{position: neighbor, score: full.Distance}
			worst, hasWorst = accepted.Peek()
			if accepted.Len() < capacity || !hasWorst || i.raBitQNodeBetter(node, worst) {
				frontier.Push(node)
				if i.acceptRaBitQNode(node, options) {
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
		if i.raBitQNodeBetter(left, right) {
			return -1
		}
		if i.raBitQNodeBetter(right, left) {
			return 1
		}
		return 0
	})
	return result, nil
}

func (i *HNSWRaBitQIndex) raBitQNodeBetter(left, right hnswScoredNode) bool {
	if left.score == right.score {
		return i.base.keys[left.position] < i.base.keys[right.position]
	}
	return left.score < right.score
}

func (i *HNSWRaBitQIndex) publicRaBitQScore(distance float32) float32 {
	if i.options.Metric == MetricIP {
		return 1 - distance
	}
	return distance
}

func (i *HNSWRaBitQIndex) acceptRaBitQNode(node hnswScoredNode, options SearchOptions) bool {
	key := i.base.keys[node.position]
	return (options.Filter == nil || options.Filter(key)) &&
		scoreWithinRadius(i.options.Metric, i.publicRaBitQScore(node.score), options.Radius)
}

func (i *HNSWRaBitQIndex) sortedRaBitQResults(nodes []hnswScoredNode) []Result {
	slices.SortFunc(nodes, func(left, right hnswScoredNode) int {
		if i.raBitQNodeBetter(left, right) {
			return -1
		}
		if i.raBitQNodeBetter(right, left) {
			return 1
		}
		return 0
	})
	return i.raBitQResults(nodes)
}

func (i *HNSWRaBitQIndex) raBitQResults(nodes []hnswScoredNode) []Result {
	results := make([]Result, len(nodes))
	for index, node := range nodes {
		results[index] = Result{Key: i.base.keys[node.position], Score: i.publicRaBitQScore(node.score)}
	}
	return results
}

func (i *HNSWRaBitQIndex) refinePrepared(ctx context.Context, query []float32, candidates []Result, options SearchOptions) ([]Result, error) {
	positions := make([]int, 0, len(candidates))
	seen := make(map[uint64]struct{}, len(candidates))
	for _, candidate := range candidates {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, duplicate := seen[candidate.Key]; duplicate {
			continue
		}
		seen[candidate.Key] = struct{}{}
		position, found := i.base.positions[candidate.Key]
		if !found {
			return nil, fmt.Errorf("%w: key %d", ErrMissingRefineVector, candidate.Key)
		}
		positions = append(positions, position)
	}
	return topKCandidatesWithOptions(ctx, i.options.Metric, query, options, len(positions), func(index int) Candidate {
		position := positions[index]
		return Candidate{Key: i.base.keys[position], Vector: i.base.vectorAt(position)}
	}, true)
}

// Add encodes one vector with the fixed model and atomically publishes a graph
// generation containing both the new topology and code.
func (i *HNSWRaBitQIndex) Add(ctx context.Context, key uint64, vector []float32) error {
	if i == nil {
		return errors.New("core: nil HNSW-RaBitQ index")
	}
	if ctx == nil {
		return errors.New("core: nil HNSW-RaBitQ add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	i.streamMu.Lock()
	defer i.streamMu.Unlock()

	i.mu.RLock()
	if err := validateHNSWRaBitQGeneration(i); err != nil {
		i.mu.RUnlock()
		return err
	}
	if _, exists := i.base.positions[key]; exists {
		i.mu.RUnlock()
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	working, err := cloneHNSWIndex(ctx, i.base)
	model := i.model
	codes := cloneRaBitQCodes(i.codes)
	i.mu.RUnlock()
	if err != nil {
		return err
	}
	code, err := model.Encode(vector)
	if err != nil {
		return fmt.Errorf("core: encode incremental HNSW-RaBitQ vector: %w", err)
	}
	if err := working.Add(ctx, key, vector); err != nil {
		return err
	}
	codes = append(codes, code)
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	if err := ctx.Err(); err != nil {
		i.mu.Unlock()
		return err
	}
	i.base = working
	i.codes = codes
	i.mu.Unlock()
	return nil
}

func cloneRaBitQCodes(source []RaBitQCode) []RaBitQCode {
	result := make([]RaBitQCode, len(source))
	for index, code := range source {
		result[index] = code
		result[index].binaryCode = slices.Clone(code.binaryCode)
		result[index].extraCode = slices.Clone(code.extraCode)
	}
	return result
}

func validateHNSWRaBitQGeneration(index *HNSWRaBitQIndex) error {
	if index == nil || index.base == nil || index.model == nil {
		return errors.New("core: invalid HNSW-RaBitQ generation")
	}
	if index.base.dimension != index.model.dimension || index.options.Metric != index.model.metric ||
		index.options.Metric != index.base.options.Metric || len(index.codes) != len(index.base.keys) {
		return errors.New("core: inconsistent HNSW-RaBitQ generation")
	}
	for _, code := range index.codes {
		if code.modelFingerprint != index.model.fingerprint || code.totalBits != index.model.totalBits ||
			code.paddedDimension != index.model.paddedDimension {
			return errors.New("core: inconsistent HNSW-RaBitQ code")
		}
	}
	return nil
}

func validateHNSWRaBitQIndex(ctx context.Context, index *HNSWRaBitQIndex) error {
	if ctx == nil {
		return errors.New("core: nil HNSW-RaBitQ validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateHNSWRaBitQGeneration(index); err != nil {
		return err
	}
	if err := index.options.Validate(); err != nil {
		return err
	}
	if index.options.hnswOptions() != index.base.options {
		return errors.New("core: HNSW-RaBitQ graph options mismatch")
	}
	state := index.model.State()
	if state.TotalBits != index.options.TotalBits || len(state.Centroids) == 0 || len(state.Centroids) > index.options.Clusters {
		return errors.New("core: HNSW-RaBitQ model options mismatch")
	}
	if err := validateHNSWIndex(ctx, index.base); err != nil {
		return err
	}
	for position, code := range index.codes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if err := code.validate(); err != nil || code.cluster >= index.model.Len() {
			return errors.New("core: invalid HNSW-RaBitQ code storage")
		}
	}
	return nil
}

var (
	_ DenseIndex         = (*HNSWRaBitQIndex)(nil)
	_ DenseProvider      = (*HNSWRaBitQIndex)(nil)
	_ DenseQuerySearcher = (*HNSWRaBitQIndex)(nil)
	_ DenseGroupSearcher = (*HNSWRaBitQIndex)(nil)
)

const (
	hnswRaBitQFileVersion = 1
	hnswRaBitQHeaderSize  = 128
	hnswRaBitQFactorBytes = 5 * 8
)

var (
	hnswRaBitQFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'H', 'R', 'B', 'Q'}

	ErrInvalidHNSWRaBitQFile        = errors.New("core: invalid HNSW-RaBitQ file")
	ErrHNSWRaBitQChecksumMismatch   = errors.New("core: HNSW-RaBitQ checksum mismatch")
	ErrUnsupportedHNSWRaBitQVersion = errors.New("core: unsupported HNSW-RaBitQ file version")
)

// Save durably and atomically publishes one complete graph/model/code
// generation in the native Go HNSW-RaBitQ format.
func (i *HNSWRaBitQIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil HNSW-RaBitQ save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidHNSWRaBitQFile)
	}
	snapshot, err := i.hnswRaBitQPersistenceSnapshot(ctx)
	if err != nil {
		return err
	}
	encoded, err := encodeHNSWRaBitQIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ailego.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save HNSW-RaBitQ file: %w", err)
	}
	return nil
}

func (i *HNSWRaBitQIndex) hnswRaBitQPersistenceSnapshot(ctx context.Context) (*HNSWRaBitQIndex, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidHNSWRaBitQFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateHNSWRaBitQGeneration(i); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	base, err := cloneHNSWIndex(ctx, i.base)
	if err != nil {
		return nil, err
	}
	model, err := RestoreRaBitQModel(i.model.State())
	if err != nil {
		return nil, fmt.Errorf("%w: snapshot model: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	return &HNSWRaBitQIndex{
		options: i.options, base: base, model: model, codes: cloneRaBitQCodes(i.codes),
	}, nil
}

// OpenHNSWRaBitQIndex reads and verifies a native Go HNSW-RaBitQ artifact.
func OpenHNSWRaBitQIndex(ctx context.Context, path string) (*HNSWRaBitQIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidHNSWRaBitQFile)
	}
	encoded, err := readHNSWFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read HNSW-RaBitQ file: %w", err)
	}
	index, err := decodeHNSWRaBitQIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open HNSW-RaBitQ file: %w", err)
	}
	return index, nil
}

func encodeHNSWRaBitQIndex(ctx context.Context, index *HNSWRaBitQIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateHNSWRaBitQIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	if err := validateHNSWRaBitQFormatOptions(index.options); err != nil {
		return nil, err
	}
	base, err := encodeHNSWIndex(ctx, index.base)
	if err != nil {
		return nil, fmt.Errorf("%w: encode graph: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	state := index.model.State()
	payloadSize, recordSize, err := checkedHNSWRaBitQPayloadSize(len(base), state, len(index.codes))
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	payload = append(payload, base...)
	for centroidIndex, centroid := range state.Centroids {
		if centroidIndex&63 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for _, value := range centroid {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
	}
	payload = append(payload, state.RotationSigns...)
	for position, code := range index.codes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		start := len(payload)
		payload = binary.LittleEndian.AppendUint32(payload, uint32(code.cluster))
		for _, factor := range [...]float64{code.coarseAdd, code.coarseRescale, code.coarseError, code.fullAdd, code.fullRescale} {
			payload = binary.LittleEndian.AppendUint64(payload, math.Float64bits(factor))
		}
		payload = append(payload, code.binaryCode...)
		payload = append(payload, code.extraCode...)
		if len(payload)-start != recordSize {
			return nil, fmt.Errorf("%w: internal code record length", ErrInvalidHNSWRaBitQFile)
		}
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidHNSWRaBitQFile)
	}

	header := make([]byte, hnswRaBitQHeaderSize)
	copy(header[:8], hnswRaBitQFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], hnswRaBitQFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], hnswRaBitQHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(hnswRaBitQHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(len(base)))
	binary.LittleEndian.PutUint64(header[40:48], uint64(len(index.codes)))
	binary.LittleEndian.PutUint32(header[48:52], uint32(index.base.dimension))
	header[52] = byte(index.options.Metric)
	header[53] = byte(index.options.TotalBits)
	binary.LittleEndian.PutUint32(header[56:60], uint32(len(state.Centroids)))
	binary.LittleEndian.PutUint32(header[60:64], uint32(len(state.RotationSigns)))
	binary.LittleEndian.PutUint64(header[64:72], math.Float64bits(state.ExtraScale))
	binary.LittleEndian.PutUint64(header[72:80], index.model.fingerprint)
	binary.LittleEndian.PutUint32(header[80:84], uint32(index.options.M))
	binary.LittleEndian.PutUint32(header[84:88], uint32(index.options.EFConstruction))
	binary.LittleEndian.PutUint32(header[88:92], uint32(index.options.Clusters))
	binary.LittleEndian.PutUint32(header[92:96], uint32(index.options.SampleCount))
	binary.LittleEndian.PutUint32(header[96:100], uint32(index.options.MaxIterations))
	binary.LittleEndian.PutUint32(header[100:104], uint32(index.options.Workers))
	binary.LittleEndian.PutUint64(header[104:112], index.options.Seed)
	binary.LittleEndian.PutUint32(header[112:116], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
	return append(header, payload...), nil
}

func decodeHNSWRaBitQIndex(ctx context.Context, encoded []byte) (*HNSWRaBitQIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil HNSW-RaBitQ decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < hnswRaBitQHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidHNSWRaBitQFile)
	}
	header := encoded[:hnswRaBitQHeaderSize]
	if !bytes.Equal(header[:8], hnswRaBitQFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidHNSWRaBitQFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != hnswRaBitQFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedHNSWRaBitQVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != hnswRaBitQHeaderSize ||
		binary.LittleEndian.Uint32(header[12:16]) != 0 || header[54] != 0 || header[55] != 0 ||
		!hnswAllZero(header[116:124]) {
		return nil, fmt.Errorf("%w: invalid header fields", ErrInvalidHNSWRaBitQFile)
	}
	if got, want := ailego.CRC32C(header[:124]), binary.LittleEndian.Uint32(header[124:128]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrHNSWRaBitQChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-hnswRaBitQHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidHNSWRaBitQFile)
	}
	payload := encoded[hnswRaBitQHeaderSize:]
	if got, want := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[112:116]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrHNSWRaBitQChecksumMismatch, got, want)
	}

	baseLength64 := binary.LittleEndian.Uint64(header[32:40])
	count64 := binary.LittleEndian.Uint64(header[40:48])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[48:52]))
	centroidCount64 := uint64(binary.LittleEndian.Uint32(header[56:60]))
	rotationBytes64 := uint64(binary.LittleEndian.Uint32(header[60:64]))
	for _, value := range []uint64{baseLength64, count64, dimension64, centroidCount64, rotationBytes64} {
		if value > uint64(maxPlatformInt()) {
			return nil, fmt.Errorf("%w: field exceeds platform capacity", ErrInvalidHNSWRaBitQFile)
		}
	}
	baseLength, count, dimension := int(baseLength64), int(count64), int(dimension64)
	centroidCount, rotationBytes := int(centroidCount64), int(rotationBytes64)
	if baseLength < hnswHeaderSize || baseLength > len(payload) || uint64(count) > math.MaxUint32 ||
		dimension < MinRaBitQDimension || dimension > MaxRaBitQDimension || centroidCount <= 0 {
		return nil, fmt.Errorf("%w: invalid size fields", ErrInvalidHNSWRaBitQFile)
	}
	options, err := decodeHNSWRaBitQOptions(header)
	if err != nil {
		return nil, err
	}
	state := RaBitQModelState{
		Dimension: dimension, Metric: options.Metric, TotalBits: options.TotalBits,
		Centroids: make([][]float32, centroidCount), ExtraScale: math.Float64frombits(binary.LittleEndian.Uint64(header[64:72])),
	}
	centroidBytes64 := uint64(centroidCount) * uint64(dimension) * 4
	if centroidBytes64 > uint64(maxPlatformInt()) || uint64(baseLength)+centroidBytes64+uint64(rotationBytes) > uint64(len(payload)) {
		return nil, fmt.Errorf("%w: model storage exceeds payload", ErrInvalidHNSWRaBitQFile)
	}
	offset := baseLength
	for centroidIndex := range state.Centroids {
		if centroidIndex&63 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		centroid := make([]float32, dimension)
		for component := range centroid {
			centroid[component] = math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
		}
		state.Centroids[centroidIndex] = centroid
	}
	state.RotationSigns = slicesCloneBytes(payload[offset : offset+rotationBytes])
	offset += rotationBytes
	model, err := RestoreRaBitQModel(state)
	if err != nil {
		return nil, fmt.Errorf("%w: restore model: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	if model.fingerprint != binary.LittleEndian.Uint64(header[72:80]) {
		return nil, fmt.Errorf("%w: model fingerprint mismatch", ErrInvalidHNSWRaBitQFile)
	}
	base, err := decodeHNSWIndex(ctx, payload[:baseLength])
	if err != nil {
		return nil, fmt.Errorf("%w: decode graph: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	if len(base.keys) != count || base.dimension != dimension {
		return nil, fmt.Errorf("%w: graph metadata mismatch", ErrInvalidHNSWRaBitQFile)
	}
	recordSize := hnswRaBitQCodeRecordSize(model.paddedDimension, model.totalBits)
	if recordSize <= 0 || count > (len(payload)-offset)/recordSize || offset+count*recordSize != len(payload) {
		return nil, fmt.Errorf("%w: invalid code payload length", ErrInvalidHNSWRaBitQFile)
	}
	codes := make([]RaBitQCode, count)
	binaryBytes := model.paddedDimension / 8
	extraBytes := model.paddedDimension * model.extraBits / 8
	for position := range codes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		cluster := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if cluster >= uint64(model.Len()) {
			return nil, fmt.Errorf("%w: code cluster out of range", ErrInvalidHNSWRaBitQFile)
		}
		factors := [5]float64{}
		for factorIndex := range factors {
			factors[factorIndex] = math.Float64frombits(binary.LittleEndian.Uint64(payload[offset : offset+8]))
			offset += 8
		}
		code := RaBitQCode{
			modelFingerprint: model.fingerprint, cluster: int(cluster),
			paddedDimension: model.paddedDimension, totalBits: model.totalBits,
			coarseAdd: factors[0], coarseRescale: factors[1], coarseError: factors[2],
			fullAdd: factors[3], fullRescale: factors[4],
			binaryCode: slicesCloneBytes(payload[offset : offset+binaryBytes]),
		}
		offset += binaryBytes
		code.extraCode = slicesCloneBytes(payload[offset : offset+extraBytes])
		offset += extraBytes
		if err := code.validate(); err != nil {
			return nil, fmt.Errorf("%w: invalid code factors", ErrInvalidHNSWRaBitQFile)
		}
		codes[position] = code
	}
	index := &HNSWRaBitQIndex{options: options, base: base, model: model, codes: codes}
	if err := validateHNSWRaBitQIndex(ctx, index); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	return index, nil
}

func decodeHNSWRaBitQOptions(header []byte) (HNSWRaBitQBuildOptions, error) {
	values := [6]uint64{
		uint64(binary.LittleEndian.Uint32(header[80:84])),
		uint64(binary.LittleEndian.Uint32(header[84:88])),
		uint64(binary.LittleEndian.Uint32(header[88:92])),
		uint64(binary.LittleEndian.Uint32(header[92:96])),
		uint64(binary.LittleEndian.Uint32(header[96:100])),
		uint64(binary.LittleEndian.Uint32(header[100:104])),
	}
	for _, value := range values {
		if value > uint64(maxPlatformInt()) {
			return HNSWRaBitQBuildOptions{}, fmt.Errorf("%w: options exceed platform capacity", ErrInvalidHNSWRaBitQFile)
		}
	}
	options := HNSWRaBitQBuildOptions{
		Metric: Metric(header[52]), TotalBits: int(header[53]), M: int(values[0]), EFConstruction: int(values[1]),
		Clusters: int(values[2]), SampleCount: int(values[3]), MaxIterations: int(values[4]), Workers: int(values[5]),
		Seed: binary.LittleEndian.Uint64(header[104:112]),
	}
	if err := options.Validate(); err != nil {
		return HNSWRaBitQBuildOptions{}, fmt.Errorf("%w: %v", ErrInvalidHNSWRaBitQFile, err)
	}
	return options, nil
}

func validateHNSWRaBitQFormatOptions(options HNSWRaBitQBuildOptions) error {
	for _, value := range []int{options.M, options.EFConstruction, options.Clusters, options.SampleCount, options.MaxIterations, options.Workers} {
		if value < 0 || uint64(value) > math.MaxUint32 {
			return fmt.Errorf("%w: options exceed format capacity", ErrInvalidHNSWRaBitQFile)
		}
	}
	return nil
}

func checkedHNSWRaBitQPayloadSize(baseLength int, state RaBitQModelState, count int) (int, int, error) {
	if baseLength < 0 || state.Dimension <= 0 || len(state.Centroids) == 0 || count < 0 {
		return 0, 0, fmt.Errorf("%w: invalid payload inputs", ErrInvalidHNSWRaBitQFile)
	}
	recordSize := hnswRaBitQCodeRecordSize(roundUpRaBitQDimension(state.Dimension), state.TotalBits)
	centroidBytes := uint64(len(state.Centroids)) * uint64(state.Dimension) * 4
	total := uint64(baseLength) + centroidBytes + uint64(len(state.RotationSigns)) + uint64(count)*uint64(recordSize)
	if recordSize <= 0 || total > uint64(maxPlatformInt()-hnswRaBitQHeaderSize) {
		return 0, 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidHNSWRaBitQFile)
	}
	return int(total), recordSize, nil
}

func hnswRaBitQCodeRecordSize(paddedDimension, totalBits int) int {
	if paddedDimension < MinRaBitQDimension || paddedDimension%64 != 0 || totalBits < 1 || totalBits > 9 {
		return 0
	}
	return 4 + hnswRaBitQFactorBytes + paddedDimension/8 + paddedDimension*(totalBits-1)/8
}

func slicesCloneBytes(source []byte) []byte {
	return append([]byte(nil), source...)
}
