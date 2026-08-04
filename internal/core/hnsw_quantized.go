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

	"github.com/gorse-io/zvec/internal/ailego"
)

// ScalarQuantizedHNSWIndex owns a stable HNSW topology, original vectors for
// refinement, and scalar codes used for graph traversal and candidate scores.
// It is immutable; stream additions belong to the unquantized source index and
// require constructing a new quantized snapshot.
type ScalarQuantizedHNSWIndex struct {
	base    *HNSWIndex
	vectors *scalarQuantizedVectors
}

// NewScalarQuantizedHNSWIndex snapshots base and quantizes every vector after
// applying the optional dimension-preserving reformer.
func NewScalarQuantizedHNSWIndex(
	ctx context.Context,
	base *HNSWIndex,
	kind Quantization,
	reformer DenseReformer,
) (*ScalarQuantizedHNSWIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized HNSW context")
	}
	if base == nil {
		return nil, errors.New("core: nil HNSW source index")
	}
	base.mu.RLock()
	snapshot, err := cloneHNSWIndex(ctx, base)
	base.mu.RUnlock()
	if err != nil {
		return nil, err
	}
	vectors, err := newScalarQuantizedVectors(
		ctx, snapshot.dimension, snapshot.options.Metric, kind, reformer, snapshot.keys, snapshot.vectors,
	)
	if err != nil {
		return nil, err
	}
	return &ScalarQuantizedHNSWIndex{base: snapshot, vectors: vectors}, nil
}

func (i *ScalarQuantizedHNSWIndex) Dimension() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.dimension
}

func (i *ScalarQuantizedHNSWIndex) Metric() Metric {
	if i == nil || i.vectors == nil {
		return 0
	}
	return i.vectors.metric
}

func (i *ScalarQuantizedHNSWIndex) Len() int {
	if i == nil || i.vectors == nil {
		return 0
	}
	return len(i.vectors.keys)
}

func (i *ScalarQuantizedHNSWIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil || i.vectors == nil {
		return nil, false
	}
	return i.vectors.vector(key)
}

func (i *ScalarQuantizedHNSWIndex) BuildOptions() HNSWBuildOptions {
	if i == nil || i.base == nil {
		return HNSWBuildOptions{}
	}
	return i.base.options
}

func (i *ScalarQuantizedHNSWIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.search(ctx, query, HNSWSearchOptions{
		SearchOptions:  SearchOptions{TopK: k},
		EF:             DefaultHNSWEFSearch,
		PrefetchOffset: DefaultHNSWPrefetchOffset,
	}, false)
}

func (i *ScalarQuantizedHNSWIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.search(ctx, query, HNSWSearchOptions{
		SearchOptions:  options,
		EF:             DefaultHNSWEFSearch,
		PrefetchOffset: DefaultHNSWPrefetchOffset,
	}, true)
}

// SearchHNSW executes scalar-code graph traversal with explicit EF and
// prefetch controls.
func (i *ScalarQuantizedHNSWIndex) SearchHNSW(ctx context.Context, query []float32, options HNSWSearchOptions) ([]Result, error) {
	return i.search(ctx, query, options, true)
}

func (i *ScalarQuantizedHNSWIndex) search(
	ctx context.Context,
	query []float32,
	options HNSWSearchOptions,
	requirePositiveTopK bool,
) ([]Result, error) {
	if i == nil || i.base == nil || i.vectors == nil {
		return nil, errors.New("core: nil scalar-quantized HNSW index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil scalar-quantized HNSW search context")
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
			return nil, errors.New("core: negative scalar-quantized HNSW top-k")
		}
		if options.Radius < 0 {
			return nil, ErrInvalidRadius
		}
	}
	queryCode, err := i.vectors.quantizedQuery(query)
	if err != nil {
		return nil, err
	}
	if options.TopK == 0 || len(i.vectors.keys) == 0 {
		return []Result{}, nil
	}
	if len(i.vectors.keys) <= DefaultHNSWBruteForceThreshold {
		positions := make([]int, len(i.vectors.keys))
		for position := range positions {
			positions[position] = position
		}
		return i.vectors.searchWithCode(ctx, queryCode, options.SearchOptions, positions)
	}

	scoreAt := func(position int) (float32, error) {
		return QuantizedDistance(i.vectors.metric, i.vectors.codes[position], queryCode)
	}
	entry := i.base.entryPoint
	for level := i.base.maxLevel; level > 0; level-- {
		nearest, err := i.searchLayer(ctx, []int{entry}, 1, level, scoreAt)
		if err != nil {
			return nil, fmt.Errorf("core: descend scalar-quantized HNSW level %d: %w", level, err)
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	capacity := max(options.EF, options.TopK)
	candidates, err := i.searchBase(ctx, entry, capacity, options, scoreAt)
	if err != nil {
		return nil, err
	}
	if len(candidates) > options.TopK {
		candidates = candidates[:options.TopK]
	}
	results := make([]Result, len(candidates))
	for position, candidate := range candidates {
		results[position] = Result{Key: i.vectors.keys[candidate.position], Score: candidate.score}
	}
	return results, nil
}

func (i *ScalarQuantizedHNSWIndex) searchLayer(
	ctx context.Context,
	entries []int,
	ef, level int,
	scoreAt func(int) (float32, error),
) ([]hnswScoredNode, error) {
	limit := min(ef, len(i.vectors.keys))
	if limit <= 0 {
		return []hnswScoredNode{}, nil
	}
	metric := i.vectors.metric
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(metric, left, right) }
	worse := func(left, right hnswScoredNode) bool { return hnswNodeBetter(metric, right, left) }
	candidates := ailego.NewHeap(better)
	results := ailego.NewHeap(worse)
	visited := make([]bool, len(i.vectors.keys))
	for _, entry := range entries {
		if entry < 0 || entry >= len(i.vectors.keys) || i.base.levels[entry] < level || visited[entry] {
			continue
		}
		score, err := scoreAt(entry)
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
		if results.Len() >= limit && hasWorst && hnswNodeBetter(metric, worst, current) {
			break
		}
		for _, neighbor := range i.base.neighbors[current.position][level] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := scoreAt(neighbor)
			if err != nil {
				return nil, err
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = results.Peek()
			if results.Len() < limit || !hasWorst || hnswNodeBetter(metric, node, worst) {
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
		if hnswNodeBetter(metric, left, right) {
			return -1
		}
		if hnswNodeBetter(metric, right, left) {
			return 1
		}
		return 0
	})
	return result, nil
}

func (i *ScalarQuantizedHNSWIndex) searchBase(
	ctx context.Context,
	entry, capacity int,
	options HNSWSearchOptions,
	scoreAt func(int) (float32, error),
) ([]hnswScoredNode, error) {
	metric := i.vectors.metric
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(metric, left, right) }
	worse := func(left, right hnswScoredNode) bool { return i.resultNodeBetter(right, left) }
	frontier := ailego.NewHeap(better)
	accepted := ailego.NewHeap(worse)
	visited := make([]bool, len(i.vectors.keys))

	score, err := scoreAt(entry)
	if err != nil {
		return nil, fmt.Errorf("core: score scalar-quantized HNSW entry point: %w", err)
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
		if accepted.Len() >= capacity && hasWorst && metric.Better(worst.score, current.score) {
			break
		}
		neighbors := i.base.neighbors[current.position][0]
		prefetchQuantizedHNSWNeighbors(i.vectors.codes, neighbors, options.PrefetchOffset, options.PrefetchLines)
		for _, neighbor := range neighbors {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := scoreAt(neighbor)
			if err != nil {
				return nil, fmt.Errorf("core: score scalar-quantized HNSW node %d: %w", neighbor, err)
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = accepted.Peek()
			if accepted.Len() < capacity || !hasWorst || !metric.Better(worst.score, node.score) {
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

func (i *ScalarQuantizedHNSWIndex) resultNodeBetter(left, right hnswScoredNode) bool {
	if left.score == right.score {
		return i.vectors.keys[left.position] < i.vectors.keys[right.position]
	}
	return i.vectors.metric.Better(left.score, right.score)
}

func (i *ScalarQuantizedHNSWIndex) acceptResult(node hnswScoredNode, options SearchOptions) bool {
	key := i.vectors.keys[node.position]
	return (options.Filter == nil || options.Filter(key)) &&
		scoreWithinRadius(i.vectors.metric, node.score, options.Radius)
}

var (
	_ DenseProvider      = (*ScalarQuantizedHNSWIndex)(nil)
	_ DenseSearcher      = (*ScalarQuantizedHNSWIndex)(nil)
	_ DenseQuerySearcher = (*ScalarQuantizedHNSWIndex)(nil)
)
