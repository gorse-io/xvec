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
