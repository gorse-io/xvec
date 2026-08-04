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

const (
	DefaultHNSWEFSearch            = 300
	MaxHNSWEFSearch                = 2048
	DefaultHNSWBruteForceThreshold = 1000
)

var ErrInvalidHNSWEF = errors.New("core: HNSW EF must be in [1, 2048]")

// HNSWSearchOptions combines common result controls with the level-zero
// exploration width.
type HNSWSearchOptions struct {
	SearchOptions
	EF int
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
		SearchOptions: SearchOptions{TopK: k},
		EF:            DefaultHNSWEFSearch,
	}, false)
}

// SearchWithOptions applies common filter and radius controls with default EF.
func (i *HNSWIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.searchHNSW(ctx, query, HNSWSearchOptions{
		SearchOptions: options,
		EF:            DefaultHNSWEFSearch,
	}, true)
}

// SearchHNSW executes a metric-aware hierarchical graph query with explicit
// EF. EF smaller than TopK is raised to TopK so the requested result count can
// be retained.
func (i *HNSWIndex) SearchHNSW(ctx context.Context, query []float32, options HNSWSearchOptions) ([]Result, error) {
	return i.searchHNSW(ctx, query, options, true)
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
		return topKCandidatesWithOptions(ctx, i.options.Metric, query, options.SearchOptions, len(i.keys), func(position int) Candidate {
			return Candidate{Key: i.keys[position], Vector: i.vectorAt(position)}
		}, requirePositiveTopK)
	}

	entry := i.entryPoint
	for level := i.maxLevel; level > 0; level-- {
		nearest, err := i.searchHNSWLayer(ctx, query, []int{entry}, 1, level)
		if err != nil {
			return nil, fmt.Errorf("core: descend HNSW level %d: %w", level, err)
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	capacity := max(options.EF, options.TopK)
	candidates, err := i.searchHNSWBase(ctx, query, entry, capacity, options.SearchOptions)
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

func (i *HNSWIndex) searchHNSWBase(ctx context.Context, query []float32, entry, capacity int, options SearchOptions) ([]hnswScoredNode, error) {
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(i.options.Metric, left, right) }
	worse := func(left, right hnswScoredNode) bool { return i.hnswResultNodeBetter(right, left) }
	frontier := ailego.NewHeap(better)
	accepted := ailego.NewHeap(worse)
	visited := make([]bool, len(i.keys))

	score, err := i.options.Metric.Compute(query, i.vectorAt(entry))
	if err != nil {
		return nil, fmt.Errorf("core: score HNSW entry point: %w", err)
	}
	start := hnswScoredNode{position: entry, score: score}
	visited[entry] = true
	frontier.Push(start)
	if i.acceptHNSWResult(start, options) {
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
		for _, neighbor := range i.neighbors[current.position][0] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := i.options.Metric.Compute(query, i.vectorAt(neighbor))
			if err != nil {
				return nil, fmt.Errorf("core: score HNSW node %d: %w", neighbor, err)
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = accepted.Peek()
			if accepted.Len() < capacity || !hasWorst || !i.options.Metric.Better(worst.score, node.score) {
				frontier.Push(node)
				if i.acceptHNSWResult(node, options) {
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
