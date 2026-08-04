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

	"github.com/gorse-io/zvec/internal/ailego"
)

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
	frontier := ailego.NewHeap(better)
	accepted := ailego.NewHeap(worse)
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
