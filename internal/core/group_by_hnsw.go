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
	"fmt"

	"github.com/gorse-io/zvec/internal/ailego"
)

// HNSWGroupSearchOptions combines group retention with the level-zero graph
// exploration controls shared by dense, sparse, scalar-quantized, and RaBitQ
// HNSW indexes.
type HNSWGroupSearchOptions struct {
	GroupByOptions
	EF             int
	PrefetchOffset uint32
	PrefetchLines  uint32
}

// Validate checks group retention and graph exploration invariants.
func (o HNSWGroupSearchOptions) Validate() error {
	if err := o.GroupByOptions.Validate(); err != nil {
		return err
	}
	if o.EF <= 0 || o.EF > MaxHNSWEFSearch {
		return ErrInvalidHNSWEF
	}
	_, err := hnswGroupCandidateCount(o.GroupByOptions)
	return err
}

func hnswGroupCandidateCount(options GroupByOptions) (int, error) {
	if options.GroupCount > maxPlatformInt()/options.TopKPerGroup {
		return 0, ErrGroupSizeOverflow
	}
	return options.GroupCount * options.TopKPerGroup, nil
}

// expandHNSWGroups reproduces the pinned native HNSW group-by strategy. The
// ordinary graph search first retains groupCount*topKPerGroup candidates. If
// they do not cover enough groups, a second best-first level-zero traversal
// starts from those candidates and stops as soon as the requested number of
// distinct groups is reached or the connected component is exhausted.
func expandHNSWGroups(
	ctx context.Context,
	metric Metric,
	keys []uint64,
	neighbors [][][]int,
	initial []hnswScoredNode,
	options GroupByOptions,
	scoreAt func(position int) (float32, error),
	publicScore func(score float32) float32,
	nodeBetter func(left, right hnswScoredNode) bool,
	prefetch func(neighbors []int),
) ([]GroupResult, error) {
	accumulator := newGroupAccumulator(metric, options.TopKPerGroup)
	groups := make(map[string]struct{}, min(options.GroupCount, len(initial)))
	add := func(node hnswScoredNode) {
		key := keys[node.position]
		score := publicScore(node.score)
		if options.Filter != nil && !options.Filter(key) {
			return
		}
		if !scoreWithinRadius(metric, score, options.Radius) {
			return
		}
		value, ok := options.Resolve(key)
		if !ok {
			return
		}
		accumulator.add(value, Result{Key: key, Score: score})
		groups[value] = struct{}{}
	}
	for _, node := range initial {
		add(node)
	}
	if len(groups) >= options.GroupCount || len(initial) == 0 {
		return accumulator.finish(options.GroupCount), nil
	}

	frontier := ailego.NewHeap(nodeBetter)
	visited := make([]bool, len(keys))
	for _, node := range initial {
		if node.position < 0 || node.position >= len(keys) || visited[node.position] {
			continue
		}
		visited[node.position] = true
		frontier.Push(node)
	}
	for frontier.Len() != 0 && len(groups) < options.GroupCount {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		current, _ := frontier.Pop()
		adjacent := neighbors[current.position][0]
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
				return nil, fmt.Errorf("core: score HNSW group expansion node %d: %w", neighbor, err)
			}
			node := hnswScoredNode{position: neighbor, score: score}
			add(node)
			if len(groups) >= options.GroupCount {
				break
			}
			frontier.Push(node)
		}
	}
	return accumulator.finish(options.GroupCount), nil
}

func groupNodeBetter(metric Metric, keys []uint64) func(left, right hnswScoredNode) bool {
	return func(left, right hnswScoredNode) bool {
		if left.score == right.score {
			return keys[left.position] < keys[right.position]
		}
		return metric.Better(left.score, right.score)
	}
}
