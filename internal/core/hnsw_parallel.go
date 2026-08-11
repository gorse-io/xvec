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
	"slices"
	"sync"

	"github.com/gorse-io/zvec/internal/ailego"
)

type parallelHNSWGraph struct {
	options   HNSWBuildOptions
	levels    []int
	neighbors [][][]int
	nodeLocks []sync.RWMutex
	score     func(left, right int) (float32, error)

	entryMu  sync.RWMutex
	entry    int
	maxLevel int
}

func buildParallelHNSW(
	ctx context.Context,
	workers int,
	options HNSWBuildOptions,
	levels []int,
	neighbors [][][]int,
	score func(left, right int) (float32, error),
) (entryPoint, maxLevel int, err error) {
	if len(levels) == 0 {
		return -1, -1, nil
	}
	graph := parallelHNSWGraph{
		options: options, levels: levels, neighbors: neighbors,
		nodeLocks: make([]sync.RWMutex, len(levels)), score: score,
		entry: 0, maxLevel: levels[0],
	}
	if err := ailego.ParallelFor(ctx, len(levels)-1, workers, func(workerCtx context.Context, offset int) error {
		position := offset + 1
		if insertErr := graph.insert(workerCtx, position); insertErr != nil {
			return fmt.Errorf("construct HNSW node %d: %w", position, insertErr)
		}
		return nil
	}); err != nil {
		return -1, -1, err
	}
	graph.entryMu.RLock()
	defer graph.entryMu.RUnlock()
	return graph.entry, graph.maxLevel, nil
}

func (g *parallelHNSWGraph) insert(ctx context.Context, position int) error {
	level := g.levels[position]
	entry, maxLevel := g.entrySnapshot()
	for currentLevel := maxLevel; currentLevel > level; currentLevel-- {
		nearest, err := g.searchLayer(ctx, position, []int{entry}, 1, currentLevel)
		if err != nil {
			return err
		}
		if len(nearest) != 0 {
			entry = nearest[0].position
		}
	}
	for currentLevel := min(level, maxLevel); currentLevel >= 0; currentLevel-- {
		candidates, err := g.searchLayer(ctx, position, []int{entry}, g.options.EFConstruction, currentLevel)
		if err != nil {
			return err
		}
		selected, err := g.selectNeighbors(ctx, position, candidates, g.maxDegree(currentLevel))
		if err != nil {
			return err
		}
		if err := g.mergeNeighbors(ctx, position, selected, currentLevel); err != nil {
			return err
		}
		for _, neighbor := range selected {
			if err := g.mergeNeighbors(ctx, neighbor, []int{position}, currentLevel); err != nil {
				return err
			}
		}
		if len(candidates) != 0 {
			entry = candidates[0].position
		}
	}
	g.entryMu.Lock()
	if level > g.maxLevel {
		g.entry = position
		g.maxLevel = level
	}
	g.entryMu.Unlock()
	return nil
}

func (g *parallelHNSWGraph) entrySnapshot() (int, int) {
	g.entryMu.RLock()
	defer g.entryMu.RUnlock()
	return g.entry, g.maxLevel
}

func (g *parallelHNSWGraph) searchLayer(
	ctx context.Context,
	query int,
	entries []int,
	ef int,
	level int,
) ([]hnswScoredNode, error) {
	limit := min(ef, len(g.levels))
	if limit <= 0 {
		return []hnswScoredNode{}, nil
	}
	better := func(left, right hnswScoredNode) bool { return hnswNodeBetter(g.options.Metric, left, right) }
	worse := func(left, right hnswScoredNode) bool { return hnswNodeBetter(g.options.Metric, right, left) }
	candidates := ailego.NewHeap(better)
	results := ailego.NewHeap(worse)
	visited := make([]bool, len(g.levels))
	for _, entry := range entries {
		if entry < 0 || entry >= len(g.levels) || g.levels[entry] < level || visited[entry] {
			continue
		}
		score, err := g.score(query, entry)
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
		if results.Len() >= limit && hasWorst && hnswNodeBetter(g.options.Metric, worst, current) {
			break
		}
		g.nodeLocks[current.position].RLock()
		for _, neighbor := range g.neighbors[current.position][level] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			score, err := g.score(query, neighbor)
			if err != nil {
				g.nodeLocks[current.position].RUnlock()
				return nil, err
			}
			node := hnswScoredNode{position: neighbor, score: score}
			worst, hasWorst = results.Peek()
			if results.Len() < limit || !hasWorst || hnswNodeBetter(g.options.Metric, node, worst) {
				candidates.Push(node)
				results.Push(node)
				if results.Len() > limit {
					_, _ = results.Pop()
				}
			}
		}
		g.nodeLocks[current.position].RUnlock()
	}
	result := results.Values()
	slices.SortFunc(result, func(left, right hnswScoredNode) int {
		if hnswNodeBetter(g.options.Metric, left, right) {
			return -1
		}
		if hnswNodeBetter(g.options.Metric, right, left) {
			return 1
		}
		return 0
	})
	return result, nil
}

func (g *parallelHNSWGraph) selectNeighbors(
	ctx context.Context,
	owner int,
	candidates []hnswScoredNode,
	limit int,
) ([]int, error) {
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
			between, err := g.score(candidate.position, accepted)
			if err != nil {
				return nil, err
			}
			if !g.options.Metric.Better(candidate.score, between) {
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

func (g *parallelHNSWGraph) mergeNeighbors(ctx context.Context, owner int, additions []int, level int) error {
	g.nodeLocks[owner].Lock()
	defer g.nodeLocks[owner].Unlock()
	current := g.neighbors[owner][level]
	merged := slices.Clone(current)
	for _, addition := range additions {
		if addition == owner || slices.Contains(merged, addition) {
			continue
		}
		merged = append(merged, addition)
	}
	limit := g.maxDegree(level)
	if len(merged) <= limit {
		g.neighbors[owner][level] = merged
		return nil
	}
	candidates := make([]hnswScoredNode, 0, len(merged))
	for _, position := range merged {
		score, err := g.score(owner, position)
		if err != nil {
			return err
		}
		candidates = append(candidates, hnswScoredNode{position: position, score: score})
	}
	slices.SortFunc(candidates, func(left, right hnswScoredNode) int {
		if hnswNodeBetter(g.options.Metric, left, right) {
			return -1
		}
		if hnswNodeBetter(g.options.Metric, right, left) {
			return 1
		}
		return 0
	})
	selected, err := g.selectNeighbors(ctx, owner, candidates, limit)
	if err != nil {
		return err
	}
	g.neighbors[owner][level] = selected
	return nil
}

func (g *parallelHNSWGraph) maxDegree(level int) int {
	if level == 0 {
		return g.options.M * 2
	}
	return g.options.M
}
