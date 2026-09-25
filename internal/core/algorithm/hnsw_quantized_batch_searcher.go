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
	"context"
	"fmt"
	"slices"
)

// searchBaseQuantized batches immutable INT8 or INT4 codes and maintains
// candidates with the same BlockHeap used by dense HNSW. Filters and radius
// searches retain the dual-heap traversal so rejected bridge nodes can still
// be expanded.
func (i *ScalarQuantizedHNSWIndex) searchBaseQuantized(
	ctx context.Context, query QuantizedVector, entry, capacity int,
	options HNSWSearchOptions, visited *hnswVisited,
) ([]hnswScoredNode, error) {
	visited.reset(len(i.vectors.keys))
	degree := min(i.base.maxDegree(0), max(0, len(i.vectors.keys)-1), initialDistanceBatchCapacity)
	visited.batchIDs = slices.Grow(visited.batchIDs[:0], degree)
	visited.batchTies = slices.Grow(visited.batchTies[:0], degree)
	visited.batchCodes = slices.Grow(visited.batchCodes[:0], degree)
	visited.batchCodeDots = slices.Grow(visited.batchCodeDots[:0], degree)
	visited.batchScores = slices.Grow(visited.batchScores[:0], degree)
	metric := i.vectors.metric

	score, err := i.vectors.distance(i.vectors.codes[entry], query)
	if err != nil {
		return nil, err
	}
	visited.blockHeap.Reset(capacity, degree)
	visited.blockHeap.pushBlockWithTies(
		[]float32{blockHeapDistance(metric, score)}, []uint32{uint32(entry)}, []uint64{i.vectors.keys[entry]},
	)
	visited.overflow = visited.overflow[:0]
	overflowCursor := 0
	visited.mark(entry)

	for visited.blockHeap.HasNext() || overflowCursor < len(visited.overflow) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var current uint32
		if visited.blockHeap.HasNext() {
			current, _ = visited.blockHeap.Pop()
		} else {
			current = visited.overflow[overflowCursor]
			overflowCursor++
		}
		if visited.expanded(int(current)) {
			continue
		}
		visited.markExpanded(int(current))
		neighbors := i.base.neighbors[int(current)][0]
		prefetchQuantizedHNSWNeighbors(i.vectors.codes, neighbors, options.PrefetchOffset, options.PrefetchLines)
		visited.batchIDs = visited.batchIDs[:0]
		visited.batchTies = visited.batchTies[:0]
		visited.batchCodes = visited.batchCodes[:0]
		visited.batchCodeDots = visited.batchCodeDots[:0]
		visited.batchScores = visited.batchScores[:0]
		for _, neighbor := range neighbors {
			if visited.seen(neighbor) {
				continue
			}
			visited.mark(neighbor)
			visited.batchIDs = append(visited.batchIDs, uint32(neighbor))
			visited.batchTies = append(visited.batchTies, i.vectors.keys[neighbor])
			visited.batchCodes = append(visited.batchCodes, i.vectors.codes[neighbor].codes)
			visited.batchCodeDots = append(visited.batchCodeDots, 0)
			visited.batchScores = append(visited.batchScores, 0)
		}
		integerCodeDots(query.kind, query.codes, visited.batchCodes, visited.batchCodeDots)
		for j, id := range visited.batchIDs {
			// Stored codes are immutable and validated at construction; the
			// query was validated and quantized before graph traversal.
			score, err := i.vectors.distanceFromDot(i.vectors.codes[id], query, float64(visited.batchCodeDots[j]))
			if err != nil {
				return nil, fmt.Errorf("core: score integer-quantized HNSW neighbor: %w", err)
			}
			visited.batchScores[j] = blockHeapDistance(metric, score)
		}
		visited.blockHeap.pushBlockWithTies(visited.batchScores, visited.batchIDs, visited.batchTies)
		visited.overflow = appendBlockHeapBoundaryTies(&visited.blockHeap, visited.batchScores, visited.batchIDs, visited.batchTies, visited.overflow)
	}

	// Integer batch products are exact, so retained scores need no reranking.
	result := make([]hnswScoredNode, visited.blockHeap.Len())
	for j := range result {
		result[j] = hnswScoredNode{
			position: int(visited.blockHeap.ID(j)),
			score:    blockHeapDistance(metric, visited.blockHeap.Distance(j)),
		}
	}
	return result, nil
}
