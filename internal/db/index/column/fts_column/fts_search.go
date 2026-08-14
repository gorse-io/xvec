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

package ftscolumn

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

// ErrInvalidFTSSearch identifies invalid top-k or non-finite scoring state.
var ErrInvalidFTSSearch = errors.New("core: invalid FTS search")

// FTSResult is one segment-local BM25 result.
type FTSResult struct {
	DocumentID uint32
	Score      float32
}

// FTSSearchOptions configures exact BM25 top-k retrieval.
type FTSSearchOptions struct {
	TopK int
	FTSQueryExecutionOptions
}

// SearchFTS executes an exact term, phrase, and boolean query and returns at
// most TopK results by descending BM25 score, breaking ties by ascending
// document ID. Once the heap is full, safe posting-block score bounds skip
// ranges that cannot beat its minimum. Scorer may hold deletion-aware
// statistics across multiple segments so independently searched segments
// remain score-comparable.
func SearchFTS(ctx context.Context, dictionary *FTSTermDictionary, node FTSQueryNode, scorer *BM25Scorer, options FTSSearchOptions) ([]FTSResult, error) {
	results, _, err := searchFTSWithStats(ctx, dictionary, node, scorer, options)
	return results, err
}

type ftsSearchStats struct {
	scoredDocuments uint64
	blockMaxSkips   uint64
	wandSkips       uint64
}

func searchFTSWithStats(ctx context.Context, dictionary *FTSTermDictionary, node FTSQueryNode, scorer *BM25Scorer, options FTSSearchOptions) ([]FTSResult, ftsSearchStats, error) {
	var stats ftsSearchStats
	if ctx == nil {
		return nil, stats, fmt.Errorf("%w: nil context", ErrInvalidFTSSearch)
	}
	if err := ctx.Err(); err != nil {
		return nil, stats, err
	}
	if options.TopK < 0 {
		return nil, stats, fmt.Errorf("%w: TopK must be non-negative", ErrInvalidFTSSearch)
	}
	if dictionary == nil {
		return nil, stats, fmt.Errorf("%w: dictionary is nil", ErrInvalidFTSQueryExecution)
	}
	if scorer == nil {
		return nil, stats, fmt.Errorf("%w: BM25 scorer is nil", ErrInvalidFTSQueryExecution)
	}
	if options.TopK == 0 {
		return []FTSResult{}, stats, nil
	}
	iterator, err := NewFTSScoredQueryIterator(ctx, dictionary, node, scorer, options.FTSQueryExecutionOptions)
	if err != nil {
		return nil, stats, err
	}
	results := make(ftsResultHeap, 0, min(options.TopK, 64))
	heap.Init(&results)
	target := uint32(0)
	for {
		exhausted := false
		if len(results) == options.TopK {
			for {
				block := ftsIteratorBlockMaxInfo(iterator.root, target)
				if block.lastDoc < target || !(block.score < results[0].Score) {
					break
				}
				stats.blockMaxSkips++
				if block.lastDoc == math.MaxUint32 {
					exhausted = true
					break
				}
				target = block.lastDoc + 1
			}
		}
		if exhausted {
			break
		}
		advanced := false
		if len(results) == options.TopK {
			var wandSkips uint64
			advanced, wandSkips = iterator.advanceCompetitive(ctx, target, results[0].Score)
			stats.wandSkips += wandSkips
		} else {
			advanced = iterator.Advance(ctx, target)
		}
		if !advanced {
			break
		}
		stats.scoredDocuments++
		score := iterator.Score()
		if float64Score := float64(score); math.IsNaN(float64Score) || math.IsInf(float64Score, 0) {
			return nil, stats, fmt.Errorf("%w: document %d has non-finite score", ErrInvalidFTSSearch, iterator.DocumentID())
		}
		documentID := iterator.DocumentID()
		if score > 0 {
			candidate := FTSResult{DocumentID: documentID, Score: score}
			if len(results) < options.TopK {
				heap.Push(&results, candidate)
			} else if ftsResultBetter(candidate, results[0]) {
				results[0] = candidate
				heap.Fix(&results, 0)
			}
		}
		if documentID == math.MaxUint32 {
			break
		}
		target = documentID + 1
	}
	if err := iterator.Err(); err != nil {
		return nil, stats, err
	}
	sort.Slice(results, func(left, right int) bool { return ftsResultBetter(results[left], results[right]) })
	return []FTSResult(results), stats, nil
}

func ftsResultBetter(left, right FTSResult) bool {
	if left.Score != right.Score {
		return left.Score > right.Score
	}
	return left.DocumentID < right.DocumentID
}

type ftsResultHeap []FTSResult

func (h ftsResultHeap) Len() int { return len(h) }
func (h ftsResultHeap) Less(left, right int) bool {
	if h[left].Score != h[right].Score {
		return h[left].Score < h[right].Score
	}
	return h[left].DocumentID > h[right].DocumentID
}
func (h ftsResultHeap) Swap(left, right int) { h[left], h[right] = h[right], h[left] }
func (h *ftsResultHeap) Push(value any)      { *h = append(*h, value.(FTSResult)) }
func (h *ftsResultHeap) Pop() any {
	old := *h
	last := len(old) - 1
	value := old[last]
	*h = old[:last]
	return value
}
