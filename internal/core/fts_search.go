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

// FTSSearchOptions configures exhaustive exact BM25 retrieval.
type FTSSearchOptions struct {
	TopK int
	FTSQueryExecutionOptions
}

// SearchFTS executes an exact term, phrase, and boolean query and returns at
// most TopK results by descending BM25 score, breaking ties by ascending
// document ID. Scorer may hold deletion-aware statistics across multiple
// segments so independently searched segments remain score-comparable.
func SearchFTS(ctx context.Context, dictionary *FTSTermDictionary, node FTSQueryNode, scorer *BM25Scorer, options FTSSearchOptions) ([]FTSResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSSearch)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.TopK < 0 {
		return nil, fmt.Errorf("%w: TopK must be non-negative", ErrInvalidFTSSearch)
	}
	if dictionary == nil {
		return nil, fmt.Errorf("%w: dictionary is nil", ErrInvalidFTSQueryExecution)
	}
	if scorer == nil {
		return nil, fmt.Errorf("%w: BM25 scorer is nil", ErrInvalidFTSQueryExecution)
	}
	if options.TopK == 0 {
		return []FTSResult{}, nil
	}
	iterator, err := NewFTSScoredQueryIterator(ctx, dictionary, node, scorer, options.FTSQueryExecutionOptions)
	if err != nil {
		return nil, err
	}
	results := make(ftsResultHeap, 0, min(options.TopK, 64))
	heap.Init(&results)
	for iterator.Next(ctx) {
		score := iterator.Score()
		if float64Score := float64(score); math.IsNaN(float64Score) || math.IsInf(float64Score, 0) {
			return nil, fmt.Errorf("%w: document %d has non-finite score", ErrInvalidFTSSearch, iterator.DocumentID())
		}
		if score <= 0 {
			continue
		}
		candidate := FTSResult{DocumentID: iterator.DocumentID(), Score: score}
		if len(results) < options.TopK {
			heap.Push(&results, candidate)
			continue
		}
		if ftsResultBetter(candidate, results[0]) {
			results[0] = candidate
			heap.Fix(&results, 0)
		}
	}
	if err := iterator.Err(); err != nil {
		return nil, err
	}
	sort.Slice(results, func(left, right int) bool { return ftsResultBetter(results[left], results[right]) })
	return []FTSResult(results), nil
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
