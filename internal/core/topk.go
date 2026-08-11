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

	"github.com/gorse-io/xvec/internal/ailego"
)

// Metric selects score computation and ordering for exact search.
type Metric uint8

const (
	MetricL2 Metric = iota + 1
	MetricIP
	MetricCosine
	MetricMIPSL2
)

// Candidate is one immutable dense vector considered by exact search.
type Candidate struct {
	Key    uint64
	Vector []float32
}

// Result is a candidate key and its public score. Results are ordered best
// first, then by ascending key for equal scores.
type Result struct {
	Key   uint64
	Score float32
}

// Compute calculates the score for left and right.
func (m Metric) Compute(left, right []float32) (float32, error) {
	switch m {
	case MetricL2:
		return ailego.L2Squared(left, right)
	case MetricIP:
		return ailego.InnerProduct(left, right)
	case MetricCosine:
		return ailego.CosineDistance(left, right)
	case MetricMIPSL2:
		return ailego.MIPSL2Squared(left, right)
	default:
		return 0, errors.New("core: invalid metric")
	}
}

// Better reports whether left should rank before right.
func (m Metric) Better(left, right float32) bool {
	if m == MetricIP {
		return left > right
	}
	return left < right
}

// TopK computes exact scores and returns at most k results. It uses O(k)
// memory and checks ctx between candidates.
func TopK(
	ctx context.Context,
	metric Metric,
	query []float32,
	candidates []Candidate,
	k int,
) ([]Result, error) {
	return topKCandidates(ctx, metric, query, k, len(candidates), func(index int) Candidate {
		return candidates[index]
	})
}

func topKCandidates(
	ctx context.Context,
	metric Metric,
	query []float32,
	k int,
	count int,
	candidateAt func(int) Candidate,
) ([]Result, error) {
	return topKCandidatesWithOptions(ctx, metric, query, SearchOptions{TopK: k}, count, candidateAt, false)
}

func topKCandidatesWithOptions(
	ctx context.Context,
	metric Metric,
	query []float32,
	options SearchOptions,
	count int,
	candidateAt func(int) Candidate,
	requirePositiveTopK bool,
) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil top-k context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: invalid query: %w", err)
	}
	if requirePositiveTopK {
		if err := options.Validate(); err != nil {
			return nil, err
		}
	} else if options.TopK < 0 {
		return nil, errors.New("core: negative top-k")
	}
	if options.TopK == 0 || count == 0 {
		return []Result{}, nil
	}
	k := options.TopK
	if k > count {
		k = count
	}

	worstFirst := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return metric.Better(right.Score, left.Score)
	}
	heap := ailego.NewHeap(worstFirst)
	for index := 0; index < count; index++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		candidate := candidateAt(index)
		if options.Filter != nil && !options.Filter(candidate.Key) {
			continue
		}
		score, err := metric.Compute(candidate.Vector, query)
		if err != nil {
			return nil, fmt.Errorf("core: score candidate %d (key %d): %w", index, candidate.Key, err)
		}
		if !scoreWithinRadius(metric, score, options.Radius) {
			continue
		}
		result := Result{Key: candidate.Key, Score: score}
		if heap.Len() < k {
			heap.Push(result)
			continue
		}
		worst, _ := heap.Peek()
		if resultBetter(metric, result, worst) {
			heap.Replace(result)
		}
	}

	results := heap.Values()
	slices.SortFunc(results, func(left, right Result) int {
		if resultBetter(metric, left, right) {
			return -1
		}
		if resultBetter(metric, right, left) {
			return 1
		}
		return 0
	})
	return results, nil
}

// BatchTopK computes independent top-k results for queries, preserving query
// order while using at most workers goroutines. A non-positive workers value
// uses GOMAXPROCS.
func BatchTopK(
	ctx context.Context,
	metric Metric,
	queries [][]float32,
	candidates []Candidate,
	k int,
	workers int,
) ([][]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil batch top-k context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	results := make([][]Result, len(queries))
	err := ailego.ParallelFor(ctx, len(queries), workers, func(ctx context.Context, index int) error {
		queryResults, err := TopK(ctx, metric, queries[index], candidates, k)
		if err != nil {
			return fmt.Errorf("core: query %d: %w", index, err)
		}
		results[index] = queryResults
		return nil
	})
	if err != nil {
		return nil, err
	}
	return results, nil
}

func resultBetter(metric Metric, left, right Result) bool {
	if left.Score == right.Score {
		return left.Key < right.Key
	}
	return metric.Better(left.Score, right.Score)
}
