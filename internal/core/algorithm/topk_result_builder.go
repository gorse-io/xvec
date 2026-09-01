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
	"errors"
	"fmt"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
	"github.com/gorse-io/xvec/internal/core/metric"
)

// Metric selects score computation and ordering for exact search.
type Metric = metric.Metric

const (
	MetricL2     = metric.L2
	MetricIP     = metric.IP
	MetricCosine = metric.Cosine
	MetricMIPSL2 = metric.MIPSL2
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
	if !metric.Valid() {
		return nil, errors.New("core: invalid metric")
	}
	if err := mathutil.ValidateDense(query, len(query)); err != nil {
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
	distance, err := metric.PrevalidatedDistance()
	if err != nil {
		return nil, err
	}
	k := min(options.TopK, count)
	worstFirst := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return metric.Better(right.Score, left.Score)
	}
	heap := container.NewHeapWithCapacity(k, worstFirst)
	for index := 0; index < count; index++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		candidate := candidateAt(index)
		if options.Filter != nil && !options.Filter(candidate.Key) {
			continue
		}
		if err := mathutil.ValidateDense(candidate.Vector, len(query)); err != nil {
			return nil, fmt.Errorf("core: validate candidate %d (key %d): %w", index, candidate.Key, err)
		}
		score := distance(candidate.Vector, query)
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

func topKPrevalidatedCandidatesWithOptions(
	ctx context.Context,
	metric Metric,
	distance mathutil.DenseDistance,
	query []float32,
	options SearchOptions,
	count int,
	candidateAt func(int) Candidate,
) ([]Result, error) {
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
	heap := container.NewHeapWithCapacity(k, worstFirst)
	for index := 0; index < count; index++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		candidate := candidateAt(index)
		if options.Filter != nil && !options.Filter(candidate.Key) {
			continue
		}
		score := distance(candidate.Vector, query)
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

// topKPrevalidatedCandidateBatchesWithOptions scans already partitioned
// candidates without flattening them into a temporary slice. IVF uses one
// batch per probed inverted list.
func topKPrevalidatedCandidateBatchesWithOptions(
	ctx context.Context,
	metric Metric,
	distance mathutil.DenseDistance,
	query []float32,
	options SearchOptions,
	batches int,
	batchLen func(int) int,
	candidateAt func(int, int) Candidate,
) ([]Result, error) {
	count := 0
	for batch := range batches {
		count += batchLen(batch)
	}
	if options.TopK == 0 || count == 0 {
		return []Result{}, nil
	}
	k := min(options.TopK, count)
	worstFirst := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return metric.Better(right.Score, left.Score)
	}
	heap := container.NewHeapWithCapacity(k, worstFirst)
	ordinal := 0
	for batch := range batches {
		for index := range batchLen(batch) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			candidate := candidateAt(batch, index)
			if options.Filter != nil && !options.Filter(candidate.Key) {
				ordinal++
				continue
			}
			score := distance(candidate.Vector, query)
			ordinal++
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
	err := parallel.ParallelFor(ctx, len(queries), workers, func(ctx context.Context, index int) error {
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
