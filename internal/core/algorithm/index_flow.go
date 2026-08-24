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
	"math"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
)

var (
	ErrInvalidTopK   = errors.New("core: top-k must be positive")
	ErrInvalidRadius = errors.New("core: radius must be finite and non-negative")
)

// CandidateFilter returns true when a document key may participate in vector
// scoring. Implementations used with segmented queries must be concurrency-safe.
type CandidateFilter func(key uint64) bool

// SearchOptions contains collection-query controls shared by exact and ANN
// indexes. Radius zero disables range filtering. For IP, a positive radius is
// a minimum similarity; for distance metrics it is a maximum distance.
type SearchOptions struct {
	TopK   int
	Radius float32
	Filter CandidateFilter
}

// Validate checks the public query invariants.
func (o SearchOptions) Validate() error {
	if o.TopK <= 0 {
		return ErrInvalidTopK
	}
	if o.Radius < 0 || math.IsNaN(float64(o.Radius)) || math.IsInf(float64(o.Radius), 0) {
		return ErrInvalidRadius
	}
	return nil
}

// DenseQuerySearcher executes one segment-local dense query.
type DenseQuerySearcher interface {
	Metric() Metric
	SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error)
}

// SparseQuerySearcher executes one segment-local sparse query.
type SparseQuerySearcher interface {
	Metric() Metric
	SearchSparseWithOptions(ctx context.Context, query SparseVector, options SearchOptions) ([]Result, error)
}

// QueryDense runs segment-local exact/ANN searches and merges their already
// ordered results into one deterministic global top-k.
func QueryDense(
	ctx context.Context,
	metric Metric,
	searchers []DenseQuerySearcher,
	query []float32,
	options SearchOptions,
	workers int,
) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil dense query context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !metric.Valid() {
		return nil, errors.New("core: invalid metric")
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	batches := make([][]Result, len(searchers))
	err := parallel.ParallelFor(ctx, len(searchers), workers, func(ctx context.Context, index int) error {
		searcher := searchers[index]
		if searcher == nil {
			return errors.New("nil dense query searcher")
		}
		if searcher.Metric() != metric {
			return fmt.Errorf("metric %d does not match query metric %d", searcher.Metric(), metric)
		}
		results, err := searcher.SearchWithOptions(ctx, query, options)
		if err != nil {
			return err
		}
		batches[index] = results
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("core: dense segment query: %w", err)
	}
	return MergeSearchResults(metric, options.TopK, batches...), nil
}

// QuerySparse runs segment-local sparse queries and globally merges them.
func QuerySparse(
	ctx context.Context,
	searchers []SparseQuerySearcher,
	query SparseVector,
	options SearchOptions,
	workers int,
) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse query context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	batches := make([][]Result, len(searchers))
	err := parallel.ParallelFor(ctx, len(searchers), workers, func(ctx context.Context, index int) error {
		searcher := searchers[index]
		if searcher == nil {
			return errors.New("nil sparse query searcher")
		}
		if searcher.Metric() != MetricIP {
			return fmt.Errorf("sparse searcher uses unsupported metric %d", searcher.Metric())
		}
		results, err := searcher.SearchSparseWithOptions(ctx, query, options)
		if err != nil {
			return err
		}
		batches[index] = results
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("core: sparse segment query: %w", err)
	}
	return MergeSearchResults(MetricIP, options.TopK, batches...), nil
}

// MergeSearchResults selects a global top-k from segment-local results. Equal
// scores remain ordered by ascending key regardless of segment order.
func MergeSearchResults(metric Metric, k int, batches ...[]Result) []Result {
	if k <= 0 {
		return []Result{}
	}
	worstFirst := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return metric.Better(right.Score, left.Score)
	}
	heap := container.NewHeap(worstFirst)
	for _, batch := range batches {
		for _, result := range batch {
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
	if results == nil {
		return []Result{}
	}
	slices.SortFunc(results, func(left, right Result) int {
		if resultBetter(metric, left, right) {
			return -1
		}
		if resultBetter(metric, right, left) {
			return 1
		}
		return 0
	})
	return results
}

func scoreWithinRadius(metric Metric, score, radius float32) bool {
	if radius == 0 {
		return true
	}
	if metric == MetricIP {
		return score >= radius
	}
	return score <= radius
}
