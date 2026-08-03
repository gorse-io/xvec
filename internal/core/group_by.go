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

var (
	ErrInvalidGroupCount = errors.New("core: group count must be positive")
	ErrInvalidGroupTopK  = errors.New("core: per-group top-k must be positive")
	ErrNilGroupResolver  = errors.New("core: nil group resolver")
)

// GroupResolver maps a document key to its stable string group value. An empty
// value is a valid group (and is used for NULL by the pinned native baseline);
// ok=false explicitly excludes a candidate. Resolvers used by segmented
// queries must be safe for concurrent calls.
type GroupResolver func(key uint64) (value string, ok bool)

// GroupByOptions controls exact and ANN group-by search. Groups are ranked by
// their best document and documents inside a group use the index metric.
type GroupByOptions struct {
	GroupCount   int
	TopKPerGroup int
	Radius       float32
	Filter       CandidateFilter
	Resolve      GroupResolver
}

// Validate checks group-by query invariants.
func (o GroupByOptions) Validate() error {
	if o.GroupCount <= 0 {
		return ErrInvalidGroupCount
	}
	if o.TopKPerGroup <= 0 {
		return ErrInvalidGroupTopK
	}
	if o.Radius < 0 || math.IsNaN(float64(o.Radius)) || math.IsInf(float64(o.Radius), 0) {
		return ErrInvalidRadius
	}
	if o.Resolve == nil {
		return ErrNilGroupResolver
	}
	return nil
}

// GroupResult contains one group value and its metric-ordered documents.
type GroupResult struct {
	Value   string
	Results []Result
}

// DenseGroupSearcher executes one segment-local dense group-by query.
type DenseGroupSearcher interface {
	Metric() Metric
	SearchGroups(ctx context.Context, query []float32, options GroupByOptions) ([]GroupResult, error)
}

// SparseGroupSearcher executes one segment-local sparse group-by query.
type SparseGroupSearcher interface {
	Metric() Metric
	SearchSparseGroups(ctx context.Context, query SparseVector, options GroupByOptions) ([]GroupResult, error)
}

// SearchGroups scans every eligible dense candidate. It retains only the best
// TopKPerGroup documents per group rather than first taking a global top-k.
func (i *DenseFlatIndex) SearchGroups(ctx context.Context, query []float32, options GroupByOptions) ([]GroupResult, error) {
	if i == nil {
		return nil, errors.New("core: nil dense Flat index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil dense Flat group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if _, err := i.metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: invalid dense group-by query: %w", err)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}

	accumulator := newGroupAccumulator(i.metric, options.TopKPerGroup)
	i.mu.RLock()
	defer i.mu.RUnlock()
	for position, key := range i.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		start := position * i.dimension
		score, err := i.metric.Compute(i.vectors[start:start+i.dimension], query)
		if err != nil {
			return nil, fmt.Errorf("core: score dense group candidate %d (key %d): %w", position, key, err)
		}
		if !scoreWithinRadius(i.metric, score, options.Radius) {
			continue
		}
		value, ok := options.Resolve(key)
		if !ok {
			continue
		}
		accumulator.add(value, Result{Key: key, Score: score})
	}
	return accumulator.finish(options.GroupCount), nil
}

// SearchSparseGroups scans every eligible sparse candidate using exact inner
// product and applies grouping only after the candidate filter and radius.
func (i *SparseFlatIndex) SearchSparseGroups(ctx context.Context, query SparseVector, options GroupByOptions) ([]GroupResult, error) {
	if i == nil {
		return nil, errors.New("core: nil sparse Flat index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse Flat group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if _, err := ailego.SparseInnerProduct(query.Indices, query.Values, nil, nil); err != nil {
		return nil, fmt.Errorf("core: validate sparse Flat group-by query: %w", err)
	}

	accumulator := newGroupAccumulator(MetricIP, options.TopKPerGroup)
	i.mu.RLock()
	defer i.mu.RUnlock()
	for position, key := range i.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		start, end := i.offsets[position], i.offsets[position+1]
		score, err := ailego.SparseInnerProduct(
			i.indices[start:end], i.values[start:end], query.Indices, query.Values,
		)
		if err != nil {
			return nil, fmt.Errorf("core: score sparse group candidate %d (key %d): %w", position, key, err)
		}
		if !scoreWithinRadius(MetricIP, score, options.Radius) {
			continue
		}
		value, ok := options.Resolve(key)
		if !ok {
			continue
		}
		accumulator.add(value, Result{Key: key, Score: score})
	}
	return accumulator.finish(options.GroupCount), nil
}

// QueryDenseGroups runs group-by searches concurrently across segments and
// merges groups before applying the final global GroupCount.
func QueryDenseGroups(
	ctx context.Context,
	metric Metric,
	searchers []DenseGroupSearcher,
	query []float32,
	options GroupByOptions,
	workers int,
) ([]GroupResult, error) {
	if ctx == nil {
		return nil, errors.New("core: nil dense group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !metric.valid() {
		return nil, errors.New("core: invalid group-by metric")
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	batches := make([][]GroupResult, len(searchers))
	err := ailego.ParallelFor(ctx, len(searchers), workers, func(ctx context.Context, index int) error {
		searcher := searchers[index]
		if searcher == nil {
			return errors.New("nil dense group searcher")
		}
		if searcher.Metric() != metric {
			return fmt.Errorf("metric %d does not match group-by metric %d", searcher.Metric(), metric)
		}
		results, err := searcher.SearchGroups(ctx, query, options)
		if err != nil {
			return err
		}
		batches[index] = results
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("core: dense segment group-by query: %w", err)
	}
	return MergeGroupResults(metric, options.GroupCount, options.TopKPerGroup, batches...), nil
}

// QuerySparseGroups runs sparse IP group-by searches across segments and
// merges their per-group candidate lists deterministically.
func QuerySparseGroups(
	ctx context.Context,
	searchers []SparseGroupSearcher,
	query SparseVector,
	options GroupByOptions,
	workers int,
) ([]GroupResult, error) {
	if ctx == nil {
		return nil, errors.New("core: nil sparse group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	batches := make([][]GroupResult, len(searchers))
	err := ailego.ParallelFor(ctx, len(searchers), workers, func(ctx context.Context, index int) error {
		searcher := searchers[index]
		if searcher == nil {
			return errors.New("nil sparse group searcher")
		}
		if searcher.Metric() != MetricIP {
			return fmt.Errorf("sparse group searcher uses unsupported metric %d", searcher.Metric())
		}
		results, err := searcher.SearchSparseGroups(ctx, query, options)
		if err != nil {
			return err
		}
		batches[index] = results
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("core: sparse segment group-by query: %w", err)
	}
	return MergeGroupResults(MetricIP, options.GroupCount, options.TopKPerGroup, batches...), nil
}

// MergeGroupResults combines segment-local groups. It first rebuilds each
// group's global top-k, then ranks groups by their best result. Ties between
// groups use their string values so segment order cannot affect output.
func MergeGroupResults(metric Metric, groupCount, topKPerGroup int, batches ...[]GroupResult) []GroupResult {
	if groupCount <= 0 || topKPerGroup <= 0 {
		return []GroupResult{}
	}
	accumulator := newGroupAccumulator(metric, topKPerGroup)
	for _, batch := range batches {
		for _, group := range batch {
			for _, result := range group.Results {
				accumulator.add(group.Value, result)
			}
		}
	}
	return accumulator.finish(groupCount)
}

type groupAccumulator struct {
	metric       Metric
	topKPerGroup int
	groups       map[string]*ailego.Heap[Result]
}

func newGroupAccumulator(metric Metric, topKPerGroup int) *groupAccumulator {
	return &groupAccumulator{
		metric:       metric,
		topKPerGroup: topKPerGroup,
		groups:       make(map[string]*ailego.Heap[Result]),
	}
}

func (a *groupAccumulator) add(value string, result Result) {
	heap := a.groups[value]
	if heap == nil {
		heap = ailego.NewHeap(func(left, right Result) bool {
			return resultBetter(a.metric, right, left)
		})
		a.groups[value] = heap
	}
	if heap.Len() < a.topKPerGroup {
		heap.Push(result)
		return
	}
	worst, _ := heap.Peek()
	if resultBetter(a.metric, result, worst) {
		heap.Replace(result)
	}
}

func (a *groupAccumulator) finish(groupCount int) []GroupResult {
	groups := make([]GroupResult, 0, len(a.groups))
	for value, heap := range a.groups {
		results := heap.Values()
		slices.SortFunc(results, func(left, right Result) int {
			if resultBetter(a.metric, left, right) {
				return -1
			}
			if resultBetter(a.metric, right, left) {
				return 1
			}
			return 0
		})
		if len(results) != 0 {
			groups = append(groups, GroupResult{Value: value, Results: results})
		}
	}
	slices.SortFunc(groups, func(left, right GroupResult) int {
		if resultBetter(a.metric, left.Results[0], right.Results[0]) {
			return -1
		}
		if resultBetter(a.metric, right.Results[0], left.Results[0]) {
			return 1
		}
		if left.Value < right.Value {
			return -1
		}
		if left.Value > right.Value {
			return 1
		}
		return 0
	})
	if len(groups) > groupCount {
		groups = groups[:groupCount]
	}
	if groups == nil {
		return []GroupResult{}
	}
	return groups
}

var (
	_ DenseGroupSearcher  = (*DenseFlatIndex)(nil)
	_ SparseGroupSearcher = (*SparseFlatIndex)(nil)
)
