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
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
)

var (
	ErrInvalidDimension = errors.New("core: invalid vector dimension")
	ErrDuplicateKey     = errors.New("core: duplicate vector key")
	ErrBuilderClosed    = errors.New("core: builder is closed")
)

// DenseProvider exposes vectors without prescribing their storage layout.
// Vector returns an independent copy so callers cannot mutate an index.
type DenseProvider interface {
	Dimension() int
	Len() int
	Vector(key uint64) ([]float32, bool)
}

// DenseSearcher is the common exact/ANN search contract.
type DenseSearcher interface {
	Search(ctx context.Context, query []float32, k int) ([]Result, error)
}

// DenseStreamer accepts incremental vectors after an index is built.
type DenseStreamer interface {
	Add(ctx context.Context, key uint64, vector []float32) error
}

// DenseIndex is the common runtime contract implemented by exact and ANN
// indexes that retain original dense vectors.
type DenseIndex interface {
	DenseProvider
	DenseSearcher
	DenseStreamer
}

// DenseBuilder collects vectors and transfers its index on Build.
type DenseBuilder interface {
	Add(ctx context.Context, key uint64, vector []float32) error
	Build(ctx context.Context) (DenseIndex, error)
}

// DenseFlatIndex stores FP32 vectors contiguously and scans every vector for
// exact search. Adds are serialized; any number of searches may run together.
type DenseFlatIndex struct {
	mu        sync.RWMutex
	dimension int
	metric    Metric
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
}

// NewDenseFlatIndex constructs an empty exact index.
func NewDenseFlatIndex(dimension int, metric Metric) (*DenseFlatIndex, error) {
	if dimension <= 0 {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if !metric.Valid() {
		return nil, errors.New("core: invalid metric")
	}
	return &DenseFlatIndex{
		dimension: dimension,
		metric:    metric,
		positions: make(map[uint64]int),
	}, nil
}

// Dimension returns the fixed vector dimension.
func (i *DenseFlatIndex) Dimension() int {
	if i == nil {
		return 0
	}
	return i.dimension
}

// Metric returns the index score metric.
func (i *DenseFlatIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.metric
}

// Len returns the number of indexed vectors.
func (i *DenseFlatIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// Add clones and appends one finite vector. Keys are unique for the lifetime
// of an index so deterministic tie-breaking remains unambiguous.
func (i *DenseFlatIndex) Add(ctx context.Context, key uint64, vector []float32) error {
	if i == nil {
		return errors.New("core: nil dense Flat index")
	}
	if ctx == nil {
		return errors.New("core: nil dense Flat add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(vector) != i.dimension {
		return fmt.Errorf("%w: got %d, want %d", ErrInvalidDimension, len(vector), i.dimension)
	}
	if _, err := mathutil.L2Squared(vector, vector); err != nil {
		return fmt.Errorf("core: validate dense Flat vector: %w", err)
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, exists := i.positions[key]; exists {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	i.positions[key] = len(i.keys)
	i.keys = append(i.keys, key)
	i.vectors = append(i.vectors, vector...)
	return nil
}

// Vector returns a cloned vector by key.
func (i *DenseFlatIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil {
		return nil, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return nil, false
	}
	start := position * i.dimension
	return slices.Clone(i.vectors[start : start+i.dimension]), true
}

// Search performs an exact top-k scan. It holds a read lock so the contiguous
// vector storage cannot move while candidate slices are being scored.
func (i *DenseFlatIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.search(ctx, query, SearchOptions{TopK: k}, false)
}

// SearchWithOptions applies a candidate filter and metric-aware radius before
// retaining the exact top-k.
func (i *DenseFlatIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.search(ctx, query, options, true)
}

func (i *DenseFlatIndex) search(ctx context.Context, query []float32, options SearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil dense Flat index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil dense Flat search context")
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return topKCandidatesWithOptions(ctx, i.metric, query, options, len(i.keys), func(position int) Candidate {
		start := position * i.dimension
		return Candidate{Key: i.keys[position], Vector: i.vectors[start : start+i.dimension]}
	}, requirePositiveTopK)
}

// DenseFlatIndexBuilder is a one-shot builder. The built index remains
// streamable through DenseFlatIndex.Add.
type DenseFlatIndexBuilder struct {
	mu    sync.Mutex
	index *DenseFlatIndex
	built bool
}

// NewDenseFlatBuilder constructs a builder for an exact dense index.
func NewDenseFlatBuilder(dimension int, metric Metric) (*DenseFlatIndexBuilder, error) {
	index, err := NewDenseFlatIndex(dimension, metric)
	if err != nil {
		return nil, err
	}
	return &DenseFlatIndexBuilder{index: index}, nil
}

// Add appends a vector while the builder is open.
func (b *DenseFlatIndexBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil dense Flat builder")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return ErrBuilderClosed
	}
	return b.index.Add(ctx, key, vector)
}

// Build closes the builder and returns its index. An empty index is valid.
func (b *DenseFlatIndexBuilder) Build(ctx context.Context) (DenseIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil dense Flat builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil dense Flat build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	b.built = true
	return b.index, nil
}

var (
	_ DenseProvider      = (*DenseFlatIndex)(nil)
	_ DenseSearcher      = (*DenseFlatIndex)(nil)
	_ DenseStreamer      = (*DenseFlatIndex)(nil)
	_ DenseIndex         = (*DenseFlatIndex)(nil)
	_ DenseQuerySearcher = (*DenseFlatIndex)(nil)
	_ DenseBuilder       = (*DenseFlatIndexBuilder)(nil)
)

// SparseVector is a canonical sparse FP32 vector. Indices must be strictly
// increasing and Values must be finite.
type SparseVector struct {
	Indices []uint32
	Values  []float32
}

// SparseProvider exposes cloned sparse vectors by document key.
type SparseProvider interface {
	Len() int
	SparseVector(key uint64) (SparseVector, bool)
}

// SparseSearcher is the common sparse exact/ANN search contract.
type SparseSearcher interface {
	SearchSparse(ctx context.Context, query SparseVector, k int) ([]Result, error)
}

// SparseStreamer accepts incremental canonical sparse vectors.
type SparseStreamer interface {
	AddSparse(ctx context.Context, key uint64, vector SparseVector) error
}

// SparseIndex combines sparse provider, search, and streaming capabilities.
type SparseIndex interface {
	SparseProvider
	SparseSearcher
	SparseStreamer
}

// SparseBuilder collects sparse vectors and transfers its index on Build.
type SparseBuilder interface {
	AddSparse(ctx context.Context, key uint64, vector SparseVector) error
	Build(ctx context.Context) (SparseIndex, error)
}

// SparseFlatIndex stores vectors in compressed sparse row form and performs an
// exact inner-product scan. Sparse Flat supports only IP, matching the public
// schema constraints.
type SparseFlatIndex struct {
	mu        sync.RWMutex
	keys      []uint64
	offsets   []int
	indices   []uint32
	values    []float32
	positions map[uint64]int
}

// Metric returns the only supported sparse metric.
func (i *SparseFlatIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return MetricIP
}

// NewSparseFlatIndex constructs an empty sparse IP index.
func NewSparseFlatIndex(metric Metric) (*SparseFlatIndex, error) {
	if metric != MetricIP {
		return nil, fmt.Errorf("core: sparse Flat supports only inner product, got metric %d", metric)
	}
	return &SparseFlatIndex{
		offsets: []int{0}, positions: make(map[uint64]int),
	}, nil
}

// Len returns the number of indexed sparse vectors.
func (i *SparseFlatIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// AddSparse validates, clones, and appends one canonical sparse vector.
func (i *SparseFlatIndex) AddSparse(ctx context.Context, key uint64, vector SparseVector) error {
	if i == nil {
		return errors.New("core: nil sparse Flat index")
	}
	if ctx == nil {
		return errors.New("core: nil sparse Flat add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, err := mathutil.SparseInnerProduct(vector.Indices, vector.Values, nil, nil); err != nil {
		return fmt.Errorf("core: validate sparse Flat vector: %w", err)
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, exists := i.positions[key]; exists {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	i.positions[key] = len(i.keys)
	i.keys = append(i.keys, key)
	i.indices = append(i.indices, vector.Indices...)
	i.values = append(i.values, vector.Values...)
	i.offsets = append(i.offsets, len(i.indices))
	return nil
}

// SparseVector returns an independent vector copy by key.
func (i *SparseFlatIndex) SparseVector(key uint64) (SparseVector, bool) {
	if i == nil {
		return SparseVector{}, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found {
		return SparseVector{}, false
	}
	start, end := i.offsets[position], i.offsets[position+1]
	return SparseVector{
		Indices: slices.Clone(i.indices[start:end]),
		Values:  slices.Clone(i.values[start:end]),
	}, true
}

// SearchSparse evaluates exact inner products and returns highest scores first,
// breaking equal scores by ascending key.
func (i *SparseFlatIndex) SearchSparse(ctx context.Context, query SparseVector, k int) ([]Result, error) {
	return i.searchSparse(ctx, query, SearchOptions{TopK: k}, false)
}

// SearchSparseWithOptions applies candidate and radius filtering before exact
// top-k retention.
func (i *SparseFlatIndex) SearchSparseWithOptions(ctx context.Context, query SparseVector, options SearchOptions) ([]Result, error) {
	return i.searchSparse(ctx, query, options, true)
}

func (i *SparseFlatIndex) searchSparse(ctx context.Context, query SparseVector, options SearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil sparse Flat index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse Flat search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if requirePositiveTopK {
		if err := options.Validate(); err != nil {
			return nil, err
		}
	} else if options.TopK < 0 {
		return nil, errors.New("core: negative top-k")
	}
	if _, err := mathutil.SparseInnerProduct(query.Indices, query.Values, nil, nil); err != nil {
		return nil, fmt.Errorf("core: validate sparse Flat query: %w", err)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if options.TopK == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}
	k := options.TopK
	if k > len(i.keys) {
		k = len(i.keys)
	}
	worstFirst := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return left.Score < right.Score
	}
	heap := container.NewHeap(worstFirst)
	for position, key := range i.keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if options.Filter != nil && !options.Filter(key) {
			continue
		}
		start, end := i.offsets[position], i.offsets[position+1]
		score, err := mathutil.SparseInnerProduct(
			i.indices[start:end], i.values[start:end], query.Indices, query.Values,
		)
		if err != nil {
			return nil, fmt.Errorf("core: score sparse candidate %d (key %d): %w", position, key, err)
		}
		if !scoreWithinRadius(MetricIP, score, options.Radius) {
			continue
		}
		result := Result{Key: key, Score: score}
		if heap.Len() < k {
			heap.Push(result)
			continue
		}
		worst, _ := heap.Peek()
		if resultBetter(MetricIP, result, worst) {
			heap.Replace(result)
		}
	}
	results := heap.Values()
	slices.SortFunc(results, func(left, right Result) int {
		if resultBetter(MetricIP, left, right) {
			return -1
		}
		if resultBetter(MetricIP, right, left) {
			return 1
		}
		return 0
	})
	return results, nil
}

// SparseFlatIndexBuilder is a one-shot sparse Flat builder.
type SparseFlatIndexBuilder struct {
	mu    sync.Mutex
	index *SparseFlatIndex
	built bool
}

// NewSparseFlatBuilder constructs a sparse IP builder.
func NewSparseFlatBuilder(metric Metric) (*SparseFlatIndexBuilder, error) {
	index, err := NewSparseFlatIndex(metric)
	if err != nil {
		return nil, err
	}
	return &SparseFlatIndexBuilder{index: index}, nil
}

// AddSparse appends while the builder is open.
func (b *SparseFlatIndexBuilder) AddSparse(ctx context.Context, key uint64, vector SparseVector) error {
	if b == nil {
		return errors.New("core: nil sparse Flat builder")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return ErrBuilderClosed
	}
	return b.index.AddSparse(ctx, key, vector)
}

// Build closes the builder and returns its streamable sparse index.
func (b *SparseFlatIndexBuilder) Build(ctx context.Context) (SparseIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil sparse Flat builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse Flat build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	b.built = true
	return b.index, nil
}

var (
	_ SparseProvider      = (*SparseFlatIndex)(nil)
	_ SparseSearcher      = (*SparseFlatIndex)(nil)
	_ SparseStreamer      = (*SparseFlatIndex)(nil)
	_ SparseIndex         = (*SparseFlatIndex)(nil)
	_ SparseQuerySearcher = (*SparseFlatIndex)(nil)
	_ SparseBuilder       = (*SparseFlatIndexBuilder)(nil)
)

var (
	ErrInvalidGroupCount = errors.New("core: group count must be positive")
	ErrInvalidGroupTopK  = errors.New("core: per-group top-k must be positive")
	ErrGroupSizeOverflow = errors.New("core: group candidate count overflows int")
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
	if _, err := mathutil.SparseInnerProduct(query.Indices, query.Values, nil, nil); err != nil {
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
		score, err := mathutil.SparseInnerProduct(
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
	if !metric.Valid() {
		return nil, errors.New("core: invalid group-by metric")
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	batches := make([][]GroupResult, len(searchers))
	err := parallel.ParallelFor(ctx, len(searchers), workers, func(ctx context.Context, index int) error {
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
	err := parallel.ParallelFor(ctx, len(searchers), workers, func(ctx context.Context, index int) error {
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
	groups       map[string]*container.Heap[Result]
}

func newGroupAccumulator(metric Metric, topKPerGroup int) *groupAccumulator {
	return &groupAccumulator{
		metric:       metric,
		topKPerGroup: topKPerGroup,
		groups:       make(map[string]*container.Heap[Result]),
	}
}

func (a *groupAccumulator) add(value string, result Result) {
	heap := a.groups[value]
	if heap == nil {
		heap = container.NewHeap(func(left, right Result) bool {
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
