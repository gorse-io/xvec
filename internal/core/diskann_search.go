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
	"io"
	"math"
	"slices"
	"sync"

	"github.com/gorse-io/zvec/internal/ailego"
)

var ErrInvalidDiskANNListSize = errors.New("core: invalid DiskANN list size")

// DiskANNSearchOptions combines common result controls with the graph
// candidate-list width. Linear is an exact disk scan used for diagnostics and
// query fallback.
type DiskANNSearchOptions struct {
	SearchOptions
	ListSize int
	Linear   bool
}

func (o DiskANNSearchOptions) Validate() error {
	if err := o.SearchOptions.Validate(); err != nil {
		return err
	}
	if o.ListSize <= 0 || uint64(o.ListSize) > math.MaxUint32 {
		return ErrInvalidDiskANNListSize
	}
	return nil
}

// DiskANNIndex owns immutable key/PQ metadata and serves graph nodes through
// a sector-aware ReaderAt. An opened index owns its file until Close.
type DiskANNIndex struct {
	closeMu         sync.RWMutex
	closed          bool
	closer          io.Closer
	dimension       int
	metric          Metric
	traversalMetric Metric
	options         DiskANNBuildOptions
	keys            []uint64
	positions       map[uint64]int
	entryPoint      int
	pq              *PQModel
	codes           []byte
	codeNorms       []float32
	nodes           *DiskANNNodeReader
}

func (i *DiskANNIndex) Dimension() int {
	if i == nil {
		return 0
	}
	return i.dimension
}

func (i *DiskANNIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.metric
}

func (i *DiskANNIndex) Len() int {
	if i == nil {
		return 0
	}
	return len(i.keys)
}

func (i *DiskANNIndex) BuildOptions() DiskANNBuildOptions {
	if i == nil {
		return DiskANNBuildOptions{}
	}
	return i.options
}

func (i *DiskANNIndex) PQChunks() int {
	if i == nil || i.pq == nil {
		return 0
	}
	return i.pq.Chunks()
}

func (i *DiskANNIndex) EntryPoint() (uint64, bool) {
	if i == nil || i.entryPoint < 0 || i.entryPoint >= len(i.keys) {
		return 0, false
	}
	return i.keys[i.entryPoint], true
}

func (i *DiskANNIndex) CacheStats() DiskANNCacheStats {
	if i == nil || i.nodes == nil {
		return DiskANNCacheStats{}
	}
	return i.nodes.CacheStats()
}

// Vector reads and clones one original FP32 vector by external key.
func (i *DiskANNIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil {
		return nil, false
	}
	i.closeMu.RLock()
	defer i.closeMu.RUnlock()
	if i.closed || i.nodes == nil {
		return nil, false
	}
	position, found := i.positions[key]
	if !found {
		return nil, false
	}
	node, err := i.nodes.ReadNode(context.Background(), uint32(position))
	if err != nil {
		return nil, false
	}
	return node.Vector, true
}

// Close releases an opened artifact. It is idempotent and waits for active
// searches that hold the immutable file generation.
func (i *DiskANNIndex) Close() error {
	if i == nil {
		return nil
	}
	i.closeMu.Lock()
	defer i.closeMu.Unlock()
	if i.closed {
		return nil
	}
	i.closed = true
	if i.closer != nil {
		return i.closer.Close()
	}
	return nil
}

func (i *DiskANNIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.searchDiskANN(ctx, query, DiskANNSearchOptions{
		SearchOptions: SearchOptions{TopK: k}, ListSize: DefaultDiskANNQueryList,
	}, false)
}

func (i *DiskANNIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.searchDiskANN(ctx, query, DiskANNSearchOptions{
		SearchOptions: options, ListSize: DefaultDiskANNQueryList,
	}, true)
}

// SearchDiskANN performs a bounded best-first traversal. PQ scores order the
// frontier; every expanded node is read from the node artifact and receives
// its exact public score before filter, radius, and top-k selection.
func (i *DiskANNIndex) SearchDiskANN(ctx context.Context, query []float32, options DiskANNSearchOptions) ([]Result, error) {
	return i.searchDiskANN(ctx, query, options, true)
}

func (i *DiskANNIndex) searchDiskANN(
	ctx context.Context,
	query []float32,
	options DiskANNSearchOptions,
	requirePositiveTopK bool,
) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil DiskANN index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.ListSize <= 0 || uint64(options.ListSize) > math.MaxUint32 {
		return nil, ErrInvalidDiskANNListSize
	}
	if requirePositiveTopK {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else {
		if options.TopK < 0 {
			return nil, errors.New("core: negative DiskANN top-k")
		}
		if options.Radius < 0 || math.IsNaN(float64(options.Radius)) || math.IsInf(float64(options.Radius), 0) {
			return nil, ErrInvalidRadius
		}
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if _, err := i.metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate DiskANN query: %w", err)
	}

	i.closeMu.RLock()
	defer i.closeMu.RUnlock()
	if i.closed {
		return nil, ErrDiskANNClosed
	}
	if options.TopK == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}
	if options.Linear {
		return i.searchDiskANNLinear(ctx, query, options.SearchOptions)
	}
	return i.searchDiskANNGraph(ctx, query, options)
}

func (i *DiskANNIndex) searchDiskANNLinear(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	collector := newDiskANNResultCollector(i.metric, options)
	batchSize := i.diskANNReadBatchSize()
	for start := 0; start < len(i.keys); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(len(i.keys), start+batchSize)
		ids := make([]uint32, end-start)
		for offset := range ids {
			ids[offset] = uint32(start + offset)
		}
		nodes, err := i.nodes.ReadNodes(ctx, ids)
		if err != nil {
			return nil, fmt.Errorf("core: linear DiskANN node read: %w", err)
		}
		for _, node := range nodes {
			score, err := i.metric.Compute(query, node.Vector)
			if err != nil {
				return nil, err
			}
			collector.Add(Result{Key: i.keys[node.ID], Score: score})
		}
	}
	return collector.Results(), nil
}

type diskANNQueueNode struct {
	id    uint32
	score float32
}

func (i *DiskANNIndex) searchDiskANNGraph(ctx context.Context, query []float32, options DiskANNSearchOptions) ([]Result, error) {
	if i.pq == nil || i.entryPoint < 0 || len(i.codes) == 0 {
		return nil, errors.New("core: inconsistent non-empty DiskANN traversal state")
	}
	prepared := prepareDiskANNPQQuery(query, i.metric)
	table, err := i.pq.DistanceTable(prepared)
	if err != nil {
		return nil, fmt.Errorf("core: build DiskANN PQ distance table: %w", err)
	}
	queryNorm := diskANNVectorNormSquared(prepared)
	capacity := min(len(i.keys), max(options.TopK, options.ListSize))
	better := func(left, right diskANNQueueNode) bool {
		if left.score == right.score {
			return left.id < right.id
		}
		return i.traversalMetric.Better(left.score, right.score)
	}
	worse := func(left, right diskANNQueueNode) bool { return better(right, left) }
	frontier := ailego.NewHeap(better)
	retained := ailego.NewHeap(worse)
	visited := make([]bool, len(i.keys))
	retainedMember := make([]bool, len(i.keys))
	expanded := make([]bool, len(i.keys))
	entry := uint32(i.entryPoint)
	entryScore, err := i.diskANNApproximateScore(table, entry, queryNorm)
	if err != nil {
		return nil, err
	}
	start := diskANNQueueNode{id: entry, score: entryScore}
	frontier.Push(start)
	retained.Push(start)
	visited[entry], retainedMember[entry] = true, true
	collector := newDiskANNResultCollector(i.metric, options.SearchOptions)
	beam := i.diskANNReadBatchSize()

	for frontier.Len() != 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		batch := make([]diskANNQueueNode, 0, beam)
		for frontier.Len() != 0 && len(batch) < cap(batch) {
			candidate, _ := frontier.Pop()
			if !retainedMember[candidate.id] || expanded[candidate.id] {
				continue
			}
			expanded[candidate.id] = true
			batch = append(batch, candidate)
		}
		if len(batch) == 0 {
			break
		}
		ids := make([]uint32, len(batch))
		for position := range batch {
			ids[position] = batch[position].id
		}
		nodes, err := i.nodes.ReadNodes(ctx, ids)
		if err != nil {
			return nil, fmt.Errorf("core: DiskANN graph node read: %w", err)
		}
		for _, node := range nodes {
			exact, err := i.metric.Compute(query, node.Vector)
			if err != nil {
				return nil, fmt.Errorf("core: score DiskANN node %d: %w", node.ID, err)
			}
			collector.Add(Result{Key: i.keys[node.ID], Score: exact})
			for _, neighbor := range node.Neighbors {
				if visited[neighbor] {
					continue
				}
				visited[neighbor] = true
				score, err := i.diskANNApproximateScore(table, neighbor, queryNorm)
				if err != nil {
					return nil, fmt.Errorf("core: score DiskANN PQ node %d: %w", neighbor, err)
				}
				candidate := diskANNQueueNode{id: neighbor, score: score}
				if retained.Len() < capacity {
					retained.Push(candidate)
					retainedMember[neighbor] = true
					frontier.Push(candidate)
					continue
				}
				worstNode, _ := retained.Peek()
				if better(candidate, worstNode) {
					replaced, _ := retained.Replace(candidate)
					retainedMember[replaced.id] = false
					retainedMember[neighbor] = true
					frontier.Push(candidate)
				}
			}
		}
	}
	return collector.Results(), nil
}

func (i *DiskANNIndex) diskANNApproximateScore(table *PQDistanceTable, id uint32, queryNorm float64) (float32, error) {
	chunks := i.pq.Chunks()
	start := int(id) * chunks
	code := PQCode{modelFingerprint: i.pq.fingerprint, codes: i.codes[start : start+chunks]}
	score, err := table.Lookup(code)
	if err != nil {
		return 0, err
	}
	if i.metric != MetricMIPSL2 {
		return score, nil
	}
	candidateNorm := float64(i.codeNorms[id])
	inner := (queryNorm + candidateNorm - float64(score)) / 2
	denominator := max(queryNorm, candidateNorm)
	if denominator == 0 {
		return 0, nil
	}
	converted := 2 - 2*inner/denominator
	if math.IsNaN(converted) || math.IsInf(converted, 0) || converted > math.MaxFloat32 || converted < -math.MaxFloat32 {
		return 0, ErrPQScoreOverflow
	}
	return float32(converted), nil
}

func diskANNVectorNormSquared(vector []float32) float64 {
	var norm float64
	for _, value := range vector {
		norm += float64(value) * float64(value)
	}
	return norm
}

func (i *DiskANNIndex) diskANNReadBatchSize() int {
	sectors := 1
	if i != nil && i.nodes != nil {
		sectors = max(1, i.nodes.layout.sectorsPerNode)
	}
	return max(1, MaxDiskANNReadSectors/sectors)
}

type diskANNResultCollector struct {
	metric  Metric
	options SearchOptions
	heap    *ailego.Heap[Result]
}

func newDiskANNResultCollector(metric Metric, options SearchOptions) *diskANNResultCollector {
	worse := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return metric.Better(right.Score, left.Score)
	}
	return &diskANNResultCollector{metric: metric, options: options, heap: ailego.NewHeap(worse)}
}

func (c *diskANNResultCollector) Add(result Result) {
	if (c.options.Filter != nil && !c.options.Filter(result.Key)) || !scoreWithinRadius(c.metric, result.Score, c.options.Radius) {
		return
	}
	if c.heap.Len() < c.options.TopK {
		c.heap.Push(result)
		return
	}
	worst, _ := c.heap.Peek()
	if resultBetter(c.metric, result, worst) {
		c.heap.Replace(result)
	}
}

func (c *diskANNResultCollector) Results() []Result {
	results := c.heap.Values()
	slices.SortFunc(results, func(left, right Result) int {
		if resultBetter(c.metric, left, right) {
			return -1
		}
		if resultBetter(c.metric, right, left) {
			return 1
		}
		return 0
	})
	if results == nil {
		return []Result{}
	}
	return results
}

// WarmCache follows graph edges breadth-first from the medoid and reads up to
// count nodes. The effective count never exceeds cache capacity.
func (i *DiskANNIndex) WarmCache(ctx context.Context, count int) (int, error) {
	if i == nil {
		return 0, errors.New("core: nil DiskANN index")
	}
	if ctx == nil {
		return 0, errors.New("core: nil DiskANN cache context")
	}
	if count < 0 {
		return 0, fmt.Errorf("%w: negative cache warm count", ErrInvalidDiskANNOptions)
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	i.closeMu.RLock()
	defer i.closeMu.RUnlock()
	if i.closed {
		return 0, ErrDiskANNClosed
	}
	target := min(count, i.nodes.cache.Capacity(), len(i.keys))
	if target == 0 {
		return 0, nil
	}
	visited := make([]bool, len(i.keys))
	queue := []uint32{uint32(i.entryPoint)}
	visited[i.entryPoint] = true
	warmed := 0
	for len(queue) != 0 && warmed < target {
		batchCount := min(i.diskANNReadBatchSize(), target-warmed, len(queue))
		ids := slices.Clone(queue[:batchCount])
		queue = queue[batchCount:]
		nodes, err := i.nodes.ReadNodes(ctx, ids)
		if err != nil {
			return warmed, err
		}
		warmed += len(nodes)
		for _, node := range nodes {
			for _, neighbor := range node.Neighbors {
				if !visited[neighbor] {
					visited[neighbor] = true
					queue = append(queue, neighbor)
				}
			}
		}
	}
	return warmed, nil
}

var (
	_ DenseProvider      = (*DiskANNIndex)(nil)
	_ DenseSearcher      = (*DiskANNIndex)(nil)
	_ DenseQuerySearcher = (*DiskANNIndex)(nil)
)
