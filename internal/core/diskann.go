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
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/ailego/io"
)

const (
	DefaultDiskANNMaxDegree    = 100
	DefaultDiskANNBuildList    = 50
	DefaultDiskANNQueryList    = 300
	DefaultDiskANNCacheNodes   = 1024
	DefaultDiskANNMaxOcclusion = 750
)

var (
	ErrInvalidDiskANNOptions = errors.New("core: invalid DiskANN options")
	ErrDiskANNKeyNotFound    = errors.New("core: DiskANN key not found")
	ErrDiskANNClosed         = errors.New("core: DiskANN index is closed")
	ErrDiskANNCapacity       = errors.New("core: DiskANN index capacity exceeded")
)

// DiskANNBuildOptions configures graph construction, product quantization,
// random-read concurrency, and the demand cache.
type DiskANNBuildOptions struct {
	Metric        Metric
	MaxDegree     int
	ListSize      int
	PQChunks      int
	Workers       int
	CacheCapacity int
}

// DefaultDiskANNBuildOptions returns the pinned public construction defaults.
func DefaultDiskANNBuildOptions(metric Metric) DiskANNBuildOptions {
	return DiskANNBuildOptions{
		Metric: metric, MaxDegree: DefaultDiskANNMaxDegree,
		ListSize: DefaultDiskANNBuildList, CacheCapacity: DefaultDiskANNCacheNodes,
	}
}

// Validate checks invariants that do not depend on vector dimension.
func (o DiskANNBuildOptions) Validate() error {
	if !o.Metric.Valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidDiskANNOptions)
	}
	if o.MaxDegree <= 0 || o.MaxDegree > MaxVamanaDegree {
		return fmt.Errorf("%w: MaxDegree must be in [1,%d]", ErrInvalidDiskANNOptions, MaxVamanaDegree)
	}
	if o.ListSize <= 0 || uint64(o.ListSize) > math.MaxUint32 {
		return fmt.Errorf("%w: ListSize must fit uint32", ErrInvalidDiskANNOptions)
	}
	if o.PQChunks < 0 {
		return fmt.Errorf("%w: PQChunks cannot be negative", ErrInvalidDiskANNOptions)
	}
	if o.Workers < 0 || o.CacheCapacity < 0 {
		return fmt.Errorf("%w: Workers and CacheCapacity cannot be negative", ErrInvalidDiskANNOptions)
	}
	return nil
}

// DiskANNBuilder collects original vectors for one immutable disk graph.
type DiskANNBuilder struct {
	mu        sync.Mutex
	dimension int
	options   DiskANNBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

func NewDiskANNBuilder(dimension int, options DiskANNBuildOptions) (*DiskANNBuilder, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if options.PQChunks > dimension {
		return nil, fmt.Errorf("%w: PQChunks cannot exceed dimension", ErrInvalidDiskANNOptions)
	}
	return &DiskANNBuilder{dimension: dimension, options: options, positions: make(map[uint64]int)}, nil
}

// Add validates and clones one unique original vector.
func (b *DiskANNBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil DiskANN builder")
	}
	if ctx == nil {
		return errors.New("core: nil DiskANN add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate DiskANN vector: %w", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if b.built {
		return ErrBuilderClosed
	}
	if _, found := b.positions[key]; found {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	if uint64(len(b.keys)) >= math.MaxUint32 || len(b.vectors) > maxPlatformInt()-b.dimension {
		return ErrDiskANNCapacity
	}
	b.positions[key] = len(b.keys)
	b.keys = append(b.keys, key)
	b.vectors = append(b.vectors, vector...)
	return nil
}

// Build constructs a Vamana topology, trains PQ traversal codes, and converts
// the graph to the native sector layout. The returned in-memory ReaderAt index
// has identical search semantics to an index reopened from disk.
func (b *DiskANNBuilder) Build(ctx context.Context) (*DiskANNIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil DiskANN builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}

	graphOptions := DefaultVamanaBuildOptions(b.options.Metric)
	graphOptions.MaxDegree = b.options.MaxDegree
	graphOptions.SearchListSize = max(b.options.ListSize, b.options.MaxDegree)
	graphOptions.MaxOcclusionSize = DefaultDiskANNMaxOcclusion
	graphBuilder, err := NewVamanaBuilder(b.dimension, graphOptions)
	if err != nil {
		return nil, err
	}
	vectors := make([][]float32, len(b.keys))
	for position, key := range b.keys {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		start := position * b.dimension
		vectors[position] = b.vectors[start : start+b.dimension]
		if err := graphBuilder.Add(ctx, key, vectors[position]); err != nil {
			return nil, fmt.Errorf("core: add DiskANN graph vector %d: %w", position, err)
		}
	}
	graph, err := graphBuilder.Build(ctx)
	if err != nil {
		return nil, fmt.Errorf("core: build DiskANN graph: %w", err)
	}

	layout, err := NewDiskANNLayout(b.options.Metric, len(b.keys), b.dimension, b.options.MaxDegree)
	if err != nil {
		return nil, err
	}
	nodes := make([]DiskANNNode, len(b.keys))
	for position := range nodes {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		nodes[position] = DiskANNNode{ID: uint32(position), Vector: slices.Clone(vectors[position])}
		nodes[position].Neighbors = make([]uint32, len(graph.neighbors[position]))
		for offset, neighbor := range graph.neighbors[position] {
			nodes[position].Neighbors[offset] = uint32(neighbor)
		}
	}
	nodeArtifact, err := encodeDiskANNNodeFile(ctx, layout, nodes)
	if err != nil {
		return nil, err
	}
	nodeReader, err := OpenDiskANNNodeReader(
		ctx, bytes.NewReader(nodeArtifact), int64(len(nodeArtifact)), b.options.CacheCapacity, b.options.Workers,
	)
	if err != nil {
		return nil, err
	}

	index := &DiskANNIndex{
		dimension: b.dimension, metric: b.options.Metric, options: b.options,
		keys: slices.Clone(b.keys), positions: cloneUint64Positions(b.positions), entryPoint: graph.entryPoint,
		nodes: nodeReader,
	}
	index.traversalMetric = diskANNTraversalMetric(b.options.Metric)
	if len(vectors) != 0 {
		prepared, traversalMetric, err := prepareDiskANNPQVectors(ctx, vectors, b.options.Metric)
		if err != nil {
			return nil, err
		}
		pqOptions := DefaultPQOptions(traversalMetric)
		pqOptions.Chunks, pqOptions.Workers = b.options.PQChunks, b.options.Workers
		model, err := TrainPQ(ctx, prepared, pqOptions)
		if err != nil {
			return nil, fmt.Errorf("core: train DiskANN PQ: %w", err)
		}
		codes, err := model.EncodeBatch(ctx, prepared, b.options.Workers)
		if err != nil {
			return nil, fmt.Errorf("core: encode DiskANN PQ: %w", err)
		}
		index.pq, index.traversalMetric = model, traversalMetric
		index.codes = make([]byte, len(codes)*model.Chunks())
		for position, code := range codes {
			copy(index.codes[position*model.Chunks():], code.codes)
		}
		if b.options.Metric == MetricMIPSL2 {
			index.codeNorms, err = diskANNPQCodeNorms(ctx, model, index.codes, len(codes))
			if err != nil {
				return nil, err
			}
		}
	}
	if err := validateDiskANNIndex(ctx, index); err != nil {
		return nil, err
	}
	b.built = true
	b.keys, b.vectors, b.positions = nil, nil, nil
	return index, nil
}

func prepareDiskANNPQVectors(ctx context.Context, vectors [][]float32, metric Metric) ([][]float32, Metric, error) {
	prepared := make([][]float32, len(vectors))
	traversalMetric := diskANNTraversalMetric(metric)
	for position, vector := range vectors {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, err
			}
		}
		prepared[position] = slices.Clone(vector)
		if metric == MetricCosine {
			normalizeRaBitQVector(prepared[position])
		}
	}
	return prepared, traversalMetric, nil
}

func diskANNTraversalMetric(metric Metric) Metric {
	if metric == MetricCosine || metric == MetricMIPSL2 {
		return MetricL2
	}
	return metric
}

func prepareDiskANNPQQuery(query []float32, metric Metric) []float32 {
	prepared := slices.Clone(query)
	if metric == MetricCosine {
		normalizeRaBitQVector(prepared)
	}
	return prepared
}

func diskANNPQCodeNorms(ctx context.Context, model *PQModel, raw []byte, count int) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN PQ norm context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model == nil || model.Chunks() == 0 || count < 0 ||
		uint64(count)*uint64(model.Chunks()) > uint64(maxPlatformInt()) ||
		len(raw) != int(uint64(count)*uint64(model.Chunks())) {
		return nil, ErrInvalidPQModel
	}
	norms := make([]float32, count)
	for position := range norms {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		var norm float64
		code := raw[position*model.Chunks() : (position+1)*model.Chunks()]
		for chunk, centroid := range code {
			start, end := model.chunkOffsets[chunk], model.chunkOffsets[chunk+1]
			pivot := int(centroid)*model.dimension + start
			for component, value := range model.pivots[pivot : pivot+end-start] {
				if component&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				norm += float64(value) * float64(value)
			}
		}
		norms[position] = float32(norm)
	}
	return norms, nil
}

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
	if i.closer != nil {
		if err := i.closer.Close(); err != nil {
			return err
		}
	}
	i.closed = true
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
	frontier := container.NewHeap(better)
	retained := container.NewHeap(worse)
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
	heap    *container.Heap[Result]
}

func newDiskANNResultCollector(metric Metric, options SearchOptions) *diskANNResultCollector {
	worse := func(left, right Result) bool {
		if left.Score == right.Score {
			return left.Key > right.Key
		}
		return metric.Better(right.Score, left.Score)
	}
	return &diskANNResultCollector{metric: metric, options: options, heap: container.NewHeap(worse)}
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

const (
	diskANNIndexFileVersion  = 1
	diskANNIndexHeaderSize   = DiskANNSectorSize
	diskANNIndexHeaderCRCPos = diskANNIndexHeaderSize - 4
)

var (
	diskANNIndexMagic = [8]byte{'Z', 'V', 'E', 'C', 'D', 'I', 'D', 'X'}

	ErrInvalidDiskANNFile             = errors.New("core: invalid DiskANN index file")
	ErrDiskANNIndexChecksumMismatch   = errors.New("core: DiskANN index checksum mismatch")
	ErrUnsupportedDiskANNIndexVersion = errors.New("core: unsupported DiskANN index version")
	errDiskANNIndexSectionChecksum    = errors.New("core: DiskANN index section checksum mismatch")
)

type diskANNIndexSections struct {
	totalLength   int64
	keysOffset    int64
	keysLength    int64
	offsetsOffset int64
	offsetsLength int64
	pivotsOffset  int64
	pivotsLength  int64
	codesOffset   int64
	codesLength   int64
	nodesOffset   int64
	nodesLength   int64
}

type diskANNIndexHeader struct {
	count            int
	dimension        int
	metric           Metric
	traversalMetric  Metric
	maxDegree        int
	listSize         int
	configuredChunks int
	actualChunks     int
	entryPoint       int
	sections         diskANNIndexSections
	keysCRC          uint32
	offsetsCRC       uint32
	pivotsCRC        uint32
	codesCRC         uint32
	nodesCRC         uint32
}

// Save atomically publishes a complete native DiskANN artifact.
func (i *DiskANNIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil DiskANN save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidDiskANNFile)
	}
	if i == nil {
		return fmt.Errorf("%w: nil index", ErrInvalidDiskANNFile)
	}
	i.closeMu.RLock()
	defer i.closeMu.RUnlock()
	if i.closed {
		return ErrDiskANNClosed
	}
	encoded, err := encodeDiskANNIndex(ctx, i)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save DiskANN file: %w", err)
	}
	return nil
}

// OpenDiskANNIndex opens and validates a complete artifact. cacheCapacity zero
// disables node caching; workers zero lets the shared parallel helper choose.
func OpenDiskANNIndex(ctx context.Context, path string, cacheCapacity, workers int) (*DiskANNIndex, error) {
	return OpenDiskANNIndexWithMmap(ctx, path, cacheCapacity, workers, false)
}

// OpenDiskANNIndexWithMmap opens a complete artifact through either ordinary
// file reads or a read-only memory mapping. The returned index owns the reader
// and releases it from Close.
func OpenDiskANNIndexWithMmap(ctx context.Context, path string, cacheCapacity, workers int, useMmap bool) (*DiskANNIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidDiskANNFile)
	}
	if cacheCapacity < 0 || workers < 0 {
		return nil, fmt.Errorf("%w: negative runtime option", ErrInvalidDiskANNOptions)
	}
	reader, err := ioutil.OpenReaderAt(path, useMmap)
	if err != nil {
		return nil, fmt.Errorf("core: open DiskANN file: %w", err)
	}
	index, err := openDiskANNIndexReader(ctx, reader, reader.Size(), cacheCapacity, workers, reader)
	if err != nil {
		_ = reader.Close()
		return nil, err
	}
	return index, nil
}

func encodeDiskANNIndex(ctx context.Context, index *DiskANNIndex) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateDiskANNIndex(ctx, index); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidDiskANNFile, err)
	}
	nodeLength := index.nodes.layout.TotalLength()
	if nodeLength < 0 || nodeLength > int64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: node artifact exceeds platform capacity", ErrInvalidDiskANNFile)
	}
	nodeArtifact := make([]byte, int(nodeLength))
	if err := readFullAt(ctx, index.nodes.reader, nodeArtifact, 0); err != nil {
		return nil, fmt.Errorf("core: snapshot DiskANN node artifact: %w", err)
	}
	actualChunks := 0
	if index.pq != nil {
		actualChunks = index.pq.Chunks()
	}
	sections, err := calculateDiskANNIndexSections(len(index.keys), index.dimension, actualChunks, nodeLength)
	if err != nil {
		return nil, err
	}
	if sections.totalLength > int64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: artifact exceeds platform capacity", ErrInvalidDiskANNFile)
	}
	encoded := make([]byte, int(sections.totalLength))
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		binary.LittleEndian.PutUint64(encoded[int(sections.keysOffset)+position*8:], key)
	}
	if index.pq != nil {
		state := index.pq.State()
		for position, offset := range state.ChunkOffsets {
			binary.LittleEndian.PutUint32(encoded[int(sections.offsetsOffset)+position*4:], uint32(offset))
		}
		for position, pivot := range state.Pivots {
			if position&16383 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			binary.LittleEndian.PutUint32(encoded[int(sections.pivotsOffset)+position*4:], math.Float32bits(pivot))
		}
		copy(encoded[sections.codesOffset:sections.codesOffset+sections.codesLength], index.codes)
	}
	copy(encoded[sections.nodesOffset:sections.nodesOffset+sections.nodesLength], nodeArtifact)
	header := diskANNIndexHeader{
		count: len(index.keys), dimension: index.dimension, metric: index.metric,
		traversalMetric: index.traversalMetric, maxDegree: index.options.MaxDegree,
		listSize: index.options.ListSize, configuredChunks: index.options.PQChunks,
		actualChunks: actualChunks, entryPoint: index.entryPoint, sections: sections,
		keysCRC:    hashutil.CRC32C(encoded[sections.keysOffset : sections.keysOffset+sections.keysLength]),
		offsetsCRC: hashutil.CRC32C(encoded[sections.offsetsOffset : sections.offsetsOffset+sections.offsetsLength]),
		pivotsCRC:  hashutil.CRC32C(encoded[sections.pivotsOffset : sections.pivotsOffset+sections.pivotsLength]),
		codesCRC:   hashutil.CRC32C(encoded[sections.codesOffset : sections.codesOffset+sections.codesLength]),
		nodesCRC:   hashutil.CRC32C(nodeArtifact),
	}
	copy(encoded[:diskANNIndexHeaderSize], encodeDiskANNIndexHeader(header))
	return encoded, nil
}

func encodeDiskANNIndexHeader(meta diskANNIndexHeader) []byte {
	header := make([]byte, diskANNIndexHeaderSize)
	copy(header[:8], diskANNIndexMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], diskANNIndexFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], diskANNIndexHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(meta.sections.totalLength))
	binary.LittleEndian.PutUint64(header[24:32], uint64(meta.count))
	binary.LittleEndian.PutUint32(header[32:36], uint32(meta.dimension))
	header[36], header[37] = byte(meta.metric), byte(meta.traversalMetric)
	binary.LittleEndian.PutUint32(header[40:44], uint32(meta.maxDegree))
	binary.LittleEndian.PutUint32(header[44:48], uint32(meta.listSize))
	binary.LittleEndian.PutUint32(header[48:52], uint32(meta.configuredChunks))
	binary.LittleEndian.PutUint32(header[52:56], uint32(meta.actualChunks))
	entry := uint64(math.MaxUint64)
	if meta.entryPoint >= 0 {
		entry = uint64(meta.entryPoint)
	}
	binary.LittleEndian.PutUint64(header[56:64], entry)
	binary.LittleEndian.PutUint64(header[64:72], uint64(meta.sections.keysOffset))
	binary.LittleEndian.PutUint64(header[72:80], uint64(meta.sections.keysLength))
	binary.LittleEndian.PutUint64(header[80:88], uint64(meta.sections.offsetsOffset))
	binary.LittleEndian.PutUint64(header[88:96], uint64(meta.sections.offsetsLength))
	binary.LittleEndian.PutUint64(header[96:104], uint64(meta.sections.pivotsOffset))
	binary.LittleEndian.PutUint64(header[104:112], uint64(meta.sections.pivotsLength))
	binary.LittleEndian.PutUint64(header[112:120], uint64(meta.sections.codesOffset))
	binary.LittleEndian.PutUint64(header[120:128], uint64(meta.sections.codesLength))
	binary.LittleEndian.PutUint64(header[128:136], uint64(meta.sections.nodesOffset))
	binary.LittleEndian.PutUint64(header[136:144], uint64(meta.sections.nodesLength))
	binary.LittleEndian.PutUint32(header[144:148], meta.keysCRC)
	binary.LittleEndian.PutUint32(header[148:152], meta.offsetsCRC)
	binary.LittleEndian.PutUint32(header[152:156], meta.pivotsCRC)
	binary.LittleEndian.PutUint32(header[156:160], meta.codesCRC)
	binary.LittleEndian.PutUint32(header[160:164], meta.nodesCRC)
	binary.LittleEndian.PutUint32(header[diskANNIndexHeaderCRCPos:], hashutil.CRC32C(header[:diskANNIndexHeaderCRCPos]))
	return header
}

func openDiskANNIndexReader(
	ctx context.Context,
	reader io.ReaderAt,
	fileSize int64,
	cacheCapacity, workers int,
	closer io.Closer,
) (*DiskANNIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil DiskANN reader-open context")
	}
	if reader == nil || fileSize < diskANNIndexHeaderSize || cacheCapacity < 0 || workers < 0 {
		return nil, ErrInvalidDiskANNFile
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	headerBytes := make([]byte, diskANNIndexHeaderSize)
	if err := readFullAt(ctx, reader, headerBytes, 0); err != nil {
		return nil, err
	}
	meta, err := decodeDiskANNIndexHeader(headerBytes, fileSize)
	if err != nil {
		return nil, err
	}
	checks := []struct {
		offset, length int64
		checksum       uint32
		name           string
	}{
		{meta.sections.keysOffset, meta.sections.keysLength, meta.keysCRC, "keys"},
		{meta.sections.offsetsOffset, meta.sections.offsetsLength, meta.offsetsCRC, "chunk offsets"},
		{meta.sections.pivotsOffset, meta.sections.pivotsLength, meta.pivotsCRC, "pivots"},
		{meta.sections.codesOffset, meta.sections.codesLength, meta.codesCRC, "codes"},
		{meta.sections.nodesOffset, meta.sections.nodesLength, meta.nodesCRC, "nodes"},
	}
	for _, check := range checks {
		if err := verifyDiskANNIndexSection(ctx, reader, check.offset, check.length, check.checksum); err != nil {
			if errors.Is(err, errDiskANNIndexSectionChecksum) {
				return nil, fmt.Errorf("%w: %s: %v", ErrDiskANNIndexChecksumMismatch, check.name, err)
			}
			return nil, err
		}
	}
	metadataEnd := meta.sections.codesOffset + meta.sections.codesLength
	if meta.sections.nodesOffset > metadataEnd {
		padding, err := readDiskANNIndexSection(ctx, reader, metadataEnd, meta.sections.nodesOffset-metadataEnd)
		if err != nil {
			return nil, err
		}
		if !allZeroBytes(padding) {
			return nil, fmt.Errorf("%w: non-zero alignment padding", ErrInvalidDiskANNFile)
		}
	}

	keysBytes, err := readDiskANNIndexSection(ctx, reader, meta.sections.keysOffset, meta.sections.keysLength)
	if err != nil {
		return nil, err
	}
	keys := make([]uint64, meta.count)
	positions := make(map[uint64]int, meta.count)
	for position := range keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := binary.LittleEndian.Uint64(keysBytes[position*8:])
		if _, duplicate := positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key", ErrInvalidDiskANNFile)
		}
		keys[position], positions[key] = key, position
	}

	var model *PQModel
	var codes []byte
	if meta.count != 0 {
		offsetBytes, err := readDiskANNIndexSection(ctx, reader, meta.sections.offsetsOffset, meta.sections.offsetsLength)
		if err != nil {
			return nil, err
		}
		offsets := make([]int, meta.actualChunks+1)
		for position := range offsets {
			value := uint64(binary.LittleEndian.Uint32(offsetBytes[position*4:]))
			if value > uint64(maxPlatformInt()) {
				return nil, fmt.Errorf("%w: chunk offset exceeds platform capacity", ErrInvalidDiskANNFile)
			}
			offsets[position] = int(value)
		}
		pivotBytes, err := readDiskANNIndexSection(ctx, reader, meta.sections.pivotsOffset, meta.sections.pivotsLength)
		if err != nil {
			return nil, err
		}
		pivots := make([]float32, len(pivotBytes)/4)
		for position := range pivots {
			if position&16383 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			pivots[position] = math.Float32frombits(binary.LittleEndian.Uint32(pivotBytes[position*4:]))
		}
		model, err = RestorePQModel(PQModelState{
			Dimension: meta.dimension, Metric: meta.traversalMetric,
			ChunkOffsets: offsets, Pivots: pivots,
		})
		if err != nil {
			return nil, fmt.Errorf("%w: restore PQ model: %v", ErrInvalidDiskANNFile, err)
		}
		codes, err = readDiskANNIndexSection(ctx, reader, meta.sections.codesOffset, meta.sections.codesLength)
		if err != nil {
			return nil, err
		}
	}
	nodeSection := io.NewSectionReader(reader, meta.sections.nodesOffset, meta.sections.nodesLength)
	nodeReader, err := OpenDiskANNNodeReader(ctx, nodeSection, meta.sections.nodesLength, cacheCapacity, workers)
	if err != nil {
		return nil, fmt.Errorf("core: open DiskANN node section: %w", err)
	}
	index := &DiskANNIndex{
		closer: closer, dimension: meta.dimension, metric: meta.metric,
		traversalMetric: meta.traversalMetric,
		options: DiskANNBuildOptions{
			Metric: meta.metric, MaxDegree: meta.maxDegree, ListSize: meta.listSize,
			PQChunks: meta.configuredChunks, Workers: workers, CacheCapacity: cacheCapacity,
		},
		keys: keys, positions: positions, entryPoint: meta.entryPoint,
		pq: model, codes: codes, nodes: nodeReader,
	}
	if meta.metric == MetricMIPSL2 && model != nil {
		index.codeNorms, err = diskANNPQCodeNorms(ctx, model, codes, meta.count)
		if err != nil {
			return nil, err
		}
	}
	if err := validateDiskANNIndex(ctx, index); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidDiskANNFile, err)
	}
	return index, nil
}

func decodeDiskANNIndexHeader(header []byte, fileSize int64) (diskANNIndexHeader, error) {
	if len(header) != diskANNIndexHeaderSize || !bytes.Equal(header[:8], diskANNIndexMagic[:]) {
		return diskANNIndexHeader{}, fmt.Errorf("%w: bad header", ErrInvalidDiskANNFile)
	}
	if version := binary.LittleEndian.Uint16(header[8:10]); version != diskANNIndexFileVersion {
		return diskANNIndexHeader{}, fmt.Errorf("%w: %d", ErrUnsupportedDiskANNIndexVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != diskANNIndexHeaderSize || binary.LittleEndian.Uint32(header[12:16]) != 0 ||
		!allZeroBytes(header[38:40]) || !allZeroBytes(header[164:diskANNIndexHeaderCRCPos]) {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid reserved header fields", ErrInvalidDiskANNFile)
	}
	if got, want := hashutil.CRC32C(header[:diskANNIndexHeaderCRCPos]), binary.LittleEndian.Uint32(header[diskANNIndexHeaderCRCPos:]); got != want {
		return diskANNIndexHeader{}, fmt.Errorf("%w: header got %08x, want %08x", ErrDiskANNIndexChecksumMismatch, got, want)
	}
	total := binary.LittleEndian.Uint64(header[16:24])
	count64 := binary.LittleEndian.Uint64(header[24:32])
	if total > math.MaxInt64 || int64(total) != fileSize || count64 > math.MaxUint32 || count64 > uint64(maxPlatformInt()) {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid total length or count", ErrInvalidDiskANNFile)
	}
	values := []uint32{
		binary.LittleEndian.Uint32(header[32:36]), binary.LittleEndian.Uint32(header[40:44]),
		binary.LittleEndian.Uint32(header[44:48]), binary.LittleEndian.Uint32(header[48:52]),
		binary.LittleEndian.Uint32(header[52:56]),
	}
	for _, value := range values {
		if uint64(value) > uint64(maxPlatformInt()) {
			return diskANNIndexHeader{}, fmt.Errorf("%w: header value exceeds platform capacity", ErrInvalidDiskANNFile)
		}
	}
	meta := diskANNIndexHeader{
		count: int(count64), dimension: int(values[0]), metric: Metric(header[36]), traversalMetric: Metric(header[37]),
		maxDegree: int(values[1]), listSize: int(values[2]), configuredChunks: int(values[3]), actualChunks: int(values[4]),
		keysCRC: binary.LittleEndian.Uint32(header[144:148]), offsetsCRC: binary.LittleEndian.Uint32(header[148:152]),
		pivotsCRC: binary.LittleEndian.Uint32(header[152:156]), codesCRC: binary.LittleEndian.Uint32(header[156:160]),
		nodesCRC: binary.LittleEndian.Uint32(header[160:164]),
	}
	options := DiskANNBuildOptions{
		Metric: meta.metric, MaxDegree: meta.maxDegree, ListSize: meta.listSize, PQChunks: meta.configuredChunks,
	}
	if err := options.Validate(); err != nil || meta.dimension <= 0 || meta.dimension > MaxRotationDimension || meta.configuredChunks > meta.dimension {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid options", ErrInvalidDiskANNFile)
	}
	expectedTraversal := diskANNTraversalMetric(meta.metric)
	if meta.traversalMetric != expectedTraversal {
		return diskANNIndexHeader{}, fmt.Errorf("%w: invalid traversal metric", ErrInvalidDiskANNFile)
	}
	entry64 := binary.LittleEndian.Uint64(header[56:64])
	if meta.count == 0 {
		if entry64 != math.MaxUint64 || meta.actualChunks != 0 {
			return diskANNIndexHeader{}, fmt.Errorf("%w: invalid empty state", ErrInvalidDiskANNFile)
		}
		meta.entryPoint = -1
	} else {
		if entry64 >= uint64(meta.count) || meta.actualChunks <= 0 || meta.actualChunks > meta.dimension ||
			(meta.configuredChunks != 0 && meta.configuredChunks != meta.actualChunks) {
			return diskANNIndexHeader{}, fmt.Errorf("%w: invalid entry point or PQ chunks", ErrInvalidDiskANNFile)
		}
		meta.entryPoint = int(entry64)
	}
	nodeLength64 := binary.LittleEndian.Uint64(header[136:144])
	if nodeLength64 > math.MaxInt64 {
		return diskANNIndexHeader{}, fmt.Errorf("%w: node length overflow", ErrInvalidDiskANNFile)
	}
	sections, err := calculateDiskANNIndexSections(meta.count, meta.dimension, meta.actualChunks, int64(nodeLength64))
	if err != nil {
		return diskANNIndexHeader{}, err
	}
	serialized := []uint64{
		binary.LittleEndian.Uint64(header[64:72]), binary.LittleEndian.Uint64(header[72:80]),
		binary.LittleEndian.Uint64(header[80:88]), binary.LittleEndian.Uint64(header[88:96]),
		binary.LittleEndian.Uint64(header[96:104]), binary.LittleEndian.Uint64(header[104:112]),
		binary.LittleEndian.Uint64(header[112:120]), binary.LittleEndian.Uint64(header[120:128]),
		binary.LittleEndian.Uint64(header[128:136]), nodeLength64,
	}
	expected := []int64{
		sections.keysOffset, sections.keysLength, sections.offsetsOffset, sections.offsetsLength,
		sections.pivotsOffset, sections.pivotsLength, sections.codesOffset, sections.codesLength,
		sections.nodesOffset, sections.nodesLength,
	}
	for position := range expected {
		if serialized[position] > math.MaxInt64 || int64(serialized[position]) != expected[position] {
			return diskANNIndexHeader{}, fmt.Errorf("%w: inconsistent section layout", ErrInvalidDiskANNFile)
		}
	}
	if sections.totalLength != fileSize {
		return diskANNIndexHeader{}, fmt.Errorf("%w: inconsistent file length", ErrInvalidDiskANNFile)
	}
	meta.sections = sections
	return meta, nil
}

func calculateDiskANNIndexSections(count, dimension, chunks int, nodeLength int64) (diskANNIndexSections, error) {
	if count < 0 || dimension <= 0 || chunks < 0 || nodeLength < diskANNNodeHeaderSize {
		return diskANNIndexSections{}, fmt.Errorf("%w: invalid section inputs", ErrInvalidDiskANNFile)
	}
	if (count == 0 && chunks != 0) || (count > 0 && (chunks <= 0 || chunks > dimension)) {
		return diskANNIndexSections{}, fmt.Errorf("%w: invalid PQ section inputs", ErrInvalidDiskANNFile)
	}
	checkedProduct := func(left, right uint64) (int64, error) {
		if right != 0 && left > math.MaxUint64/right {
			return 0, ErrInvalidDiskANNFile
		}
		value := left * right
		if value > math.MaxInt64 {
			return 0, ErrInvalidDiskANNFile
		}
		return int64(value), nil
	}
	keysLength, err := checkedProduct(uint64(count), 8)
	if err != nil {
		return diskANNIndexSections{}, fmt.Errorf("%w: keys length overflow", ErrInvalidDiskANNFile)
	}
	offsetsLength, pivotsLength, codesLength := int64(0), int64(0), int64(0)
	if count != 0 {
		offsetsLength, err = checkedProduct(uint64(chunks+1), 4)
		if err == nil {
			pivotsLength, err = checkedProduct(uint64(PQCentroidCount)*uint64(dimension), 4)
		}
		if err == nil {
			codesLength, err = checkedProduct(uint64(count), uint64(chunks))
		}
		if err != nil {
			return diskANNIndexSections{}, fmt.Errorf("%w: PQ length overflow", ErrInvalidDiskANNFile)
		}
	}
	sections := diskANNIndexSections{keysOffset: diskANNIndexHeaderSize, keysLength: keysLength}
	sections.offsetsOffset = sections.keysOffset + sections.keysLength
	sections.offsetsLength = offsetsLength
	sections.pivotsOffset = sections.offsetsOffset + sections.offsetsLength
	sections.pivotsLength = pivotsLength
	sections.codesOffset = sections.pivotsOffset + sections.pivotsLength
	sections.codesLength = codesLength
	metadataEnd := sections.codesOffset + sections.codesLength
	if metadataEnd < 0 || metadataEnd > math.MaxInt64-(DiskANNSectorSize-1) {
		return diskANNIndexSections{}, fmt.Errorf("%w: metadata length overflow", ErrInvalidDiskANNFile)
	}
	sections.nodesOffset = (metadataEnd + DiskANNSectorSize - 1) / DiskANNSectorSize * DiskANNSectorSize
	sections.nodesLength = nodeLength
	if nodeLength > math.MaxInt64-sections.nodesOffset {
		return diskANNIndexSections{}, fmt.Errorf("%w: total length overflow", ErrInvalidDiskANNFile)
	}
	sections.totalLength = sections.nodesOffset + nodeLength
	return sections, nil
}

func readDiskANNIndexSection(ctx context.Context, reader io.ReaderAt, offset, length int64) ([]byte, error) {
	if length < 0 || length > int64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: section exceeds platform capacity", ErrInvalidDiskANNFile)
	}
	buffer := make([]byte, int(length))
	if len(buffer) != 0 {
		if err := readFullAt(ctx, reader, buffer, offset); err != nil {
			return nil, err
		}
	}
	return buffer, nil
}

func verifyDiskANNIndexSection(ctx context.Context, reader io.ReaderAt, offset, length int64, want uint32) error {
	const chunkSize = 1 << 20
	bufferSize := chunkSize
	if length < chunkSize {
		bufferSize = int(length)
	}
	buffer := make([]byte, bufferSize)
	var crc uint32
	for readOffset := int64(0); readOffset < length; {
		if err := ctx.Err(); err != nil {
			return err
		}
		readLength := min(len(buffer), int(length-readOffset))
		if err := readFullAt(ctx, reader, buffer[:readLength], offset+readOffset); err != nil {
			return err
		}
		crc = hashutil.UpdateCRC32C(crc, buffer[:readLength])
		readOffset += int64(readLength)
	}
	if crc != want {
		return fmt.Errorf("%w: got %08x, want %08x", errDiskANNIndexSectionChecksum, crc, want)
	}
	return nil
}

func validateDiskANNIndex(ctx context.Context, index *DiskANNIndex) error {
	if ctx == nil {
		return errors.New("core: nil DiskANN validation context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil || index.dimension <= 0 || index.dimension > MaxRotationDimension || index.nodes == nil {
		return errors.New("core: invalid DiskANN index")
	}
	if err := index.options.Validate(); err != nil {
		return err
	}
	count := len(index.keys)
	if count > math.MaxUint32 || len(index.positions) != count || index.nodes.layout.count != count ||
		index.nodes.layout.dimension != index.dimension || index.nodes.layout.metric != index.metric ||
		index.nodes.layout.maxDegree != index.options.MaxDegree {
		return errors.New("core: inconsistent DiskANN storage")
	}
	seen := make(map[uint64]struct{}, count)
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if _, duplicate := seen[key]; duplicate || index.positions[key] != position {
			return errors.New("core: invalid DiskANN key map")
		}
		seen[key] = struct{}{}
	}
	expectedTraversal := diskANNTraversalMetric(index.metric)
	if index.traversalMetric != expectedTraversal {
		return errors.New("core: invalid DiskANN traversal metric")
	}
	if count == 0 {
		if index.entryPoint != -1 || index.pq != nil || len(index.codes) != 0 || len(index.codeNorms) != 0 {
			return errors.New("core: invalid empty DiskANN state")
		}
		return nil
	}
	if index.entryPoint < 0 || index.entryPoint >= count || index.pq == nil || index.pq.dimension != index.dimension ||
		index.pq.metric != index.traversalMetric || uint64(count)*uint64(index.pq.Chunks()) > uint64(maxPlatformInt()) ||
		len(index.codes) != int(uint64(count)*uint64(index.pq.Chunks())) ||
		(index.options.PQChunks != 0 && index.options.PQChunks != index.pq.Chunks()) {
		return errors.New("core: inconsistent DiskANN PQ state")
	}
	if index.metric == MetricMIPSL2 {
		if len(index.codeNorms) != count {
			return errors.New("core: missing DiskANN MIPSL2 code norms")
		}
		for _, norm := range index.codeNorms {
			if norm < 0 || !finiteFloat32(norm) {
				return errors.New("core: invalid DiskANN MIPSL2 code norm")
			}
		}
	} else if len(index.codeNorms) != 0 {
		return errors.New("core: unexpected DiskANN code norms")
	}
	return nil
}
