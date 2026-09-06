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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/ailego/io"
	"github.com/gorse-io/xvec/pkg/rabitq"
)

const (
	ivfRaBitQFormatVersion       = 2
	ivfRaBitQHeaderSize          = 32
	ivfRaBitQReadChunkSize       = 64 << 10
	maxIVFRaBitQFileSize         = 1 << 30
	ivfRaBitQBruteForceThreshold = 1000
)

var (
	ErrInvalidIVFRaBitQOptions = errors.New("core: invalid IVF-RaBitQ build options")
	ErrInvalidIVFRaBitQFile    = errors.New("core: invalid IVF-RaBitQ file")
)

// IVFRaBitQBuildOptions configures IVF centroid training and RaBitQ encoding.
type IVFRaBitQBuildOptions struct {
	Metric        Metric
	NList         int
	TotalBits     int
	SampleCount   int
	NIterations   int
	MaxIterations int
	Workers       int
	Seed          uint64
}

// DefaultIVFRaBitQBuildOptions returns the zvec-compatible defaults.
func DefaultIVFRaBitQBuildOptions(metric Metric) IVFRaBitQBuildOptions {
	return IVFRaBitQBuildOptions{
		Metric: metric, NList: DefaultIVFNList, TotalBits: DefaultRaBitQTotalBits,
		NIterations: DefaultIVFNIterations, MaxIterations: DefaultKMeansIterations,
	}
}

func (o IVFRaBitQBuildOptions) Validate() error {
	if o.NList <= 0 {
		return fmt.Errorf("%w: NList must be positive", ErrInvalidIVFRaBitQOptions)
	}
	if o.SampleCount < 0 {
		return fmt.Errorf("%w: SampleCount cannot be negative", ErrInvalidIVFRaBitQOptions)
	}
	if o.NIterations <= 0 {
		return fmt.Errorf("%w: NIterations must be positive", ErrInvalidIVFRaBitQOptions)
	}
	rabitq := RaBitQOptions{
		Metric: o.Metric, TotalBits: o.TotalBits, Clusters: o.NList,
		SampleCount: o.SampleCount, MaxIterations: o.MaxIterations,
		Workers: o.Workers, Seed: o.Seed ^ 0x6976667261626974,
	}
	if err := rabitq.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidIVFRaBitQOptions, err)
	}
	return nil
}

func (o IVFRaBitQBuildOptions) ivfOptions() IVFBuildOptions {
	return IVFBuildOptions{
		Metric: o.Metric, NList: o.NList, NIterations: o.NIterations,
		Tolerance: DefaultKMeansTolerance, Workers: o.Workers, Seed: o.Seed,
	}
}

func (o IVFRaBitQBuildOptions) raBitQOptions() RaBitQOptions {
	return RaBitQOptions{
		Metric: o.Metric, TotalBits: o.TotalBits, Clusters: o.NList,
		SampleCount: o.SampleCount, MaxIterations: o.MaxIterations,
		Workers: o.Workers, Seed: o.Seed ^ 0x6976667261626974,
	}
}

// IVFRaBitQBuilder builds an IVF layout and FastScan-packed RaBitQ list data.
type IVFRaBitQBuilder struct {
	mu        sync.Mutex
	dimension int
	options   IVFRaBitQBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

func NewIVFRaBitQBuilder(dimension int, options IVFRaBitQBuildOptions) (*IVFRaBitQBuilder, error) {
	if dimension < MinRaBitQDimension || dimension > MaxRaBitQDimension {
		return nil, fmt.Errorf("%w: dimension must be in [%d,%d]", ErrInvalidIVFRaBitQOptions, MinRaBitQDimension, MaxRaBitQDimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &IVFRaBitQBuilder{dimension: dimension, options: options, positions: make(map[uint64]int)}, nil
}

func (b *IVFRaBitQBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil IVF-RaBitQ builder")
	}
	if ctx == nil {
		return errors.New("core: nil IVF-RaBitQ add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate IVF-RaBitQ vector: %w", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return ErrBuilderClosed
	}
	if _, exists := b.positions[key]; exists {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	b.positions[key] = len(b.keys)
	b.keys = append(b.keys, key)
	b.vectors = append(b.vectors, vector...)
	return nil
}

func (b *IVFRaBitQBuilder) Build(ctx context.Context) (*IVFRaBitQIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil IVF-RaBitQ builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF-RaBitQ build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	vectors := denseVectorViews(b.vectors, b.dimension)
	raBitQOptions := b.options.raBitQOptions()
	if len(vectors) > 0 && raBitQOptions.Clusters > len(vectors) {
		raBitQOptions.Clusters = len(vectors)
	}
	model, err := trainHNSWRaBitQModel(ctx, b.dimension, vectors, raBitQOptions)
	if err != nil {
		return nil, fmt.Errorf("core: train IVF-RaBitQ model: %w", err)
	}
	base, err := buildIVFBaseFromRaBitQ(ctx, b.dimension, b.options.ivfOptions(), b.keys, b.vectors, b.positions, model)
	if err != nil {
		return nil, err
	}
	listCodes, err := buildIVFRaBitQListCodes(ctx, base, model)
	if err != nil {
		return nil, fmt.Errorf("core: encode IVF-RaBitQ FastScan lists: %w", err)
	}
	index := &IVFRaBitQIndex{options: b.options, base: base, model: model, listCodes: listCodes}
	if err := validateIVFRaBitQIndex(ctx, index); err != nil {
		return nil, err
	}
	b.built = true
	b.keys, b.vectors, b.positions = nil, nil, nil
	return index, nil
}

// buildIVFBaseFromRaBitQ uses the RaBitQ reformer's coarse centroids for both
// list probing and residual-code membership. Training an independent IVF model
// would make list selection disagree with the centroid recorded in each code.
func buildIVFBaseFromRaBitQ(
	ctx context.Context,
	dimension int,
	options IVFBuildOptions,
	keys []uint64,
	vectors []float32,
	positions map[uint64]int,
	raBitQ *RaBitQModel,
) (*IVFIndex, error) {
	base := &IVFIndex{
		dimension: dimension, options: options, keys: slices.Clone(keys), vectors: slices.Clone(vectors),
		positions: cloneUint64Positions(positions),
	}
	if len(keys) == 0 {
		return base, nil
	}
	state := raBitQ.State()
	counts := make([]int, len(state.Centroids))
	base.model = &KMeansModel{
		metric: options.Metric, dimension: dimension, centroids: cloneVectors(state.Centroids),
		counts: counts, converged: true,
	}
	base.lists = make([]ivfList, len(state.Centroids))
	base.listForPosition = make([]int, len(keys))
	for position, vector := range denseVectorViews(base.vectors, dimension) {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		_, cluster, err := raBitQ.prepareAndCluster(vector)
		if err != nil {
			return nil, fmt.Errorf("core: assign IVF-RaBitQ vector %d: %w", position, err)
		}
		if cluster < 0 || cluster >= len(base.lists) {
			return nil, fmt.Errorf("core: build IVF-RaBitQ lists: code %d has cluster %d", position, cluster)
		}
		base.lists[cluster].positions = append(base.lists[cluster].positions, position)
		base.listForPosition[position] = cluster
		counts[cluster]++
	}
	if err := base.cacheCosineMagnitudes(ctx); err != nil {
		return nil, fmt.Errorf("core: cache IVF-RaBitQ cosine magnitudes: %w", err)
	}
	return base, nil
}

type ivfRaBitQListCodes struct {
	batchData []byte
	exData    []byte
}

func buildIVFRaBitQListCodes(ctx context.Context, base *IVFIndex, model *RaBitQModel) ([]ivfRaBitQListCodes, error) {
	listCount := model.Len()
	result := make([]ivfRaBitQListCodes, listCount)
	if len(base.keys) == 0 {
		return result, nil
	}
	batchBytes := rabitq.BatchDataBytes(model.paddedDimension)
	exBytes := rabitq.ExDataBytes(model.paddedDimension, model.extraBits)
	for cluster, list := range base.lists {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := len(list.positions)
		storage := &result[cluster]
		storage.batchData = make([]byte, (count+rabitq.BatchSize-1)/rabitq.BatchSize*batchBytes)
		storage.exData = make([]byte, count*exBytes)
		for offset := 0; offset < count; offset += rabitq.BatchSize {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			numPoints := min(rabitq.BatchSize, count-offset)
			rotated := make([]float32, numPoints*model.paddedDimension)
			for lane, position := range list.positions[offset : offset+numPoints] {
				start := position * base.dimension
				prepared, err := prepareRaBitQVector(base.vectors[start:start+base.dimension], model.dimension, model.metric)
				if err != nil {
					return nil, err
				}
				if err := model.rotator.Rotate(prepared, rotated[lane*model.paddedDimension:(lane+1)*model.paddedDimension]); err != nil {
					return nil, err
				}
			}
			batchID := offset / rabitq.BatchSize
			if err := rabitq.QuantizeSplitBatch(
				rotated, model.rotatedCentroids[cluster], numPoints, model.paddedDimension, model.extraBits,
				storage.batchData[batchID*batchBytes:(batchID+1)*batchBytes], storage.exData[offset*exBytes:(offset+numPoints)*exBytes],
				model.rabitqMetric(), rabitq.RaBitQConfig{TConst: model.extraScale},
			); err != nil {
				return nil, err
			}
		}
	}
	return result, nil
}

// IVFRaBitQIndex combines IVF candidate selection with RaBitQ scoring.
type IVFRaBitQIndex struct {
	mu        sync.RWMutex
	options   IVFRaBitQBuildOptions
	base      *IVFIndex
	model     *RaBitQModel
	listCodes []ivfRaBitQListCodes
}

func (i *IVFRaBitQIndex) Dimension() int {
	if i == nil || i.base == nil {
		return 0
	}
	return i.base.Dimension()
}
func (i *IVFRaBitQIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.options.Metric
}
func (i *IVFRaBitQIndex) Len() int {
	if i == nil || i.base == nil {
		return 0
	}
	return i.base.Len()
}
func (i *IVFRaBitQIndex) NList() int {
	if i == nil || i.base == nil {
		return 0
	}
	return i.base.NList()
}
func (i *IVFRaBitQIndex) Vector(key uint64) ([]float32, bool) {
	if i == nil || i.base == nil {
		return nil, false
	}
	return i.base.Vector(key)
}

// IVFRaBitQSearchOptions controls probed lists and optional linear code scan.
type IVFRaBitQSearchOptions struct {
	SearchOptions
	NProbe int
	Linear bool
}

func (o IVFRaBitQSearchOptions) Validate() error {
	if err := o.SearchOptions.Validate(); err != nil {
		return err
	}
	if o.NProbe <= 0 {
		return ErrInvalidIVFNProbe
	}
	return nil
}

func (i *IVFRaBitQIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.search(ctx, query, IVFRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: k}, NProbe: DefaultIVFNProbe}, false)
}
func (i *IVFRaBitQIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.search(ctx, query, IVFRaBitQSearchOptions{SearchOptions: options, NProbe: DefaultIVFNProbe}, true)
}
func (i *IVFRaBitQIndex) SearchIVFRaBitQ(ctx context.Context, query []float32, options IVFRaBitQSearchOptions) ([]Result, error) {
	return i.search(ctx, query, options, true)
}

// SearchIVFRaBitQGroups probes IVF lists, scores their RaBitQ codes, and keeps
// the best documents per resolved group.
func (i *IVFRaBitQIndex) SearchIVFRaBitQGroups(
	ctx context.Context,
	vector []float32,
	search IVFRaBitQSearchOptions,
	groups GroupByOptions,
) ([]GroupResult, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF-RaBitQ index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF-RaBitQ group-by context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if search.NProbe <= 0 {
		return nil, ErrInvalidIVFNProbe
	}
	if err := groups.Validate(); err != nil {
		return nil, err
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	query, err := i.model.prepareBatchQuery(vector)
	if err != nil {
		return nil, fmt.Errorf("core: prepare IVF-RaBitQ group-by query: %w", err)
	}
	lists, err := i.raBitQSearchLists(ctx, vector, search.NProbe, search.Linear)
	if err != nil {
		return nil, fmt.Errorf("core: probe IVF-RaBitQ group-by lists: %w", err)
	}
	accumulator := newGroupAccumulator(i.options.Metric, groups.TopKPerGroup)
	err = i.visitRaBitQLists(ctx, query, lists, func(lane ivfRaBitQLane) error {
		key := i.base.keys[lane.position]
		if groups.Filter != nil && !groups.Filter(key) {
			return nil
		}
		score, err := lane.fullDistance(query)
		if err != nil {
			return err
		}
		if i.options.Metric == MetricIP {
			score = 1 - score
		}
		if !scoreWithinRadius(i.options.Metric, score, groups.Radius) {
			return nil
		}
		value, ok := groups.Resolve(key)
		if ok {
			accumulator.add(value, Result{Key: key, Score: score})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return accumulator.finish(groups.GroupCount), nil
}

func (i *IVFRaBitQIndex) search(ctx context.Context, vector []float32, options IVFRaBitQSearchOptions, strict bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF-RaBitQ index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF-RaBitQ search context")
	}
	if options.NProbe <= 0 {
		return nil, ErrInvalidIVFNProbe
	}
	if strict {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else if options.TopK < 0 {
		return nil, errors.New("core: negative IVF-RaBitQ top-k")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateIVFRaBitQIndex(ctx, i); err != nil {
		return nil, err
	}
	query, err := i.model.prepareBatchQuery(vector)
	if err != nil {
		return nil, fmt.Errorf("core: prepare IVF-RaBitQ query: %w", err)
	}
	if options.TopK == 0 || len(i.base.keys) == 0 {
		return []Result{}, nil
	}
	lists, err := i.raBitQSearchLists(ctx, vector, options.NProbe, options.Linear)
	if err != nil {
		return nil, fmt.Errorf("core: probe IVF-RaBitQ lists: %w", err)
	}
	return i.scanLists(ctx, query, lists, options.SearchOptions)
}

func (i *IVFRaBitQIndex) raBitQSearchLists(ctx context.Context, vector []float32, nprobe int, linear bool) ([]int, error) {
	if linear || len(i.base.keys) <= ivfRaBitQBruteForceThreshold {
		lists := make([]int, len(i.base.lists))
		for list := range lists {
			lists[list] = list
		}
		return lists, nil
	}
	return i.base.ProbedLists(ctx, vector, nprobe)
}

type ivfRaBitQLane struct {
	position            int
	estimate, lower, ip float32
	exData              []byte
}

func (l ivfRaBitQLane) fullDistance(query *raBitQBatchQuery) (float32, error) {
	var distance float32
	if query.extraBits == 0 {
		distance = l.estimate
	} else {
		distance = rabitq.SplitDistanceBoosting(
			l.exData, query.ipFunc, query.batch, query.paddedDimension, query.extraBits, l.ip,
		)
	}
	if !finiteFloat32(distance) {
		return 0, fmt.Errorf("%w: non-finite IVF-RaBitQ distance", ErrInvalidRaBitQCode)
	}
	return distance, nil
}

func (i *IVFRaBitQIndex) visitRaBitQLists(
	ctx context.Context,
	query *raBitQBatchQuery,
	lists []int,
	visit func(ivfRaBitQLane) error,
) error {
	batchBytes := rabitq.BatchDataBytes(i.model.paddedDimension)
	exBytes := rabitq.ExDataBytes(i.model.paddedDimension, i.model.extraBits)
	estimates := make([]float32, rabitq.BatchSize)
	lowers := make([]float32, rabitq.BatchSize)
	ips := make([]float32, rabitq.BatchSize)
	for _, cluster := range lists {
		if cluster < 0 || cluster >= len(i.base.lists) || cluster >= len(i.listCodes) {
			return fmt.Errorf("%w: cluster %d out of range", ErrInvalidIVFRaBitQFile, cluster)
		}
		if err := query.prepareCluster(cluster); err != nil {
			return err
		}
		positions := i.base.lists[cluster].positions
		storage := i.listCodes[cluster]
		for offset := 0; offset < len(positions); offset += rabitq.BatchSize {
			if err := ctx.Err(); err != nil {
				return err
			}
			batchID := offset / rabitq.BatchSize
			rabitq.SplitBatchEstDist(
				storage.batchData[batchID*batchBytes:(batchID+1)*batchBytes], query.batch,
				i.model.paddedDimension, estimates, lowers, ips, true,
			)
			batchSize := min(rabitq.BatchSize, len(positions)-offset)
			for lane := 0; lane < batchSize; lane++ {
				if !finiteFloat32(estimates[lane]) || !finiteFloat32(lowers[lane]) || !finiteFloat32(ips[lane]) {
					return fmt.Errorf("%w: non-finite IVF-RaBitQ estimate", ErrInvalidRaBitQCode)
				}
				item := ivfRaBitQLane{
					position: positions[offset+lane], estimate: estimates[lane], lower: lowers[lane], ip: ips[lane],
				}
				if exBytes > 0 {
					exOffset := (offset + lane) * exBytes
					item.exData = storage.exData[exOffset : exOffset+exBytes]
				}
				if err := visit(item); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (i *IVFRaBitQIndex) scanLists(ctx context.Context, query *raBitQBatchQuery, lists []int, options SearchOptions) ([]Result, error) {
	better := func(left, right hnswScoredNode) bool {
		if left.score == right.score {
			return i.base.keys[left.position] < i.base.keys[right.position]
		}
		return left.score < right.score
	}
	worse := func(left, right hnswScoredNode) bool { return better(right, left) }
	selected := container.NewHeap(worse)
	err := i.visitRaBitQLists(ctx, query, lists, func(lane ivfRaBitQLane) error {
		key := i.base.keys[lane.position]
		if options.Filter != nil && !options.Filter(key) {
			return nil
		}
		if worstNode, ok := selected.Peek(); query.extraBits > 0 && selected.Len() >= options.TopK && ok && lane.lower >= worstNode.score {
			return nil
		}
		score, err := lane.fullDistance(query)
		if err != nil {
			return err
		}
		publicScore := score
		if i.options.Metric == MetricIP {
			publicScore = 1 - score
		}
		if !scoreWithinRadius(i.options.Metric, publicScore, options.Radius) {
			return nil
		}
		node := hnswScoredNode{position: lane.position, score: score}
		if selected.Len() < options.TopK {
			selected.Push(node)
		} else if worstNode, ok := selected.Peek(); ok && better(node, worstNode) {
			selected.Replace(node)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	nodes := selected.Values()
	slices.SortFunc(nodes, func(left, right hnswScoredNode) int {
		if better(left, right) {
			return -1
		}
		if better(right, left) {
			return 1
		}
		return 0
	})
	results := make([]Result, len(nodes))
	for index, node := range nodes {
		score := node.score
		if i.options.Metric == MetricIP {
			score = 1 - score
		}
		results[index] = Result{Key: i.base.keys[node.position], Score: score}
	}
	return results, nil
}

func validateIVFRaBitQIndex(ctx context.Context, index *IVFRaBitQIndex) error {
	if index == nil || index.base == nil || index.model == nil {
		return fmt.Errorf("%w: incomplete index", ErrInvalidIVFRaBitQFile)
	}
	if err := index.options.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidIVFRaBitQFile, err)
	}
	if err := validateIVFIndex(ctx, index.base); err != nil {
		return fmt.Errorf("%w: invalid IVF base: %v", ErrInvalidIVFRaBitQFile, err)
	}
	if index.model.metric != index.options.Metric || index.model.totalBits != index.options.TotalBits {
		return fmt.Errorf("%w: RaBitQ model options mismatch", ErrInvalidIVFRaBitQFile)
	}
	if index.base.dimension != index.model.dimension || len(index.listCodes) != index.model.Len() {
		return fmt.Errorf("%w: generation size mismatch", ErrInvalidIVFRaBitQFile)
	}
	if index.base.options != index.options.ivfOptions() {
		return fmt.Errorf("%w: IVF options mismatch", ErrInvalidIVFRaBitQFile)
	}
	if len(index.base.keys) != 0 && len(index.listCodes) != len(index.base.lists) {
		return fmt.Errorf("%w: list/code count mismatch", ErrInvalidIVFRaBitQFile)
	}
	batchBytes := rabitq.BatchDataBytes(index.model.paddedDimension)
	exBytes := rabitq.ExDataBytes(index.model.paddedDimension, index.model.extraBits)
	for cluster, storage := range index.listCodes {
		if cluster&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		count := 0
		if cluster < len(index.base.lists) {
			count = len(index.base.lists[cluster].positions)
		}
		if len(storage.batchData) != (count+rabitq.BatchSize-1)/rabitq.BatchSize*batchBytes || len(storage.exData) != count*exBytes {
			return fmt.Errorf("%w: invalid FastScan storage for list %d", ErrInvalidIVFRaBitQFile, cluster)
		}
		for offset := 0; offset < count; offset += rabitq.BatchSize {
			batchID := offset / rabitq.BatchSize
			batch := rabitq.NewBatchDataMap(storage.batchData[batchID*batchBytes:], index.model.paddedDimension)
			batchSize := min(rabitq.BatchSize, count-offset)
			for lane := 0; lane < batchSize; lane++ {
				if !validRaBitQFactors(batch.FAdd(lane), batch.FRescale(lane), batch.FError(lane)) || batch.FError(lane) < 0 {
					return fmt.Errorf("%w: invalid one-bit factors in list %d lane %d", ErrInvalidIVFRaBitQFile, cluster, offset+lane)
				}
				if exBytes > 0 {
					ex := rabitq.NewExDataMap(storage.exData[(offset+lane)*exBytes:], index.model.paddedDimension, index.model.extraBits)
					if !validRaBitQFactors(ex.FAddEx(), ex.FRescaleEx()) {
						return fmt.Errorf("%w: invalid extra-bit factors in list %d lane %d", ErrInvalidIVFRaBitQFile, cluster, offset+lane)
					}
				}
			}
		}
	}
	if len(index.base.keys) != 0 {
		state := index.model.State()
		if len(state.Centroids) != len(index.base.model.centroids) {
			return fmt.Errorf("%w: centroid count mismatch", ErrInvalidIVFRaBitQFile)
		}
		for cluster := range state.Centroids {
			if !slices.Equal(state.Centroids[cluster], index.base.model.centroids[cluster]) {
				return fmt.Errorf("%w: centroid %d mismatch", ErrInvalidIVFRaBitQFile, cluster)
			}
		}
	}
	return nil
}

func validRaBitQFactors(factors ...float32) bool {
	for _, factor := range factors {
		if !finiteFloat32(factor) {
			return false
		}
	}
	return true
}

type diskIVFRaBitQListCodes struct {
	BatchData []byte `json:"batch_data"`
	ExtraData []byte `json:"extra_data,omitempty"`
}
type diskIVFRaBitQ struct {
	Options   IVFRaBitQBuildOptions    `json:"options"`
	Base      []byte                   `json:"base"`
	Model     RaBitQModelState         `json:"model"`
	ListCodes []diskIVFRaBitQListCodes `json:"list_codes"`
}

func (i *IVFRaBitQIndex) Save(ctx context.Context, path string) error {
	if i == nil {
		return errors.New("core: nil IVF-RaBitQ index")
	}
	if ctx == nil {
		return errors.New("core: nil IVF-RaBitQ save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidIVFRaBitQFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := validateIVFRaBitQIndex(ctx, i); err != nil {
		return err
	}
	base, err := encodeIVFIndex(ctx, i.base)
	if err != nil {
		return fmt.Errorf("core: encode IVF-RaBitQ base: %w", err)
	}
	listCodes := make([]diskIVFRaBitQListCodes, len(i.listCodes))
	for cluster, codes := range i.listCodes {
		listCodes[cluster] = diskIVFRaBitQListCodes{
			BatchData: slices.Clone(codes.batchData), ExtraData: slices.Clone(codes.exData),
		}
	}
	payload, err := json.Marshal(diskIVFRaBitQ{Options: i.options, Base: base, Model: i.model.State(), ListCodes: listCodes})
	if err != nil {
		return fmt.Errorf("core: encode IVF-RaBitQ payload: %w", err)
	}
	if len(payload) > maxIVFRaBitQFileSize-ivfRaBitQHeaderSize {
		return fmt.Errorf("%w: file too large", ErrInvalidIVFRaBitQFile)
	}
	header := make([]byte, ivfRaBitQHeaderSize)
	copy(header[:8], "XIVFRBQ\x00")
	binary.LittleEndian.PutUint32(header[8:12], ivfRaBitQFormatVersion)
	binary.LittleEndian.PutUint64(header[12:20], uint64(len(payload)))
	binary.LittleEndian.PutUint32(header[20:24], hashutil.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[28:32], hashutil.CRC32C(header[:28]))
	encoded := append(header, payload...)
	if err := ioutil.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save IVF-RaBitQ file: %w", err)
	}
	return nil
}

type ivfRaBitQContextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r ivfRaBitQContextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

func readIVFRaBitQRecord(ctx context.Context, r io.Reader, size int64) ([]byte, error) {
	encoded := make([]byte, int(size))
	for offset := 0; offset < len(encoded); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+ivfRaBitQReadChunkSize, len(encoded))
		n, err := io.ReadFull(r, encoded[offset:end])
		offset += n
		if err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var extra [1]byte
	if n, err := r.Read(extra[:]); n != 0 || (err != nil && !errors.Is(err, io.EOF)) {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("%w: file changed while reading", ErrInvalidIVFRaBitQFile)
	}
	return encoded, nil
}

func decodeIVFRaBitQPayload(ctx context.Context, payload []byte) (diskIVFRaBitQ, error) {
	var disk diskIVFRaBitQ
	decoder := json.NewDecoder(ivfRaBitQContextReader{ctx: ctx, r: bytes.NewReader(payload)})
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&disk); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return diskIVFRaBitQ{}, ctxErr
		}
		return diskIVFRaBitQ{}, fmt.Errorf("%w: decode payload: %w", ErrInvalidIVFRaBitQFile, err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return diskIVFRaBitQ{}, ctxErr
		}
		return diskIVFRaBitQ{}, fmt.Errorf("%w: trailing payload data", ErrInvalidIVFRaBitQFile)
	}
	return disk, nil
}

func OpenIVFRaBitQIndex(ctx context.Context, path string) (*IVFRaBitQIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil IVF-RaBitQ open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("core: open IVF-RaBitQ file: %w", err)
	}
	defer func() {
		_ = file.Close()
	}()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("core: stat IVF-RaBitQ file: %w", err)
	}
	if info.Size() < ivfRaBitQHeaderSize || info.Size() > maxIVFRaBitQFileSize {
		return nil, fmt.Errorf("%w: invalid file size", ErrInvalidIVFRaBitQFile)
	}
	encoded, err := readIVFRaBitQRecord(ctx, file, info.Size())
	if err != nil {
		return nil, fmt.Errorf("core: read IVF-RaBitQ file: %w", err)
	}
	header, payload := encoded[:ivfRaBitQHeaderSize], encoded[ivfRaBitQHeaderSize:]
	if string(header[:8]) != "XIVFRBQ\x00" || binary.LittleEndian.Uint32(header[8:12]) != ivfRaBitQFormatVersion {
		return nil, fmt.Errorf("%w: magic or version mismatch", ErrInvalidIVFRaBitQFile)
	}
	if binary.LittleEndian.Uint64(header[12:20]) != uint64(len(payload)) || hashutil.CRC32C(payload) != binary.LittleEndian.Uint32(header[20:24]) || hashutil.CRC32C(header[:28]) != binary.LittleEndian.Uint32(header[28:32]) {
		return nil, fmt.Errorf("%w: length or checksum mismatch", ErrInvalidIVFRaBitQFile)
	}
	disk, err := decodeIVFRaBitQPayload(ctx, payload)
	if err != nil {
		return nil, err
	}
	base, err := decodeIVFIndex(ctx, disk.Base)
	if err != nil {
		return nil, fmt.Errorf("%w: decode base: %v", ErrInvalidIVFRaBitQFile, err)
	}
	model, err := RestoreRaBitQModel(disk.Model)
	if err != nil {
		return nil, fmt.Errorf("%w: restore model: %v", ErrInvalidIVFRaBitQFile, err)
	}
	listCodes := make([]ivfRaBitQListCodes, len(disk.ListCodes))
	for cluster, codes := range disk.ListCodes {
		listCodes[cluster] = ivfRaBitQListCodes{
			batchData: slices.Clone(codes.BatchData), exData: slices.Clone(codes.ExtraData),
		}
	}
	index := &IVFRaBitQIndex{options: disk.Options, base: base, model: model, listCodes: listCodes}
	if err := validateIVFRaBitQIndex(ctx, index); err != nil {
		return nil, err
	}
	return index, nil
}

var _ DenseProvider = (*IVFRaBitQIndex)(nil)
var _ DenseSearcher = (*IVFRaBitQIndex)(nil)
var _ DenseQuerySearcher = (*IVFRaBitQIndex)(nil)
