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
	"os"
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/ailego/io"
	"github.com/gorse-io/xvec/internal/ailego/math"
)

const (
	DefaultIVFNList       = 1024
	DefaultIVFNIterations = 10
)

var (
	ErrInvalidIVFOptions = errors.New("core: invalid IVF build options")
	ErrInvalidIVFList    = errors.New("core: invalid IVF list")
)

// IVFBuildOptions configures centroid training and deterministic list
// assignment. Quantization is layered onto the built layout separately.
type IVFBuildOptions struct {
	Metric      Metric
	NList       int
	NIterations int
	Tolerance   float64
	Workers     int
	Seed        uint64
}

// DefaultIVFBuildOptions returns the public baseline defaults.
func DefaultIVFBuildOptions(metric Metric) IVFBuildOptions {
	return IVFBuildOptions{
		Metric:      metric,
		NList:       DefaultIVFNList,
		NIterations: DefaultIVFNIterations,
		Tolerance:   DefaultKMeansTolerance,
	}
}

// Validate checks IVF build invariants.
func (o IVFBuildOptions) Validate() error {
	if !o.Metric.Valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidIVFOptions)
	}
	if o.NList <= 0 {
		return fmt.Errorf("%w: NList must be positive", ErrInvalidIVFOptions)
	}
	if o.NIterations <= 0 {
		return fmt.Errorf("%w: NIterations must be positive", ErrInvalidIVFOptions)
	}
	if o.Tolerance < 0 || math.IsNaN(o.Tolerance) || math.IsInf(o.Tolerance, 0) {
		return fmt.Errorf("%w: Tolerance must be finite and non-negative", ErrInvalidIVFOptions)
	}
	return nil
}

// IVFBuilder collects original vectors and builds a one-shot IVF layout. The
// resulting index supports concurrent search and incremental streaming.
type IVFBuilder struct {
	mu        sync.Mutex
	dimension int
	options   IVFBuildOptions
	keys      []uint64
	vectors   []float32
	positions map[uint64]int
	built     bool
}

// NewIVFBuilder constructs an empty IVF builder.
func NewIVFBuilder(dimension int, options IVFBuildOptions) (*IVFBuilder, error) {
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: got %d", ErrInvalidDimension, dimension)
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &IVFBuilder{
		dimension: dimension,
		options:   options,
		positions: make(map[uint64]int),
	}, nil
}

// Add clones one finite original vector. Keys remain unique for the builder's
// lifetime.
func (b *IVFBuilder) Add(ctx context.Context, key uint64, vector []float32) error {
	if b == nil {
		return errors.New("core: nil IVF builder")
	}
	if ctx == nil {
		return errors.New("core: nil IVF add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, b.dimension); err != nil {
		return fmt.Errorf("core: validate IVF vector: %w", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
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

// Build trains at most NList centroids, assigns each vector to its best
// centroid, and transfers builder-owned storage into an immutable index. An
// empty builder produces a valid empty layout without invoking k-means.
func (b *IVFBuilder) Build(ctx context.Context) (*IVFIndex, error) {
	if b == nil {
		return nil, errors.New("core: nil IVF builder")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF build context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.built {
		return nil, ErrBuilderClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	index := &IVFIndex{
		dimension: b.dimension,
		options:   b.options,
		keys:      b.keys,
		vectors:   b.vectors,
		positions: b.positions,
	}
	if len(b.keys) != 0 {
		training := make([][]float32, len(b.keys))
		for position := range b.keys {
			if position&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			start := position * b.dimension
			training[position] = b.vectors[start : start+b.dimension]
		}
		kmeans := DefaultKMeansOptions(b.options.NList, b.options.Metric)
		kmeans.MaxIterations = b.options.NIterations
		kmeans.Tolerance = b.options.Tolerance
		kmeans.Workers = b.options.Workers
		kmeans.Seed = b.options.Seed
		model, labels, err := trainKMeansWithAssignments(ctx, training, kmeans)
		if err != nil {
			return nil, fmt.Errorf("core: train IVF centroids: %w", err)
		}
		index.model = model
		index.lists = make([]ivfList, model.Len())
		index.listForPosition = make([]int, len(labels))
		for position, label := range labels {
			if position&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			index.lists[label].positions = append(index.lists[label].positions, position)
			index.listForPosition[position] = label
		}
		if err := index.cacheCosineMagnitudes(ctx); err != nil {
			return nil, err
		}
	}

	b.built = true
	b.keys = nil
	b.vectors = nil
	b.positions = nil
	return index, nil
}

type ivfList struct{ positions []int }

// IVFIndex is the streamable output of IVF construction. It retains original
// vectors for exact refinement and stores list membership by vector position.
type IVFIndex struct {
	mu                 sync.RWMutex
	dimension          int
	options            IVFBuildOptions
	model              *KMeansModel
	keys               []uint64
	vectors            []float32
	vectorMagnitudes   []float32
	centroidMagnitudes []float32
	positions          map[uint64]int
	lists              []ivfList
	listForPosition    []int
}

// Dimension returns the fixed vector dimension.
func (i *IVFIndex) Dimension() int {
	if i == nil {
		return 0
	}
	return i.dimension
}

// Metric returns the configured metric.
func (i *IVFIndex) Metric() Metric {
	if i == nil {
		return 0
	}
	return i.options.Metric
}

// Len returns the number of built vectors.
func (i *IVFIndex) Len() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.keys)
}

// NList returns the effective number of trained centroids. It is zero for an
// empty index and never exceeds the vector count or configured NList. Duplicate
// samples can still leave an empty assigned list.
func (i *IVFIndex) NList() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.lists)
}

// BuildOptions returns the value-semantic construction settings.
func (i *IVFIndex) BuildOptions() IVFBuildOptions {
	if i == nil {
		return IVFBuildOptions{}
	}
	return i.options
}

// Vector returns a cloned original vector by key.
func (i *IVFIndex) Vector(key uint64) ([]float32, bool) {
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

// Centroids returns a deep copy of trained or online-bootstrapped centroids.
func (i *IVFIndex) Centroids() [][]float32 {
	if i == nil {
		return [][]float32{}
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.model == nil {
		return [][]float32{}
	}
	return i.model.Centroids()
}

// TrainingCost returns the current list-assignment objective. Empty indexes
// return zero.
func (i *IVFIndex) TrainingCost() float64 {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.model == nil {
		return 0
	}
	return i.model.Cost()
}

// TrainingIterations returns completed k-means rounds.
func (i *IVFIndex) TrainingIterations() int {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if i.model == nil {
		return 0
	}
	return i.model.Iterations()
}

// TrainingConverged reports whether centroid training stopped on tolerance.
func (i *IVFIndex) TrainingConverged() bool {
	if i == nil {
		return false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.model != nil && i.model.converged
}

// List returns original candidate clones in stable builder insertion order.
func (i *IVFIndex) List(list int) ([]Candidate, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if list < 0 || list >= len(i.lists) {
		return nil, fmt.Errorf("%w: %d", ErrInvalidIVFList, list)
	}
	positions := i.lists[list].positions
	result := make([]Candidate, len(positions))
	for index, position := range positions {
		start := position * i.dimension
		result[index] = Candidate{
			Key:    i.keys[position],
			Vector: slices.Clone(i.vectors[start : start+i.dimension]),
		}
	}
	return result, nil
}

// ListForKey returns the current list containing key.
func (i *IVFIndex) ListForKey(key uint64) (int, bool) {
	if i == nil {
		return 0, false
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	position, found := i.positions[key]
	if !found || position >= len(i.listForPosition) {
		return 0, false
	}
	return i.listForPosition[position], true
}

var _ DenseProvider = (*IVFIndex)(nil)

const DefaultIVFNProbe = 10

var ErrInvalidIVFNProbe = errors.New("core: IVF NProbe must be positive")

// IVFSearchOptions combines common exact-result controls with the number of
// centroid lists to probe.
type IVFSearchOptions struct {
	SearchOptions
	NProbe int
}

// Validate checks top-k, radius, and probe-count invariants.
func (o IVFSearchOptions) Validate() error {
	if err := o.SearchOptions.Validate(); err != nil {
		return err
	}
	if o.NProbe <= 0 {
		return ErrInvalidIVFNProbe
	}
	return nil
}

// Search uses the baseline default NProbe. A zero top-k returns an empty
// result for consistency with the common DenseSearcher contract.
func (i *IVFIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.searchIVF(ctx, query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: k},
		NProbe:        DefaultIVFNProbe,
	}, false)
}

// SearchWithOptions applies a filter and exact-result radius while using the
// baseline default NProbe.
func (i *IVFIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.searchIVF(ctx, query, IVFSearchOptions{
		SearchOptions: options,
		NProbe:        DefaultIVFNProbe,
	}, true)
}

// SearchIVF probes the metric-best centroids and exact-scores originals in
// only those lists.
func (i *IVFIndex) SearchIVF(ctx context.Context, query []float32, options IVFSearchOptions) ([]Result, error) {
	return i.searchIVF(ctx, query, options, true)
}

func (i *IVFIndex) searchIVF(ctx context.Context, query []float32, options IVFSearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if options.NProbe <= 0 {
		return nil, ErrInvalidIVFNProbe
	}
	if requirePositiveTopK {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else {
		if options.TopK < 0 {
			return nil, errors.New("core: negative IVF top-k")
		}
		if options.Radius < 0 {
			return nil, ErrInvalidRadius
		}
	}
	if _, err := i.options.Metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate IVF query: %w", err)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if options.TopK == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}

	queryMagnitude, err := i.cosineQueryMagnitude(query)
	if err != nil {
		return nil, fmt.Errorf("core: prepare IVF query: %w", err)
	}
	lists, err := i.probedListsLocked(ctx, query, queryMagnitude, options.NProbe)
	if err != nil {
		return nil, err
	}
	batchLen := func(batch int) int { return len(i.lists[lists[batch]].positions) }
	if i.options.Metric == MetricCosine {
		var candidateMagnitude float32
		return topKPrevalidatedCandidateBatchesWithOptions(ctx, i.options.Metric, func(candidate, query []float32) (float32, error) {
			return mathutil.CosineDistanceWithMagnitudesPrevalidated(candidate, query, candidateMagnitude, queryMagnitude)
		}, query, options.SearchOptions, len(lists), batchLen, func(batch, index int) Candidate {
			position := i.lists[lists[batch]].positions[index]
			candidateMagnitude = i.vectorMagnitudes[position]
			start := position * i.dimension
			return Candidate{Key: i.keys[position], Vector: i.vectors[start : start+i.dimension]}
		})
	}
	distance, err := i.options.Metric.PrevalidatedDistance()
	if err != nil {
		return nil, err
	}
	return topKPrevalidatedCandidateBatchesWithOptions(ctx, i.options.Metric, distance, query, options.SearchOptions, len(lists), batchLen, func(batch, index int) Candidate {
		position := i.lists[lists[batch]].positions[index]
		start := position * i.dimension
		return Candidate{Key: i.keys[position], Vector: i.vectors[start : start+i.dimension]}
	})
}

// ProbedLists returns up to nprobe centroid indexes in metric-best order.
func (i *IVFIndex) ProbedLists(ctx context.Context, query []float32, nprobe int) ([]int, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF probe context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if nprobe <= 0 {
		return nil, ErrInvalidIVFNProbe
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if _, err := i.options.Metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate IVF probe query: %w", err)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	queryMagnitude, err := i.cosineQueryMagnitude(query)
	if err != nil {
		return nil, fmt.Errorf("core: prepare IVF probe query: %w", err)
	}
	return i.probedListsLocked(ctx, query, queryMagnitude, nprobe)
}

func (i *IVFIndex) probedListsLocked(ctx context.Context, query []float32, queryMagnitude float32, nprobe int) ([]int, error) {
	if len(i.lists) == 0 {
		return []int{}, nil
	}
	centroids := i.model.centroids
	if i.options.Metric == MetricCosine {
		count := min(nprobe, len(centroids))
		var centroidMagnitude float32
		results, err := topKPrevalidatedCandidatesWithOptions(ctx, i.options.Metric, func(centroid, query []float32) (float32, error) {
			return mathutil.CosineDistanceWithMagnitudesPrevalidated(centroid, query, centroidMagnitude, queryMagnitude)
		}, query, SearchOptions{TopK: count}, len(centroids), func(index int) Candidate {
			centroidMagnitude = i.centroidMagnitudes[index]
			return Candidate{Key: uint64(index), Vector: centroids[index]}
		})
		if err != nil {
			return nil, fmt.Errorf("core: select IVF centroids: %w", err)
		}
		lists := make([]int, len(results))
		for index, result := range results {
			lists[index] = int(result.Key)
		}
		return lists, nil
	}
	candidates := make([]Candidate, len(centroids))
	for index := range centroids {
		candidates[index] = Candidate{Key: uint64(index), Vector: centroids[index]}
	}
	count := min(nprobe, len(candidates))
	results, err := TopK(ctx, i.options.Metric, query, candidates, count)
	if err != nil {
		return nil, fmt.Errorf("core: select IVF centroids: %w", err)
	}
	lists := make([]int, len(results))
	for index, result := range results {
		lists[index] = int(result.Key)
	}
	return lists, nil
}

func (i *IVFIndex) cosineQueryMagnitude(query []float32) (float32, error) {
	if i.options.Metric != MetricCosine {
		return 0, nil
	}
	return mathutil.L2MagnitudePrevalidated(query)
}

func (i *IVFIndex) cacheCosineMagnitudes(ctx context.Context) error {
	if i.options.Metric != MetricCosine {
		return nil
	}
	i.vectorMagnitudes = make([]float32, len(i.keys))
	for position := range i.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		start := position * i.dimension
		magnitude, err := mathutil.L2MagnitudePrevalidated(i.vectors[start : start+i.dimension])
		if err != nil {
			return fmt.Errorf("core: cache IVF vector magnitude %d: %w", position, err)
		}
		i.vectorMagnitudes[position] = magnitude
	}
	i.centroidMagnitudes = make([]float32, len(i.model.centroids))
	for index, centroid := range i.model.centroids {
		magnitude, err := mathutil.L2MagnitudePrevalidated(centroid)
		if err != nil {
			return fmt.Errorf("core: cache IVF centroid magnitude %d: %w", index, err)
		}
		i.centroidMagnitudes[index] = magnitude
	}
	return nil
}

var _ DenseSearcher = (*IVFIndex)(nil)
var _ DenseQuerySearcher = (*IVFIndex)(nil)

var ErrIVFCapacity = errors.New("core: IVF index capacity exceeded")

// Add incrementally inserts one unique key and finite original vector. While
// the index contains fewer vectors than configured lists, each new vector
// extends the centroid set and starts its own list. Once NList is reached, the
// trained centroids remain fixed and additions enter their metric-best list.
func (i *IVFIndex) Add(ctx context.Context, key uint64, vector []float32) error {
	if i == nil {
		return errors.New("core: nil IVF index")
	}
	if ctx == nil {
		return errors.New("core: nil IVF add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, i.dimension); err != nil {
		return fmt.Errorf("core: validate incremental IVF vector: %w", err)
	}
	var vectorMagnitude float32
	if i.options.Metric == MetricCosine {
		var err error
		vectorMagnitude, err = mathutil.L2MagnitudePrevalidated(vector)
		if err != nil {
			return fmt.Errorf("core: prepare incremental IVF vector: %w", err)
		}
	}

	i.mu.Lock()
	defer i.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, exists := i.positions[key]; exists {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	if len(i.keys) == maxPlatformInt() || len(i.vectors) > maxPlatformInt()-i.dimension {
		return ErrIVFCapacity
	}

	list := 0
	var score float32
	var err error
	growCentroids := len(i.lists) < i.options.NList
	if growCentroids {
		list = len(i.lists)
		score, err = i.options.Metric.Compute(vector, vector)
	} else {
		if i.model == nil {
			return fmt.Errorf("%w: missing trained centroids", ErrInvalidIVFFile)
		}
		list, score, err = i.model.Nearest(vector)
	}
	if err != nil {
		return fmt.Errorf("core: assign incremental IVF vector: %w", err)
	}
	delta := float64(score)
	if i.options.Metric == MetricIP {
		delta = -delta
	}
	cost := delta
	if i.model != nil {
		cost += i.model.cost
	}
	if math.IsNaN(cost) || math.IsInf(cost, 0) {
		return fmt.Errorf("core: assign incremental IVF vector: non-finite objective")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	position := len(i.keys)
	i.positions[key] = position
	i.keys = append(i.keys, key)
	i.vectors = append(i.vectors, vector...)
	if i.options.Metric == MetricCosine {
		i.vectorMagnitudes = append(i.vectorMagnitudes, vectorMagnitude)
	}
	i.listForPosition = append(i.listForPosition, list)
	if i.model == nil {
		i.model = &KMeansModel{
			metric:     i.options.Metric,
			dimension:  i.dimension,
			centroids:  [][]float32{slices.Clone(vector)},
			counts:     []int{1},
			cost:       cost,
			iterations: 0,
			converged:  false,
		}
		i.lists = append(i.lists, ivfList{positions: []int{position}})
		if i.options.Metric == MetricCosine {
			i.centroidMagnitudes = append(i.centroidMagnitudes, vectorMagnitude)
		}
		return nil
	}
	if growCentroids {
		i.model.centroids = append(i.model.centroids, slices.Clone(vector))
		if i.options.Metric == MetricCosine {
			i.centroidMagnitudes = append(i.centroidMagnitudes, vectorMagnitude)
		}
		i.model.counts = append(i.model.counts, 1)
		i.lists = append(i.lists, ivfList{positions: []int{position}})
	} else {
		i.model.counts[list]++
		i.lists[list].positions = append(i.lists[list].positions, position)
	}
	i.model.cost = cost
	return nil
}

var (
	_ DenseStreamer = (*IVFIndex)(nil)
	_ DenseIndex    = (*IVFIndex)(nil)
)

const (
	ivfFileVersion    = 1
	ivfHeaderSize     = 112
	ivfReadChunk      = 1 << 20
	ivfRecordOverhead = 12 // key uint64 plus list uint32
)

var (
	ivfFileMagic = [8]byte{'Z', 'V', 'E', 'C', 'I', 'V', 'F', 0}

	// ErrInvalidIVFFile reports a structurally or semantically invalid native
	// Go IVF artifact.
	ErrInvalidIVFFile = errors.New("core: invalid IVF file")
	// ErrIVFChecksumMismatch distinguishes detected bit flips from other
	// format violations.
	ErrIVFChecksumMismatch = errors.New("core: IVF checksum mismatch")
	// ErrUnsupportedIVFVersion reports a well-identified artifact from an
	// unsupported native Go IVF format version.
	ErrUnsupportedIVFVersion = errors.New("core: unsupported IVF file version")
)

// Save durably publishes the immutable index as one checksummed native Go IVF
// file. Replacing an existing file is atomic to concurrent openers.
func (i *IVFIndex) Save(ctx context.Context, path string) error {
	if ctx == nil {
		return errors.New("core: nil IVF save context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidIVFFile)
	}
	snapshot, err := i.persistenceSnapshot(ctx)
	if err != nil {
		return err
	}
	encoded, err := encodeIVFIndex(ctx, snapshot)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFileAtomic(ctx, path, encoded, 0o600); err != nil {
		return fmt.Errorf("core: save IVF file: %w", err)
	}
	return nil
}

func (i *IVFIndex) persistenceSnapshot(ctx context.Context) (*IVFIndex, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidIVFFile)
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	snapshot := &IVFIndex{
		dimension:          i.dimension,
		options:            i.options,
		keys:               append([]uint64(nil), i.keys...),
		vectors:            append([]float32(nil), i.vectors...),
		vectorMagnitudes:   append([]float32(nil), i.vectorMagnitudes...),
		centroidMagnitudes: append([]float32(nil), i.centroidMagnitudes...),
		positions:          make(map[uint64]int, len(i.positions)),
		lists:              make([]ivfList, len(i.lists)),
		listForPosition:    append([]int(nil), i.listForPosition...),
	}
	for key, position := range i.positions {
		snapshot.positions[key] = position
	}
	for list := range i.lists {
		if list&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		snapshot.lists[list].positions = append([]int(nil), i.lists[list].positions...)
	}
	if i.model != nil {
		centroids, err := cloneVectorsContext(ctx, i.model.centroids)
		if err != nil {
			return nil, err
		}
		snapshot.model = &KMeansModel{
			metric:     i.model.metric,
			dimension:  i.model.dimension,
			centroids:  centroids,
			counts:     append([]int(nil), i.model.counts...),
			cost:       i.model.cost,
			iterations: i.model.iterations,
			converged:  i.model.converged,
		}
	}
	return snapshot, nil
}

// OpenIVFIndex reads and fully verifies a native Go IVF artifact. It never
// returns an index backed by the source file.
func OpenIVFIndex(ctx context.Context, path string) (*IVFIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil IVF open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, fmt.Errorf("%w: empty path", ErrInvalidIVFFile)
	}
	encoded, err := readIVFFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("core: read IVF file: %w", err)
	}
	index, err := decodeIVFIndex(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("core: open IVF file: %w", err)
	}
	return index, nil
}

func encodeIVFIndex(ctx context.Context, index *IVFIndex) ([]byte, error) {
	if index == nil {
		return nil, fmt.Errorf("%w: nil index", ErrInvalidIVFFile)
	}
	if err := validateIVFIndex(ctx, index); err != nil {
		return nil, err
	}
	count := len(index.keys)
	nlist := len(index.lists)
	payloadSize, err := checkedIVFPayloadSize(index.dimension, count, nlist)
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, payloadSize)
	if index.model != nil {
		for centroidIndex, centroid := range index.model.centroids {
			if centroidIndex&255 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			for _, value := range centroid {
				payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
			}
		}
	}
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		payload = binary.LittleEndian.AppendUint64(payload, key)
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
		payload = binary.LittleEndian.AppendUint32(payload, uint32(index.listForPosition[position]))
	}
	if len(payload) != payloadSize {
		return nil, fmt.Errorf("%w: internal payload length", ErrInvalidIVFFile)
	}

	header := make([]byte, ivfHeaderSize)
	copy(header[:8], ivfFileMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], ivfFileVersion)
	binary.LittleEndian.PutUint16(header[10:12], ivfHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], uint64(ivfHeaderSize+payloadSize))
	binary.LittleEndian.PutUint64(header[24:32], uint64(payloadSize))
	binary.LittleEndian.PutUint64(header[32:40], uint64(count))
	binary.LittleEndian.PutUint32(header[40:44], uint32(index.dimension))
	binary.LittleEndian.PutUint32(header[44:48], uint32(nlist))
	header[48] = byte(index.options.Metric)
	binary.LittleEndian.PutUint32(header[52:56], uint32(index.options.NList))
	binary.LittleEndian.PutUint32(header[56:60], uint32(index.options.NIterations))
	binary.LittleEndian.PutUint64(header[60:68], uint64(int64(index.options.Workers)))
	binary.LittleEndian.PutUint64(header[68:76], index.options.Seed)
	binary.LittleEndian.PutUint64(header[76:84], math.Float64bits(index.options.Tolerance))
	if index.model != nil {
		if index.model.converged {
			header[49] = 1
		}
		binary.LittleEndian.PutUint64(header[84:92], math.Float64bits(index.model.cost))
		binary.LittleEndian.PutUint32(header[92:96], uint32(index.model.iterations))
	}
	binary.LittleEndian.PutUint32(header[96:100], hashutil.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[108:112], hashutil.CRC32C(header[:108]))
	return append(header, payload...), nil
}

func decodeIVFIndex(ctx context.Context, encoded []byte) (*IVFIndex, error) {
	if ctx == nil {
		return nil, errors.New("core: nil IVF decode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < ivfHeaderSize {
		return nil, fmt.Errorf("%w: truncated header", ErrInvalidIVFFile)
	}
	header := encoded[:ivfHeaderSize]
	if !bytes.Equal(header[:8], ivfFileMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalidIVFFile)
	}
	version := binary.LittleEndian.Uint16(header[8:10])
	if version != ivfFileVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedIVFVersion, version)
	}
	if binary.LittleEndian.Uint16(header[10:12]) != ivfHeaderSize {
		return nil, fmt.Errorf("%w: bad header size", ErrInvalidIVFFile)
	}
	if binary.LittleEndian.Uint32(header[12:16]) != 0 ||
		binary.LittleEndian.Uint16(header[50:52]) != 0 ||
		binary.LittleEndian.Uint64(header[100:108]) != 0 {
		return nil, fmt.Errorf("%w: nonzero reserved field", ErrInvalidIVFFile)
	}
	if got, want := hashutil.CRC32C(header[:108]), binary.LittleEndian.Uint32(header[108:112]); got != want {
		return nil, fmt.Errorf("%w: header got %08x, want %08x", ErrIVFChecksumMismatch, got, want)
	}
	if binary.LittleEndian.Uint64(header[16:24]) != uint64(len(encoded)) ||
		binary.LittleEndian.Uint64(header[24:32]) != uint64(len(encoded)-ivfHeaderSize) {
		return nil, fmt.Errorf("%w: inconsistent file length", ErrInvalidIVFFile)
	}

	count64 := binary.LittleEndian.Uint64(header[32:40])
	dimension64 := uint64(binary.LittleEndian.Uint32(header[40:44]))
	nlist64 := uint64(binary.LittleEndian.Uint32(header[44:48]))
	if dimension64 == 0 || dimension64 > MaxRotationDimension {
		return nil, fmt.Errorf("%w: invalid dimension %d", ErrInvalidIVFFile, dimension64)
	}
	if count64 > uint64(maxPlatformInt()) || nlist64 > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: count exceeds platform capacity", ErrInvalidIVFFile)
	}
	if (count64 == 0 && nlist64 != 0) || (count64 != 0 && (nlist64 == 0 || nlist64 > count64)) {
		return nil, fmt.Errorf("%w: invalid effective list count", ErrInvalidIVFFile)
	}
	dimension, count, nlist := int(dimension64), int(count64), int(nlist64)
	payloadSize, err := checkedIVFPayloadSize(dimension, count, nlist)
	if err != nil || payloadSize != len(encoded)-ivfHeaderSize {
		return nil, fmt.Errorf("%w: invalid payload length", ErrInvalidIVFFile)
	}
	payload := encoded[ivfHeaderSize:]
	if got, want := hashutil.CRC32C(payload), binary.LittleEndian.Uint32(header[96:100]); got != want {
		return nil, fmt.Errorf("%w: payload got %08x, want %08x", ErrIVFChecksumMismatch, got, want)
	}

	options, err := decodeIVFOptions(header)
	if err != nil {
		return nil, err
	}
	trainingCost := math.Float64frombits(binary.LittleEndian.Uint64(header[84:92]))
	trainingIterations := uint64(binary.LittleEndian.Uint32(header[92:96]))
	converged := header[49]
	if converged > 1 || math.IsNaN(trainingCost) || math.IsInf(trainingCost, 0) || trainingIterations > uint64(options.NIterations) {
		return nil, fmt.Errorf("%w: invalid training metadata", ErrInvalidIVFFile)
	}
	if count == 0 && (trainingCost != 0 || trainingIterations != 0 || converged != 0) {
		return nil, fmt.Errorf("%w: empty index has training metadata", ErrInvalidIVFFile)
	}

	index := &IVFIndex{
		dimension:       dimension,
		options:         options,
		keys:            make([]uint64, count),
		vectors:         make([]float32, count*dimension),
		positions:       make(map[uint64]int, count),
		lists:           make([]ivfList, nlist),
		listForPosition: make([]int, count),
	}
	offset := 0
	centroids := make([][]float32, nlist)
	for list := 0; list < nlist; list++ {
		if list&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		centroid := make([]float32, dimension)
		for component := range centroid {
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: non-finite centroid", ErrInvalidIVFFile)
			}
			centroid[component] = value
		}
		centroids[list] = centroid
	}
	counts := make([]int, nlist)
	for position := 0; position < count; position++ {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := binary.LittleEndian.Uint64(payload[offset : offset+8])
		offset += 8
		if _, duplicate := index.positions[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate key %d", ErrInvalidIVFFile, key)
		}
		index.keys[position] = key
		index.positions[key] = position
		start := position * dimension
		for component := 0; component < dimension; component++ {
			value := math.Float32frombits(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			if !finiteFloat32(value) {
				return nil, fmt.Errorf("%w: non-finite vector", ErrInvalidIVFFile)
			}
			index.vectors[start+component] = value
		}
		list := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		offset += 4
		if list >= nlist64 {
			return nil, fmt.Errorf("%w: vector list %d out of range", ErrInvalidIVFFile, list)
		}
		index.listForPosition[position] = int(list)
		index.lists[list].positions = append(index.lists[list].positions, position)
		counts[list]++
	}
	if offset != len(payload) {
		return nil, fmt.Errorf("%w: unconsumed payload", ErrInvalidIVFFile)
	}
	if count != 0 {
		index.model = &KMeansModel{
			metric:     options.Metric,
			dimension:  dimension,
			centroids:  centroids,
			counts:     counts,
			cost:       trainingCost,
			iterations: int(trainingIterations),
			converged:  converged == 1,
		}
		if err := index.cacheCosineMagnitudes(ctx); err != nil {
			return nil, err
		}
	}
	return index, nil
}

func decodeIVFOptions(header []byte) (IVFBuildOptions, error) {
	workers64 := int64(binary.LittleEndian.Uint64(header[60:68]))
	if int64(int(workers64)) != workers64 {
		return IVFBuildOptions{}, fmt.Errorf("%w: workers exceed platform capacity", ErrInvalidIVFFile)
	}
	options := IVFBuildOptions{
		Metric:      Metric(header[48]),
		NList:       int(binary.LittleEndian.Uint32(header[52:56])),
		NIterations: int(binary.LittleEndian.Uint32(header[56:60])),
		Workers:     int(workers64),
		Seed:        binary.LittleEndian.Uint64(header[68:76]),
		Tolerance:   math.Float64frombits(binary.LittleEndian.Uint64(header[76:84])),
	}
	if err := options.Validate(); err != nil {
		return IVFBuildOptions{}, fmt.Errorf("%w: %v", ErrInvalidIVFFile, err)
	}
	return options, nil
}

func validateIVFIndex(ctx context.Context, index *IVFIndex) error {
	if index.dimension <= 0 || index.dimension > MaxRotationDimension {
		return fmt.Errorf("%w: invalid dimension", ErrInvalidIVFFile)
	}
	if err := index.options.Validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidIVFFile, err)
	}
	if index.options.NList > math.MaxUint32 || index.options.NIterations > math.MaxUint32 {
		return fmt.Errorf("%w: options exceed format capacity", ErrInvalidIVFFile)
	}
	count := len(index.keys)
	if _, err := checkedIVFPayloadSize(index.dimension, count, len(index.lists)); err != nil {
		return err
	}
	if count > maxPlatformInt()/index.dimension || len(index.vectors) != count*index.dimension || len(index.positions) != count {
		return fmt.Errorf("%w: inconsistent vector storage", ErrInvalidIVFFile)
	}
	if index.options.Metric == MetricCosine {
		if len(index.vectorMagnitudes) != count || len(index.centroidMagnitudes) != len(index.lists) {
			return fmt.Errorf("%w: inconsistent cosine magnitude cache", ErrInvalidIVFFile)
		}
	} else if len(index.vectorMagnitudes) != 0 || len(index.centroidMagnitudes) != 0 {
		return fmt.Errorf("%w: unexpected magnitude cache", ErrInvalidIVFFile)
	}
	if count == 0 {
		if index.model != nil || len(index.lists) != 0 || len(index.listForPosition) != 0 {
			return fmt.Errorf("%w: inconsistent empty index", ErrInvalidIVFFile)
		}
		return nil
	}
	if index.model == nil || index.model.metric != index.options.Metric || index.model.dimension != index.dimension ||
		len(index.model.centroids) != len(index.lists) || len(index.model.counts) != len(index.lists) ||
		len(index.listForPosition) != count || len(index.lists) == 0 || len(index.lists) > count ||
		index.model.iterations < 0 || index.model.iterations > index.options.NIterations ||
		math.IsNaN(index.model.cost) || math.IsInf(index.model.cost, 0) {
		return fmt.Errorf("%w: inconsistent trained index", ErrInvalidIVFFile)
	}
	seen := make([]bool, count)
	counts := make([]int, len(index.lists))
	for list, centroid := range index.model.centroids {
		if len(centroid) != index.dimension {
			return fmt.Errorf("%w: invalid centroid dimension", ErrInvalidIVFFile)
		}
		for _, value := range centroid {
			if !finiteFloat32(value) {
				return fmt.Errorf("%w: non-finite centroid", ErrInvalidIVFFile)
			}
		}
		for _, position := range index.lists[list].positions {
			if position < 0 || position >= count || seen[position] || index.listForPosition[position] != list {
				return fmt.Errorf("%w: invalid list membership", ErrInvalidIVFFile)
			}
			seen[position] = true
			counts[list]++
		}
		if counts[list] != index.model.counts[list] {
			return fmt.Errorf("%w: inconsistent centroid count", ErrInvalidIVFFile)
		}
	}
	for position, key := range index.keys {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if !seen[position] {
			return fmt.Errorf("%w: incomplete list coverage", ErrInvalidIVFFile)
		}
		if mapped, ok := index.positions[key]; !ok || mapped != position {
			return fmt.Errorf("%w: inconsistent key map", ErrInvalidIVFFile)
		}
		start := position * index.dimension
		for _, value := range index.vectors[start : start+index.dimension] {
			if !finiteFloat32(value) {
				return fmt.Errorf("%w: non-finite vector", ErrInvalidIVFFile)
			}
		}
	}
	return nil
}

func checkedIVFPayloadSize(dimension, count, nlist int) (int, error) {
	if dimension <= 0 || count < 0 || nlist < 0 {
		return 0, fmt.Errorf("%w: invalid size", ErrInvalidIVFFile)
	}
	dim := uint64(dimension)
	centroidBytes := uint64(nlist) * dim * 4
	recordBytes := uint64(count) * (ivfRecordOverhead + dim*4)
	if nlist != 0 && centroidBytes/dim/4 != uint64(nlist) ||
		count != 0 && recordBytes/(ivfRecordOverhead+dim*4) != uint64(count) ||
		centroidBytes > math.MaxUint64-recordBytes {
		return 0, fmt.Errorf("%w: payload size overflow", ErrInvalidIVFFile)
	}
	total := centroidBytes + recordBytes
	if total > uint64(maxPlatformInt()-ivfHeaderSize) {
		return 0, fmt.Errorf("%w: payload exceeds platform capacity", ErrInvalidIVFFile)
	}
	return int(total), nil
}

func readIVFFile(ctx context.Context, path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < 0 || uint64(info.Size()) > uint64(maxPlatformInt()) {
		return nil, fmt.Errorf("%w: file exceeds platform capacity", ErrInvalidIVFFile)
	}
	encoded := make([]byte, int(info.Size()))
	for offset := 0; offset < len(encoded); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+ivfReadChunk, len(encoded))
		if _, err := io.ReadFull(file, encoded[offset:end]); err != nil {
			return nil, err
		}
		offset = end
	}
	return encoded, nil
}

func maxPlatformInt() int { return int(^uint(0) >> 1) }
