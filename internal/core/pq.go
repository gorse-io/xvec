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
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
)

const (
	PQBits                   = 8
	PQCentroidCount          = 1 << PQBits
	DefaultPQMaxTrainSamples = 200_000
	DefaultPQIterations      = 12
	DefaultPQKMC2ChainLength = 32
)

var (
	ErrInvalidPQOptions    = errors.New("core: invalid PQ options")
	ErrInvalidPQModel      = errors.New("core: invalid PQ model")
	ErrInvalidPQCode       = errors.New("core: invalid PQ code")
	ErrPQModelMismatch     = errors.New("core: PQ code belongs to a different model")
	ErrPQUnsupportedMetric = errors.New("core: PQ supports L2 and inner product only")
	ErrPQScoreOverflow     = errors.New("core: PQ score overflows float32")
)

// PQOptions configures deterministic 8-bit product-quantizer training.
// Chunks zero resolves to half the vector dimension, matching DiskANN's
// public auto setting. Training uses the first MaxTrainSamples vectors.
type PQOptions struct {
	Metric          Metric
	Chunks          int
	MaxTrainSamples int
	MaxIterations   int
	Workers         int
	Seed            uint64
}

// DefaultPQOptions returns the pinned 256-centroid, 12-iteration defaults.
func DefaultPQOptions(metric Metric) PQOptions {
	return PQOptions{
		Metric: metric, MaxTrainSamples: DefaultPQMaxTrainSamples,
		MaxIterations: DefaultPQIterations,
	}
}

// Validate checks options that do not depend on vector dimension.
func (o PQOptions) Validate() error {
	if o.Metric != MetricL2 && o.Metric != MetricIP {
		return fmt.Errorf("%w: %w", ErrInvalidPQOptions, ErrPQUnsupportedMetric)
	}
	if o.Chunks < 0 {
		return fmt.Errorf("%w: Chunks cannot be negative", ErrInvalidPQOptions)
	}
	if o.MaxTrainSamples <= 0 {
		return fmt.Errorf("%w: MaxTrainSamples must be positive", ErrInvalidPQOptions)
	}
	if o.MaxIterations <= 0 {
		return fmt.Errorf("%w: MaxIterations must be positive", ErrInvalidPQOptions)
	}
	if o.Workers < 0 {
		return fmt.Errorf("%w: Workers cannot be negative", ErrInvalidPQOptions)
	}
	return nil
}

// PQModelState is the complete portable state of a trained quantizer. Pivots
// are centroid-major: row c contains all dimensions for centroid ID c, while
// ChunkOffsets determines which portion of each row belongs to each chunk.
type PQModelState struct {
	Dimension    int
	Metric       Metric
	ChunkOffsets []int
	Pivots       []float32
}

// PQModel is an immutable 8-bit product quantizer.
type PQModel struct {
	dimension    int
	metric       Metric
	distance     mathutil.DenseDistance
	chunkOffsets []int
	pivots       []float32
	fingerprint  uint64
}

// TrainPQ partitions dimensions into contiguous chunks and trains one
// independent 256-entry codebook per chunk. When fewer than 256 samples are
// available, unused rows repeat centroid zero and therefore never win a tie.
func TrainPQ(ctx context.Context, vectors [][]float32, options PQOptions) (*PQModel, error) {
	if ctx == nil {
		return nil, errors.New("core: nil PQ training context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, ErrEmptyTrainingSet
	}
	dimension := len(vectors[0])
	if dimension <= 0 || dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: dimension must be in [1,%d]", ErrInvalidPQOptions, MaxRotationDimension)
	}
	if err := validateTrainingVectors(ctx, vectors, dimension, true); err != nil {
		return nil, err
	}
	chunks := options.Chunks
	if chunks == 0 {
		chunks = dimension / 2
	}
	if chunks <= 0 || chunks > dimension {
		return nil, fmt.Errorf("%w: resolved Chunks must be in [1,%d]", ErrInvalidPQOptions, dimension)
	}
	offsets := pqChunkOffsets(dimension, chunks)
	sampleCount := min(len(vectors), options.MaxTrainSamples)
	pivots := make([]float32, PQCentroidCount*dimension)
	trainingContext := ctx
	err := parallel.ParallelFor(ctx, chunks, options.Workers, func(workerContext context.Context, chunk int) error {
		if err := workerContext.Err(); err != nil {
			return err
		}
		if err := trainingContext.Err(); err != nil {
			return err
		}
		start, end := offsets[chunk], offsets[chunk+1]
		if options.Metric == MetricL2 {
			centroids, err := trainPQL2Chunk(
				workerContext, vectors, sampleCount, start, end,
				min(PQCentroidCount, sampleCount), options.MaxIterations,
				options.Seed^(0x70716b6d00000000+uint64(chunk)),
			)
			if err != nil {
				return fmt.Errorf("core: train PQ chunk %d: %w", chunk, err)
			}
			width := end - start
			centroidCount := len(centroids) / width
			for centroid := 0; centroid < PQCentroidCount; centroid++ {
				source := 0
				if centroid < centroidCount {
					source = centroid * width
				}
				copy(pivots[centroid*dimension+start:centroid*dimension+end], centroids[source:source+width])
			}
			return nil
		}
		training := make([][]float32, sampleCount)
		for sample := range training {
			if sample&1023 == 0 {
				if err := trainingContext.Err(); err != nil {
					return err
				}
			}
			training[sample] = slices.Clone(vectors[sample][start:end])
		}
		kmeans := DefaultKMeansOptions(PQCentroidCount, options.Metric)
		kmeans.MaxIterations = options.MaxIterations
		kmeans.Workers = 1
		kmeans.Seed = options.Seed ^ (0x70716b6d00000000 + uint64(chunk))
		initialCentroids, err := initializePQKMC2(
			workerContext, training, min(PQCentroidCount, len(training)), options.Metric, kmeans.Seed,
		)
		if err != nil {
			return fmt.Errorf("core: initialize PQ chunk %d: %w", chunk, err)
		}
		kmeans.InitialCentroids = initialCentroids
		model, err := TrainKMeans(workerContext, training, kmeans)
		if err != nil {
			return fmt.Errorf("core: train PQ chunk %d: %w", chunk, err)
		}
		centroids := model.Centroids()
		for centroid := 0; centroid < PQCentroidCount; centroid++ {
			source := centroids[0]
			if centroid < len(centroids) {
				source = centroids[centroid]
			}
			copy(pivots[centroid*dimension+start:centroid*dimension+end], source)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return RestorePQModel(PQModelState{
		Dimension: dimension, Metric: options.Metric,
		ChunkOffsets: offsets, Pivots: pivots,
	})
}

// trainPQL2Chunk mirrors zvec's NumericalKmeans layout: one compact feature
// matrix per chunk, one worker per chunk, and direct squared-L2 scoring. It
// avoids millions of tiny subvector slices and generic distance dispatches.
func trainPQL2Chunk(
	ctx context.Context,
	vectors [][]float32,
	sampleCount, start, end, clusters, maxIterations int,
	seed uint64,
) ([]float32, error) {
	width := end - start
	training := make([]float32, sampleCount*width)
	for sample := 0; sample < sampleCount; sample++ {
		if sample&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		copy(training[sample*width:(sample+1)*width], vectors[sample][start:end])
	}
	centroids, err := initializePQL2KMC2(ctx, training, sampleCount, width, clusters, seed)
	if err != nil {
		return nil, err
	}
	scores := make([]float32, sampleCount)
	counts := make([]int, clusters)
	sums := make([]float64, clusters*width)
	next := make([]float32, len(centroids))
	empty := make([]int, 0, clusters)
	used := make([]bool, sampleCount)
	previousCost := float64(0)
	for iteration := 0; iteration < maxIterations; iteration++ {
		clear(counts)
		clear(sums)
		cost := float64(0)
		for sample := 0; sample < sampleCount; sample++ {
			if sample&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			vector := training[sample*width : (sample+1)*width]
			bestLabel := 0
			bestScore := pqL2SquaredSmall(vector, centroids[:width])
			for centroid := 1; centroid < clusters; centroid++ {
				offset := centroid * width
				score := pqL2SquaredSmall(vector, centroids[offset:offset+width])
				if score < bestScore {
					bestLabel, bestScore = centroid, score
				}
			}
			scores[sample] = bestScore
			counts[bestLabel]++
			sum := sums[bestLabel*width : (bestLabel+1)*width]
			for coordinate, value := range vector {
				sum[coordinate] += float64(value)
			}
			cost += float64(bestScore)
		}

		copy(next, centroids)
		empty = empty[:0]
		for centroid, count := range counts {
			if count == 0 {
				empty = append(empty, centroid)
				continue
			}
			sum := sums[centroid*width : (centroid+1)*width]
			destination := next[centroid*width : (centroid+1)*width]
			for coordinate := range destination {
				destination[coordinate] = float32(sum[coordinate] / float64(count))
			}
		}
		changed := len(empty) != 0
		if changed {
			clear(used)
			for _, centroid := range empty {
				selected := -1
				for sample, score := range scores {
					if sample&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					if !used[sample] && (selected < 0 || scores[selected] < score) {
						selected = sample
					}
				}
				if selected < 0 {
					break
				}
				used[selected] = true
				copy(next[centroid*width:(centroid+1)*width], training[selected*width:(selected+1)*width])
			}
		}
		centroids, next = next, centroids
		if !changed && math.Abs(cost-previousCost) < DefaultKMeansTolerance {
			break
		}
		previousCost = cost
	}
	return centroids, nil
}

func initializePQL2KMC2(
	ctx context.Context,
	training []float32,
	sampleCount, width, clusters int,
	seed uint64,
) ([]float32, error) {
	random := splitMix64{state: seed}
	indices, err := pqUniformSampleIndicesInto(ctx, nil, sampleCount, 1, &random)
	if err != nil {
		return nil, err
	}
	centroids := make([]float32, clusters*width)
	copy(centroids[:width], training[indices[0]*width:(indices[0]+1)*width])
	scores := make([]float32, min(DefaultPQKMC2ChainLength, sampleCount))
	for selectedCount := 1; selectedCount < clusters; selectedCount++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		indices, err = pqUniformSampleIndicesInto(
			ctx, indices[:0], sampleCount, min(DefaultPQKMC2ChainLength, sampleCount), &random,
		)
		if err != nil {
			return nil, err
		}
		for candidate, sample := range indices {
			vector := training[sample*width : (sample+1)*width]
			best := float32(math.MaxFloat32)
			for centroid := 0; centroid < selectedCount; centroid++ {
				offset := centroid * width
				score := pqL2SquaredSmall(vector, centroids[offset:offset+width])
				best = min(best, score)
			}
			scores[candidate] = best
		}
		selectedScore, selected := scores[0], 0
		for candidate := 1; candidate < len(indices); candidate++ {
			if selectedScore == 0 || selectedScore*float32(random.float64()) < scores[candidate] {
				selectedScore, selected = scores[candidate], candidate
			}
		}
		copy(
			centroids[selectedCount*width:(selectedCount+1)*width],
			training[indices[selected]*width:(indices[selected]+1)*width],
		)
	}
	return centroids, nil
}

func pqL2SquaredSmall(left, right []float32) float32 {
	switch len(left) {
	case 1:
		difference := left[0] - right[0]
		return difference * difference
	case 2:
		first := left[0] - right[0]
		second := left[1] - right[1]
		return first*first + second*second
	default:
		var score float32
		for coordinate, value := range left {
			difference := value - right[coordinate]
			score += difference * difference
		}
		return score
	}
}

// RestorePQModel validates and clones a complete portable model snapshot.
func RestorePQModel(state PQModelState) (*PQModel, error) {
	if state.Dimension <= 0 || state.Dimension > MaxRotationDimension {
		return nil, fmt.Errorf("%w: dimension must be in [1,%d]", ErrInvalidPQModel, MaxRotationDimension)
	}
	if state.Metric != MetricL2 && state.Metric != MetricIP {
		return nil, fmt.Errorf("%w: %w", ErrInvalidPQModel, ErrPQUnsupportedMetric)
	}
	if len(state.ChunkOffsets) < 2 || len(state.ChunkOffsets)-1 > state.Dimension ||
		state.ChunkOffsets[0] != 0 || state.ChunkOffsets[len(state.ChunkOffsets)-1] != state.Dimension {
		return nil, fmt.Errorf("%w: invalid chunk offsets", ErrInvalidPQModel)
	}
	for index := 1; index < len(state.ChunkOffsets); index++ {
		if state.ChunkOffsets[index] <= state.ChunkOffsets[index-1] {
			return nil, fmt.Errorf("%w: chunk offsets must be strictly increasing", ErrInvalidPQModel)
		}
	}
	if len(state.Pivots) != PQCentroidCount*state.Dimension {
		return nil, fmt.Errorf("%w: got %d pivots, want %d", ErrInvalidPQModel, len(state.Pivots), PQCentroidCount*state.Dimension)
	}
	for _, value := range state.Pivots {
		if !finiteFloat32(value) {
			return nil, fmt.Errorf("%w: non-finite pivot", ErrInvalidPQModel)
		}
	}
	distance, err := state.Metric.PrevalidatedDistance()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidPQModel, err)
	}
	model := &PQModel{
		dimension: state.Dimension, metric: state.Metric,
		distance:     distance,
		chunkOffsets: slices.Clone(state.ChunkOffsets), pivots: slices.Clone(state.Pivots),
	}
	model.fingerprint = fingerprintPQModel(model)
	return model, nil
}

// State returns an independent complete model snapshot.
func (m *PQModel) State() PQModelState {
	if m == nil {
		return PQModelState{}
	}
	return PQModelState{
		Dimension: m.dimension, Metric: m.metric,
		ChunkOffsets: slices.Clone(m.chunkOffsets), Pivots: slices.Clone(m.pivots),
	}
}

func (m *PQModel) Dimension() int {
	if m == nil {
		return 0
	}
	return m.dimension
}

func (m *PQModel) Metric() Metric {
	if m == nil {
		return 0
	}
	return m.metric
}

func (m *PQModel) Chunks() int {
	if m == nil {
		return 0
	}
	return len(m.chunkOffsets) - 1
}

func (m *PQModel) ChunkOffsets() []int {
	if m == nil {
		return nil
	}
	return slices.Clone(m.chunkOffsets)
}

// Pivots returns the full centroid-major pivot matrix.
func (m *PQModel) Pivots() []float32 {
	if m == nil {
		return nil
	}
	return slices.Clone(m.pivots)
}

// Code restores one immutable code owned by this model.
func (m *PQModel) Code(encoded []byte) (PQCode, error) {
	if err := m.validate(); err != nil {
		return PQCode{}, err
	}
	if len(encoded) != m.Chunks() {
		return PQCode{}, fmt.Errorf("%w: got %d bytes, want %d", ErrInvalidPQCode, len(encoded), m.Chunks())
	}
	return PQCode{modelFingerprint: m.fingerprint, codes: slices.Clone(encoded)}, nil
}

// Encode selects the best centroid independently in every chunk.
func (m *PQModel) Encode(vector []float32) (PQCode, error) {
	if err := m.validate(); err != nil {
		return PQCode{}, err
	}
	if err := validateTrainingVector(vector, m.dimension); err != nil {
		return PQCode{}, err
	}
	codes := make([]byte, m.Chunks())
	for chunk := range codes {
		start, end := m.chunkOffsets[chunk], m.chunkOffsets[chunk+1]
		best := 0
		bestScore, err := m.subspaceScore(vector, 0, start, end)
		if err != nil {
			return PQCode{}, err
		}
		for centroid := 1; centroid < PQCentroidCount; centroid++ {
			score, err := m.subspaceScore(vector, centroid, start, end)
			if err != nil {
				return PQCode{}, err
			}
			if m.metric.Better(score, bestScore) {
				best, bestScore = centroid, score
			}
		}
		codes[chunk] = byte(best)
	}
	return PQCode{modelFingerprint: m.fingerprint, codes: codes}, nil
}

// EncodeBatch converts vectors concurrently while preserving input order.
func (m *PQModel) EncodeBatch(ctx context.Context, vectors [][]float32, workers int) ([]PQCode, error) {
	if ctx == nil {
		return nil, errors.New("core: nil PQ encoding context")
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	if workers < 0 {
		return nil, fmt.Errorf("%w: workers cannot be negative", ErrInvalidPQOptions)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]PQCode, len(vectors))
	err := parallel.ParallelFor(ctx, len(vectors), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		code, err := m.Encode(vectors[index])
		if err != nil {
			return fmt.Errorf("core: encode PQ vector %d: %w", index, err)
		}
		result[index] = code
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Decode reconstructs a vector from one centroid ID per chunk.
func (m *PQModel) Decode(code PQCode) ([]float32, error) {
	if err := m.validateCode(code); err != nil {
		return nil, err
	}
	vector := make([]float32, m.dimension)
	for chunk, encoded := range code.codes {
		start, end := m.chunkOffsets[chunk], m.chunkOffsets[chunk+1]
		pivot := int(encoded)*m.dimension + start
		copy(vector[start:end], m.pivots[pivot:pivot+end-start])
	}
	return vector, nil
}

func (m *PQModel) validate() error {
	if m == nil || m.dimension <= 0 || (m.metric != MetricL2 && m.metric != MetricIP) ||
		len(m.chunkOffsets) < 2 || len(m.pivots) != PQCentroidCount*m.dimension || m.fingerprint == 0 {
		return ErrInvalidPQModel
	}
	return nil
}

func (m *PQModel) validateCode(code PQCode) error {
	if err := m.validate(); err != nil {
		return err
	}
	if err := code.validate(); err != nil {
		return err
	}
	if code.modelFingerprint != m.fingerprint {
		return ErrPQModelMismatch
	}
	if len(code.codes) != m.Chunks() {
		return ErrInvalidPQCode
	}
	return nil
}

func (m *PQModel) subspaceScore(vector []float32, centroid, start, end int) (float32, error) {
	pivot := m.pivots[centroid*m.dimension+start : centroid*m.dimension+end]
	distance := m.distance
	if distance == nil {
		var err error
		distance, err = m.metric.PrevalidatedDistance()
		if err != nil {
			return 0, err
		}
	}
	return distance(vector[start:end], pivot)
}

// PQCode stores one unsigned 8-bit centroid ID per chunk.
type PQCode struct {
	modelFingerprint uint64
	codes            []byte
}

func (c PQCode) Chunks() int { return len(c.codes) }

func (c PQCode) Bytes() []byte { return slices.Clone(c.codes) }

func (c PQCode) validate() error {
	if c.modelFingerprint == 0 || len(c.codes) == 0 {
		return ErrInvalidPQCode
	}
	return nil
}

func pqChunkOffsets(dimension, chunks int) []int {
	offsets := make([]int, chunks+1)
	base, larger := dimension/chunks, dimension%chunks
	for chunk := 0; chunk < chunks; chunk++ {
		width := base
		if chunk < larger {
			width++
		}
		offsets[chunk+1] = offsets[chunk] + width
	}
	return offsets
}

// initializePQKMC2 implements the pinned length-32 Markov-chain centroid
// initializer with a deterministic native seed.
func initializePQKMC2(ctx context.Context, vectors [][]float32, clusters int, metric Metric, seed uint64) ([][]float32, error) {
	if clusters <= 0 || clusters > len(vectors) {
		return nil, ErrInvalidCentroid
	}
	distance, err := metric.PrevalidatedDistance()
	if err != nil {
		return nil, err
	}
	random := splitMix64{state: seed}
	first, err := pqUniformSampleIndices(ctx, len(vectors), 1, &random)
	if err != nil {
		return nil, err
	}
	centroids := [][]float32{slices.Clone(vectors[first[0]])}
	for len(centroids) < clusters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		indices, err := pqUniformSampleIndices(ctx, len(vectors), min(DefaultPQKMC2ChainLength, len(vectors)), &random)
		if err != nil {
			return nil, err
		}
		scores := make([]float32, len(indices))
		for candidateIndex, vectorIndex := range indices {
			best := float32(math.MaxFloat32)
			for centroidIndex, centroid := range centroids {
				if centroidIndex&63 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				score, err := distance(centroid, vectors[vectorIndex])
				if err != nil {
					return nil, err
				}
				if metric == MetricIP {
					score = -score
				}
				if score < best {
					best = score
				}
			}
			scores[candidateIndex] = best
		}
		selectedScore, selected := scores[0], 0
		for candidate := 1; candidate < len(scores); candidate++ {
			if selectedScore == 0 || selectedScore*float32(random.float64()) < scores[candidate] {
				selectedScore, selected = scores[candidate], candidate
			}
		}
		centroids = append(centroids, slices.Clone(vectors[indices[selected]]))
	}
	return centroids, nil
}

func pqUniformSampleIndices(ctx context.Context, count, sample int, random *splitMix64) ([]int, error) {
	return pqUniformSampleIndicesInto(ctx, nil, count, sample, random)
}

func pqUniformSampleIndicesInto(
	ctx context.Context,
	result []int,
	count, sample int,
	random *splitMix64,
) ([]int, error) {
	result = slices.Grow(result[:0], sample)
	remaining := sample
	for index := 0; index < count && remaining > 0; index++ {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if random.intn(count-index) < remaining {
			result = append(result, index)
			remaining--
		}
	}
	return result, nil
}

func fingerprintPQModel(model *PQModel) uint64 {
	hash := sha256.New()
	var header [9]byte
	binary.LittleEndian.PutUint64(header[:8], uint64(model.dimension))
	header[8] = byte(model.metric)
	_, _ = hash.Write(header[:])
	var buffer [16 * 1024]byte
	const offsetsPerBlock = len(buffer) / 8
	for start := 0; start < len(model.chunkOffsets); start += offsetsPerBlock {
		count := min(offsetsPerBlock, len(model.chunkOffsets)-start)
		offsetBytes := buffer[:count*8]
		for index, offset := range model.chunkOffsets[start : start+count] {
			binary.LittleEndian.PutUint64(offsetBytes[index*8:], uint64(offset))
		}
		_, _ = hash.Write(offsetBytes)
	}
	const pivotsPerBlock = len(buffer) / 4
	for start := 0; start < len(model.pivots); start += pivotsPerBlock {
		count := min(pivotsPerBlock, len(model.pivots)-start)
		pivotBytes := buffer[:count*4]
		for index, pivot := range model.pivots[start : start+count] {
			binary.LittleEndian.PutUint32(pivotBytes[index*4:], math.Float32bits(pivot))
		}
		_, _ = hash.Write(pivotBytes)
	}
	sum := hash.Sum(nil)
	fingerprint := binary.LittleEndian.Uint64(sum[:8])
	if fingerprint == 0 {
		return 1
	}
	return fingerprint
}

// PQDistanceTable stores chunk-major public scores for one query. L2 entries
// are squared distances and inner-product entries are similarities.
type PQDistanceTable struct {
	modelFingerprint uint64
	metric           Metric
	chunks           int
	values           []float32
}

func (t *PQDistanceTable) Metric() Metric {
	if t == nil {
		return 0
	}
	return t.metric
}

func (t *PQDistanceTable) Chunks() int {
	if t == nil {
		return 0
	}
	return t.chunks
}

func (t *PQDistanceTable) Centroids() int {
	if t == nil {
		return 0
	}
	return PQCentroidCount
}

func (t *PQDistanceTable) Values() []float32 {
	if t == nil {
		return nil
	}
	return slices.Clone(t.values)
}

// DistanceTable computes all 256 query scores for every chunk.
func (m *PQModel) DistanceTable(query []float32) (*PQDistanceTable, error) {
	if err := m.validate(); err != nil {
		return nil, err
	}
	if err := validateTrainingVector(query, m.dimension); err != nil {
		return nil, err
	}
	values := make([]float32, m.Chunks()*PQCentroidCount)
	for chunk := 0; chunk < m.Chunks(); chunk++ {
		start, end := m.chunkOffsets[chunk], m.chunkOffsets[chunk+1]
		for centroid := 0; centroid < PQCentroidCount; centroid++ {
			score, err := m.subspaceScore(query, centroid, start, end)
			if err != nil {
				return nil, fmt.Errorf("core: build PQ distance table chunk %d centroid %d: %w", chunk, centroid, err)
			}
			values[chunk*PQCentroidCount+centroid] = score
		}
	}
	return &PQDistanceTable{
		modelFingerprint: m.fingerprint, metric: m.metric,
		chunks: m.Chunks(), values: values,
	}, nil
}

// Lookup sums one precomputed entry per code byte.
func (t *PQDistanceTable) Lookup(code PQCode) (float32, error) {
	if err := t.validate(); err != nil {
		return 0, err
	}
	if err := code.validate(); err != nil {
		return 0, err
	}
	if code.modelFingerprint != t.modelFingerprint {
		return 0, ErrPQModelMismatch
	}
	if len(code.codes) != t.chunks {
		return 0, ErrInvalidPQCode
	}
	var score float64
	for chunk, centroid := range code.codes {
		score += float64(t.values[chunk*PQCentroidCount+int(centroid)])
	}
	if math.IsNaN(score) || math.IsInf(score, 0) || score > math.MaxFloat32 || score < -math.MaxFloat32 {
		return 0, ErrPQScoreOverflow
	}
	return float32(score), nil
}

// LookupBatch evaluates codes concurrently while preserving input order.
func (t *PQDistanceTable) LookupBatch(ctx context.Context, codes []PQCode, workers int) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("core: nil PQ lookup context")
	}
	if err := t.validate(); err != nil {
		return nil, err
	}
	if workers < 0 {
		return nil, fmt.Errorf("%w: workers cannot be negative", ErrInvalidPQOptions)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]float32, len(codes))
	err := parallel.ParallelFor(ctx, len(codes), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		score, err := t.Lookup(codes[index])
		if err != nil {
			return fmt.Errorf("core: lookup PQ code %d: %w", index, err)
		}
		result[index] = score
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Distance builds a table and evaluates one code.
func (m *PQModel) Distance(query []float32, code PQCode) (float32, error) {
	table, err := m.DistanceTable(query)
	if err != nil {
		return 0, err
	}
	return table.Lookup(code)
}

func (t *PQDistanceTable) validate() error {
	if t == nil || t.modelFingerprint == 0 || (t.metric != MetricL2 && t.metric != MetricIP) ||
		t.chunks <= 0 || len(t.values) != t.chunks*PQCentroidCount {
		return ErrInvalidPQModel
	}
	return nil
}
