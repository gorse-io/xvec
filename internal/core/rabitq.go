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
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego"
)

const (
	MinRaBitQDimension      = 64
	MaxRaBitQDimension      = 4095
	MinRaBitQTotalBits      = 1
	MaxRaBitQTotalBits      = 9
	DefaultRaBitQTotalBits  = 7
	DefaultRaBitQClusters   = 16
	raBitQScalingSampleSize = 100
)

var (
	ErrInvalidRaBitQOptions  = errors.New("core: invalid RaBitQ options")
	ErrInvalidRaBitQModel    = errors.New("core: invalid RaBitQ model")
	ErrInvalidRaBitQCode     = errors.New("core: invalid RaBitQ code")
	ErrRaBitQModelMismatch   = errors.New("core: RaBitQ code belongs to a different model")
	ErrRaBitQUnsupportedType = errors.New("core: RaBitQ supports L2, IP, and cosine only")
)

// RaBitQOptions configures deterministic centroid and rotation training.
// SampleCount zero uses every vector. A fixed seed produces bit-for-bit stable
// model state across worker counts and supported platforms.
type RaBitQOptions struct {
	Metric        Metric
	TotalBits     int
	Clusters      int
	SampleCount   int
	MaxIterations int
	Workers       int
	Seed          uint64
}

// DefaultRaBitQOptions returns the pinned public defaults.
func DefaultRaBitQOptions(metric Metric) RaBitQOptions {
	return RaBitQOptions{
		Metric:        metric,
		TotalBits:     DefaultRaBitQTotalBits,
		Clusters:      DefaultRaBitQClusters,
		MaxIterations: DefaultKMeansIterations,
	}
}

// Validate checks options that do not depend on the training data.
func (o RaBitQOptions) Validate() error {
	if o.Metric != MetricL2 && o.Metric != MetricIP && o.Metric != MetricCosine {
		return fmt.Errorf("%w: %w", ErrInvalidRaBitQOptions, ErrRaBitQUnsupportedType)
	}
	if o.TotalBits < MinRaBitQTotalBits || o.TotalBits > MaxRaBitQTotalBits {
		return fmt.Errorf("%w: TotalBits must be in [%d,%d]", ErrInvalidRaBitQOptions, MinRaBitQTotalBits, MaxRaBitQTotalBits)
	}
	if o.Clusters <= 0 {
		return fmt.Errorf("%w: Clusters must be positive", ErrInvalidRaBitQOptions)
	}
	if o.SampleCount < 0 {
		return fmt.Errorf("%w: SampleCount cannot be negative", ErrInvalidRaBitQOptions)
	}
	if o.MaxIterations <= 0 {
		return fmt.Errorf("%w: MaxIterations must be positive", ErrInvalidRaBitQOptions)
	}
	if o.Workers < 0 {
		return fmt.Errorf("%w: Workers cannot be negative", ErrInvalidRaBitQOptions)
	}
	return nil
}

// RaBitQModelState is the complete portable state needed to restore a trained
// converter. Centroids are stored before rotation; RotationSigns contains four
// little-endian sign-bit rounds for the padded dimension.
type RaBitQModelState struct {
	Dimension     int
	Metric        Metric
	TotalBits     int
	Centroids     [][]float32
	RotationSigns []byte
	ExtraScale    float64
}

// RaBitQModel is an immutable trained centroid converter.
type RaBitQModel struct {
	dimension        int
	paddedDimension  int
	metric           Metric
	totalBits        int
	extraBits        int
	centroids        [][]float32
	rotatedCentroids [][]float32
	rotationSigns    []byte
	extraScale       float64
	rotator          *FHTRotator
	fingerprint      uint64
}

// TrainRaBitQ trains centroids, deterministic rotation state, and the expected
// extra-code scale used by the baseline's faster converter.
func TrainRaBitQ(ctx context.Context, vectors [][]float32, options RaBitQOptions) (*RaBitQModel, error) {
	if ctx == nil {
		return nil, errors.New("core: nil RaBitQ training context")
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
	if dimension < MinRaBitQDimension || dimension > MaxRaBitQDimension {
		return nil, fmt.Errorf("%w: dimension must be in [%d,%d]", ErrInvalidRaBitQOptions, MinRaBitQDimension, MaxRaBitQDimension)
	}
	if err := validateTrainingVectors(ctx, vectors, dimension, true); err != nil {
		return nil, err
	}

	prepared, err := prepareRaBitQTrainingVectors(ctx, vectors, options.Metric)
	if err != nil {
		return nil, err
	}
	training, err := sampleRaBitQTraining(ctx, prepared, options.SampleCount, options.Seed)
	if err != nil {
		return nil, err
	}
	clusterMetric := options.Metric
	if clusterMetric == MetricCosine {
		clusterMetric = MetricIP
	}
	kmeans := DefaultKMeansOptions(options.Clusters, clusterMetric)
	kmeans.MaxIterations = options.MaxIterations
	kmeans.Workers = options.Workers
	kmeans.Seed = options.Seed ^ 0x7261626974716b6d
	kmeans.Spherical = options.Metric == MetricIP || options.Metric == MetricCosine
	model, err := TrainKMeans(ctx, training, kmeans)
	if err != nil {
		return nil, fmt.Errorf("core: train RaBitQ centroids: %w", err)
	}

	paddedDimension := roundUpRaBitQDimension(dimension)
	random := splitMix64{state: options.Seed ^ 0x726162697471726f}
	signs := make([]byte, 4*paddedDimension/8)
	for index := range signs {
		signs[index] = byte(random.next())
	}
	extraScale := float64(0)
	if options.TotalBits > 1 {
		extraScale, err = trainRaBitQExtraScale(
			ctx, paddedDimension, options.TotalBits-1, options.Workers,
			options.Seed^0x7261626974717363,
		)
		if err != nil {
			return nil, err
		}
	}
	return RestoreRaBitQModel(RaBitQModelState{
		Dimension: dimension, Metric: options.Metric, TotalBits: options.TotalBits,
		Centroids: model.Centroids(), RotationSigns: signs, ExtraScale: extraScale,
	})
}

// RestoreRaBitQModel validates and restores exact portable model state.
func RestoreRaBitQModel(state RaBitQModelState) (*RaBitQModel, error) {
	if state.Dimension < MinRaBitQDimension || state.Dimension > MaxRaBitQDimension {
		return nil, fmt.Errorf("%w: dimension must be in [%d,%d]", ErrInvalidRaBitQModel, MinRaBitQDimension, MaxRaBitQDimension)
	}
	if state.Metric != MetricL2 && state.Metric != MetricIP && state.Metric != MetricCosine {
		return nil, fmt.Errorf("%w: %w", ErrInvalidRaBitQModel, ErrRaBitQUnsupportedType)
	}
	if state.TotalBits < MinRaBitQTotalBits || state.TotalBits > MaxRaBitQTotalBits {
		return nil, fmt.Errorf("%w: TotalBits must be in [%d,%d]", ErrInvalidRaBitQModel, MinRaBitQTotalBits, MaxRaBitQTotalBits)
	}
	if len(state.Centroids) == 0 {
		return nil, fmt.Errorf("%w: no centroids", ErrInvalidRaBitQModel)
	}
	if err := validateTrainingVectors(context.Background(), state.Centroids, state.Dimension, true); err != nil {
		return nil, fmt.Errorf("%w: invalid centroids: %w", ErrInvalidRaBitQModel, err)
	}
	extraBits := state.TotalBits - 1
	if extraBits == 0 {
		if state.ExtraScale != 0 {
			return nil, fmt.Errorf("%w: one-bit model must have zero ExtraScale", ErrInvalidRaBitQModel)
		}
	} else if state.ExtraScale <= 0 || math.IsNaN(state.ExtraScale) || math.IsInf(state.ExtraScale, 0) {
		return nil, fmt.Errorf("%w: ExtraScale must be finite and positive", ErrInvalidRaBitQModel)
	}

	paddedDimension := roundUpRaBitQDimension(state.Dimension)
	wantSigns := 4 * paddedDimension / 8
	if len(state.RotationSigns) != wantSigns {
		return nil, fmt.Errorf("%w: got %d rotation-sign bytes, want %d", ErrInvalidRaBitQModel, len(state.RotationSigns), wantSigns)
	}
	rotator, err := newRaBitQPaddedRotator(state.Dimension, paddedDimension, state.RotationSigns)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidRaBitQModel, err)
	}
	centroids := cloneVectors(state.Centroids)
	rotatedCentroids := make([][]float32, len(centroids))
	for index, centroid := range centroids {
		rotatedCentroids[index], err = rotateRaBitQVector(rotator, state.Dimension, paddedDimension, centroid)
		if err != nil {
			return nil, fmt.Errorf("%w: rotate centroid %d: %w", ErrInvalidRaBitQModel, index, err)
		}
	}
	m := &RaBitQModel{
		dimension: state.Dimension, paddedDimension: paddedDimension,
		metric: state.Metric, totalBits: state.TotalBits, extraBits: extraBits,
		centroids: centroids, rotatedCentroids: rotatedCentroids,
		rotationSigns: slices.Clone(state.RotationSigns), extraScale: state.ExtraScale,
		rotator: rotator,
	}
	m.fingerprint = fingerprintRaBitQModel(m)
	return m, nil
}

// State returns an independent complete model snapshot.
func (m *RaBitQModel) State() RaBitQModelState {
	if m == nil {
		return RaBitQModelState{}
	}
	return RaBitQModelState{
		Dimension: m.dimension, Metric: m.metric, TotalBits: m.totalBits,
		Centroids: cloneVectors(m.centroids), RotationSigns: slices.Clone(m.rotationSigns),
		ExtraScale: m.extraScale,
	}
}

func (m *RaBitQModel) Dimension() int {
	if m == nil {
		return 0
	}
	return m.dimension
}

func (m *RaBitQModel) PaddedDimension() int {
	if m == nil {
		return 0
	}
	return m.paddedDimension
}

func (m *RaBitQModel) Metric() Metric {
	if m == nil {
		return 0
	}
	return m.metric
}

func (m *RaBitQModel) TotalBits() int {
	if m == nil {
		return 0
	}
	return m.totalBits
}

func (m *RaBitQModel) Len() int {
	if m == nil {
		return 0
	}
	return len(m.centroids)
}

func (m *RaBitQModel) Centroids() [][]float32 {
	if m == nil {
		return nil
	}
	return cloneVectors(m.centroids)
}

// Encode converts one vector into an immutable split RaBitQ code.
func (m *RaBitQModel) Encode(vector []float32) (RaBitQCode, error) {
	if err := m.validate(); err != nil {
		return RaBitQCode{}, err
	}
	prepared, err := prepareRaBitQVector(vector, m.dimension, m.metric)
	if err != nil {
		return RaBitQCode{}, err
	}
	metric := m.metric
	if metric == MetricCosine {
		metric = MetricIP
	}
	cluster, _, err := nearestCentroid(metric, m.centroids, prepared)
	if err != nil {
		return RaBitQCode{}, err
	}
	rotated, err := rotateRaBitQVector(m.rotator, m.dimension, m.paddedDimension, prepared)
	if err != nil {
		return RaBitQCode{}, err
	}
	code, err := quantizeRaBitQVector(
		rotated, m.rotatedCentroids[cluster], cluster, m.totalBits,
		m.extraScale, m.metric != MetricL2,
	)
	if err != nil {
		return RaBitQCode{}, err
	}
	code.modelFingerprint = m.fingerprint
	return code, nil
}

// EncodeBatch converts vectors concurrently while preserving input order.
func (m *RaBitQModel) EncodeBatch(ctx context.Context, vectors [][]float32, workers int) ([]RaBitQCode, error) {
	if ctx == nil {
		return nil, errors.New("core: nil RaBitQ encoding context")
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]RaBitQCode, len(vectors))
	err := ailego.ParallelFor(ctx, len(vectors), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		code, err := m.Encode(vectors[index])
		if err != nil {
			return fmt.Errorf("core: encode RaBitQ vector %d: %w", index, err)
		}
		result[index] = code
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// RaBitQQuery owns rotated query state and per-centroid factors. It is
// immutable and safe for concurrent estimates.
type RaBitQQuery struct {
	modelFingerprint uint64
	metric           Metric
	totalBits        int
	extraBits        int
	paddedDimension  int
	rotated          []float32
	sum              float64
	gAdd             []float64
	gError           []float64
}

// PrepareQuery rotates a query and precomputes all centroid-dependent terms.
func (m *RaBitQModel) PrepareQuery(vector []float32) (*RaBitQQuery, error) {
	if err := m.validate(); err != nil {
		return nil, err
	}
	prepared, err := prepareRaBitQVector(vector, m.dimension, m.metric)
	if err != nil {
		return nil, err
	}
	rotated, err := rotateRaBitQVector(m.rotator, m.dimension, m.paddedDimension, prepared)
	if err != nil {
		return nil, err
	}
	query := &RaBitQQuery{
		modelFingerprint: m.fingerprint, metric: m.metric,
		totalBits: m.totalBits, extraBits: m.extraBits,
		paddedDimension: m.paddedDimension, rotated: rotated,
		gAdd: make([]float64, len(m.centroids)), gError: make([]float64, len(m.centroids)),
	}
	for _, value := range rotated {
		query.sum += float64(value)
	}
	for cluster, centroid := range m.rotatedCentroids {
		var squaredDistance, dot float64
		for index, value := range rotated {
			difference := float64(value) - float64(centroid[index])
			squaredDistance += difference * difference
			dot += float64(value) * float64(centroid[index])
		}
		query.gError[cluster] = math.Sqrt(squaredDistance)
		if m.metric == MetricL2 {
			query.gAdd[cluster] = squaredDistance
		} else {
			query.gAdd[cluster] = -dot
		}
	}
	return query, nil
}

// RaBitQEstimate is a lower-is-better approximate distance and the baseline's
// probabilistic error envelope. The bounds are useful for candidate pruning;
// they are not a deterministic guarantee. IP uses 1-inner-product, while
// cosine uses 1-cosine.
type RaBitQEstimate struct {
	Distance   float32
	LowerBound float32
	UpperBound float32
}

// EstimateCoarse evaluates only the one-bit sign code.
func (q *RaBitQQuery) EstimateCoarse(code RaBitQCode) (RaBitQEstimate, error) {
	if err := q.validateCode(code); err != nil {
		return RaBitQEstimate{}, err
	}
	dot := raBitQBinaryDot(code.binaryCode, q.rotated)
	centered := dot - .5*q.sum
	distance := code.coarseAdd + q.gAdd[code.cluster] + code.coarseRescale*centered
	errorBound := code.coarseError * q.gError[code.cluster]
	return makeRaBitQEstimate(distance, errorBound)
}

// Estimate evaluates all configured bits. For a one-bit model it is identical
// to EstimateCoarse.
func (q *RaBitQQuery) Estimate(code RaBitQCode) (RaBitQEstimate, error) {
	if err := q.validateCode(code); err != nil {
		return RaBitQEstimate{}, err
	}
	if q.extraBits == 0 {
		return q.EstimateCoarse(code)
	}
	dot := raBitQFullCodeDot(code, q.rotated)
	center := -(float64(uint64(1)<<q.extraBits) - .5)
	distance := code.fullAdd + q.gAdd[code.cluster] + code.fullRescale*(dot+center*q.sum)
	errorBound := code.coarseError * q.gError[code.cluster] / float64(uint64(1)<<q.extraBits)
	return makeRaBitQEstimate(distance, errorBound)
}

func (q *RaBitQQuery) validateCode(code RaBitQCode) error {
	if q == nil || q.paddedDimension <= 0 || len(q.rotated) != q.paddedDimension || len(q.gAdd) == 0 || len(q.gAdd) != len(q.gError) {
		return ErrInvalidRaBitQModel
	}
	if err := code.validate(); err != nil {
		return err
	}
	if code.modelFingerprint != q.modelFingerprint {
		return ErrRaBitQModelMismatch
	}
	if code.totalBits != q.totalBits || code.paddedDimension != q.paddedDimension || code.cluster >= len(q.gAdd) {
		return ErrInvalidRaBitQCode
	}
	return nil
}

func makeRaBitQEstimate(distance, errorBound float64) (RaBitQEstimate, error) {
	if math.IsNaN(distance) || math.IsInf(distance, 0) || math.IsNaN(errorBound) || math.IsInf(errorBound, 0) || errorBound < 0 {
		return RaBitQEstimate{}, ErrInvalidRaBitQCode
	}
	if math.Abs(distance) > math.MaxFloat32 || errorBound > math.MaxFloat32 {
		return RaBitQEstimate{}, ErrQuantizationOverflow
	}
	lower, upper := distance-errorBound, distance+errorBound
	if lower < -math.MaxFloat32 || upper > math.MaxFloat32 {
		return RaBitQEstimate{}, ErrQuantizationOverflow
	}
	return RaBitQEstimate{Distance: float32(distance), LowerBound: float32(lower), UpperBound: float32(upper)}, nil
}

func (m *RaBitQModel) validate() error {
	if m == nil || m.dimension < MinRaBitQDimension || m.dimension > MaxRaBitQDimension ||
		m.paddedDimension != roundUpRaBitQDimension(m.dimension) || len(m.centroids) == 0 ||
		len(m.centroids) != len(m.rotatedCentroids) || m.rotator == nil || m.fingerprint == 0 {
		return ErrInvalidRaBitQModel
	}
	return nil
}

func prepareRaBitQTrainingVectors(ctx context.Context, vectors [][]float32, metric Metric) ([][]float32, error) {
	prepared := make([][]float32, len(vectors))
	for index, vector := range vectors {
		if index&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		prepared[index] = slices.Clone(vector)
		if metric == MetricCosine {
			normalizeRaBitQVector(prepared[index])
		}
	}
	return prepared, nil
}

func prepareRaBitQVector(vector []float32, dimension int, metric Metric) ([]float32, error) {
	if err := validateTrainingVector(vector, dimension); err != nil {
		return nil, err
	}
	prepared := slices.Clone(vector)
	if metric == MetricCosine {
		normalizeRaBitQVector(prepared)
	}
	return prepared, nil
}

func normalizeRaBitQVector(vector []float32) {
	var normSquared float64
	for _, value := range vector {
		normSquared += float64(value) * float64(value)
	}
	if normSquared == 0 {
		return
	}
	inverse := float32(1 / math.Sqrt(normSquared))
	for index := range vector {
		vector[index] *= inverse
	}
}

func sampleRaBitQTraining(ctx context.Context, vectors [][]float32, sampleCount int, seed uint64) ([][]float32, error) {
	if sampleCount == 0 || sampleCount >= len(vectors) {
		return vectors, nil
	}
	indices := make([]int, sampleCount)
	for index := range indices {
		indices[index] = index
	}
	random := splitMix64{state: seed ^ 0x7261626974717361}
	for index := sampleCount; index < len(vectors); index++ {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		selected := random.intn(index + 1)
		if selected < sampleCount {
			indices[selected] = index
		}
	}
	result := make([][]float32, sampleCount)
	for index, selected := range indices {
		result[index] = vectors[selected]
	}
	return result, nil
}

func newRaBitQPaddedRotator(dimension, paddedDimension int, signs []byte) (*FHTRotator, error) {
	if paddedDimension < dimension || paddedDimension%64 != 0 {
		return nil, ErrInvalidRotator
	}
	bytesPerRound := paddedDimension / 8
	if len(signs) != 4*bytesPerRound {
		return nil, ErrInvalidSigns
	}
	truncated := floorPowerOfTwo(dimension)
	return &FHTRotator{
		dimension: paddedDimension, truncated: truncated, bytesPerRound: bytesPerRound,
		inverseSqrtSize: 1 / float32(math.Sqrt(float64(float32(truncated)))), signs: slices.Clone(signs),
	}, nil
}

func rotateRaBitQVector(rotator *FHTRotator, dimension, paddedDimension int, vector []float32) ([]float32, error) {
	if len(vector) != dimension {
		return nil, ailego.ErrDimensionMismatch
	}
	padded := make([]float32, paddedDimension)
	copy(padded, vector)
	return rotator.Rotate(padded)
}

func roundUpRaBitQDimension(dimension int) int {
	return (dimension + 63) / 64 * 64
}

func fingerprintRaBitQModel(m *RaBitQModel) uint64 {
	hash := sha256.New()
	var scratch [8]byte
	binary.LittleEndian.PutUint64(scratch[:], uint64(m.dimension))
	_, _ = hash.Write(scratch[:])
	binary.LittleEndian.PutUint64(scratch[:], uint64(m.metric))
	_, _ = hash.Write(scratch[:])
	binary.LittleEndian.PutUint64(scratch[:], uint64(m.totalBits))
	_, _ = hash.Write(scratch[:])
	binary.LittleEndian.PutUint64(scratch[:], math.Float64bits(m.extraScale))
	_, _ = hash.Write(scratch[:])
	_, _ = hash.Write(m.rotationSigns)
	for _, centroid := range m.centroids {
		for _, value := range centroid {
			binary.LittleEndian.PutUint32(scratch[:4], math.Float32bits(value))
			_, _ = hash.Write(scratch[:4])
		}
	}
	sum := hash.Sum(nil)
	value := binary.LittleEndian.Uint64(sum)
	if value == 0 {
		return 1
	}
	return value
}
