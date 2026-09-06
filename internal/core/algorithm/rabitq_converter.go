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

	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
	"github.com/gorse-io/xvec/pkg/rabitq"
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
	rotator          rabitq.Rotator
	queryConfig      rabitq.RaBitQConfig
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
		extraScale = rabitq.FasterConfig(paddedDimension, options.TotalBits).TConst
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
	rotator, err := rabitq.ChooseRotator(state.Dimension, rabitq.FhtKacRotator, paddedDimension)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidRaBitQModel, err)
	}
	if err := rotator.Load(state.RotationSigns); err != nil {
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
		rotator: rotator, queryConfig: rabitq.FasterConfig(paddedDimension, rabitq.SplitSingleQueryNumBits),
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

func (m *RaBitQModel) rabitqMetric() rabitq.MetricType {
	if m.metric == MetricL2 {
		return rabitq.MetricL2
	}
	return rabitq.MetricIP
}

// Encode converts one vector into an immutable split RaBitQ code.
func (m *RaBitQModel) Encode(vector []float32) (RaBitQCode, error) {
	if err := m.validate(); err != nil {
		return RaBitQCode{}, err
	}
	prepared, cluster, err := m.prepareAndCluster(vector)
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

func (m *RaBitQModel) prepareAndCluster(vector []float32) ([]float32, int, error) {
	prepared, err := prepareRaBitQVector(vector, m.dimension, m.metric)
	if err != nil {
		return nil, 0, err
	}
	metric := m.metric
	if metric == MetricCosine {
		metric = MetricIP
	}
	cluster, _, err := nearestCentroid(metric, m.centroids, prepared)
	if err != nil {
		return nil, 0, err
	}
	return prepared, cluster, nil
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
	err := parallel.ParallelFor(ctx, len(vectors), workers, func(ctx context.Context, index int) error {
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
	single           *rabitq.SplitSingleQuery
	ipFunc           rabitq.ExcodeIPFunc
	gAdd             []float32
	gError           []float32
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
	single, err := rabitq.NewSplitSingleQuery(rotated, m.paddedDimension, m.extraBits, m.queryConfig, m.rabitqMetric())
	if err != nil {
		return nil, fmt.Errorf("core: prepare split-single RaBitQ query: %w", err)
	}
	ipFunc, err := rabitq.SelectExcodeIPFunc(m.extraBits)
	if err != nil {
		return nil, fmt.Errorf("core: select RaBitQ extra-code kernel: %w", err)
	}
	query := &RaBitQQuery{
		modelFingerprint: m.fingerprint, metric: m.metric,
		totalBits: m.totalBits, extraBits: m.extraBits,
		paddedDimension: m.paddedDimension, rotated: rotated, single: single, ipFunc: ipFunc,
		gAdd: make([]float32, len(m.centroids)), gError: make([]float32, len(m.centroids)),
	}
	for cluster, centroid := range m.rotatedCentroids {
		query.gError[cluster], err = raBitQResidualNorm(rotated, centroid)
		if err != nil {
			return nil, err
		}
		if m.metric == MetricL2 {
			query.gAdd[cluster] = query.gError[cluster] * query.gError[cluster]
			if math.IsInf(float64(query.gAdd[cluster]), 0) {
				return nil, fmt.Errorf("core: RaBitQ query-to-centroid squared distance is not representable")
			}
		} else {
			query.gAdd[cluster] = -rabitq.DotProduct(rotated, centroid)
		}
	}
	return query, nil
}

// raBitQBatchQuery is the IVF query path. One FastScan LUT is reused while
// cluster-specific g factors are swapped before scanning each list.
type raBitQBatchQuery struct {
	paddedDimension int
	extraBits       int
	batch           *rabitq.SplitBatchQuery
	ipFunc          rabitq.ExcodeIPFunc
	residualNorm    []float32
	centroidIP      []float32
}

func (m *RaBitQModel) prepareBatchQuery(vector []float32) (*raBitQBatchQuery, error) {
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
	batch, err := rabitq.NewSplitBatchQuery(rotated, m.paddedDimension, m.extraBits, m.rabitqMetric(), true)
	if err != nil {
		return nil, fmt.Errorf("core: prepare split-batch RaBitQ query: %w", err)
	}
	ipFunc, err := rabitq.SelectExcodeIPFunc(m.extraBits)
	if err != nil {
		return nil, fmt.Errorf("core: select RaBitQ extra-code kernel: %w", err)
	}
	query := &raBitQBatchQuery{
		paddedDimension: m.paddedDimension, extraBits: m.extraBits,
		batch: batch, ipFunc: ipFunc, residualNorm: make([]float32, len(m.rotatedCentroids)),
	}
	if m.metric != MetricL2 {
		query.centroidIP = make([]float32, len(m.rotatedCentroids))
	}
	for cluster, centroid := range m.rotatedCentroids {
		query.residualNorm[cluster], err = raBitQResidualNorm(rotated, centroid)
		if err != nil {
			return nil, err
		}
		if m.metric == MetricL2 && math.IsInf(float64(query.residualNorm[cluster]*query.residualNorm[cluster]), 0) {
			return nil, fmt.Errorf("core: RaBitQ query-to-centroid squared distance is not representable")
		}
		if m.metric != MetricL2 {
			query.centroidIP[cluster] = rabitq.DotProduct(rotated, centroid)
			if math.IsNaN(float64(query.centroidIP[cluster])) || math.IsInf(float64(query.centroidIP[cluster]), 0) {
				return nil, fmt.Errorf("core: RaBitQ query-to-centroid inner product is not representable")
			}
		}
	}
	return query, nil
}

func (q *raBitQBatchQuery) prepareCluster(cluster int) error {
	if q == nil || q.batch == nil || cluster < 0 || cluster >= len(q.residualNorm) {
		return ErrInvalidRaBitQCode
	}
	if len(q.centroidIP) == 0 {
		q.batch.SetGAdd(q.residualNorm[cluster])
	} else {
		q.batch.SetGAdd(q.residualNorm[cluster], q.centroidIP[cluster])
	}
	return nil
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
	_, distance, lower := rabitq.SplitSingleEstDist(
		code.binData, q.single, q.paddedDimension, q.gAdd[code.cluster], q.gError[code.cluster],
	)
	return makeRaBitQEstimate(distance, lower)
}

// Estimate evaluates all configured bits. For a one-bit model it is identical
// to EstimateCoarse.
func (q *RaBitQQuery) Estimate(code RaBitQCode) (RaBitQEstimate, error) {
	if err := q.validateCode(code); err != nil {
		return RaBitQEstimate{}, err
	}
	distance, lower, _ := rabitq.SplitSingleFullDist(
		code.binData, code.exData, q.ipFunc, q.single, q.paddedDimension, q.extraBits,
		q.gAdd[code.cluster], q.gError[code.cluster],
	)
	return makeRaBitQEstimate(distance, lower)
}

func (q *RaBitQQuery) validateCode(code RaBitQCode) error {
	if q == nil || q.paddedDimension <= 0 || len(q.rotated) != q.paddedDimension || q.single == nil || q.ipFunc == nil ||
		len(q.gAdd) == 0 || len(q.gAdd) != len(q.gError) {
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

func makeRaBitQEstimate(distance, lower float32) (RaBitQEstimate, error) {
	errorBound := distance - lower
	if math.IsNaN(float64(distance)) || math.IsInf(float64(distance), 0) ||
		math.IsNaN(float64(errorBound)) || math.IsInf(float64(errorBound), 0) || errorBound < 0 {
		return RaBitQEstimate{}, ErrInvalidRaBitQCode
	}
	upper := distance + errorBound
	if math.IsNaN(float64(upper)) || math.IsInf(float64(upper), 0) {
		return RaBitQEstimate{}, ErrQuantizationOverflow
	}
	return RaBitQEstimate{Distance: distance, LowerBound: lower, UpperBound: upper}, nil
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
	var scale float64
	for _, value := range vector {
		absolute := math.Abs(float64(value))
		if absolute > scale {
			scale = absolute
		}
	}
	if scale == 0 {
		return
	}
	var scaledNormSquared float64
	for _, value := range vector {
		scaled := float64(value) / scale
		scaledNormSquared += scaled * scaled
	}
	norm := scale * math.Sqrt(scaledNormSquared)
	for index := range vector {
		vector[index] = float32(float64(vector[index]) / norm)
	}
}

func raBitQResidualNorm(left, right []float32) (float32, error) {
	var scale float64
	for index := range left {
		difference := math.Abs(float64(left[index]) - float64(right[index]))
		if difference > scale {
			scale = difference
		}
	}
	if scale == 0 {
		return 0, nil
	}
	var scaledNormSquared float64
	for index := range left {
		difference := (float64(left[index]) - float64(right[index])) / scale
		scaledNormSquared += difference * difference
	}
	norm := scale * math.Sqrt(scaledNormSquared)
	if math.IsNaN(norm) || math.IsInf(norm, 0) || norm > math.MaxFloat32 {
		return 0, fmt.Errorf("core: RaBitQ query-to-centroid distance is not representable")
	}
	return float32(norm), nil
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

func rotateRaBitQVector(rotator rabitq.Rotator, dimension, paddedDimension int, vector []float32) ([]float32, error) {
	if len(vector) != dimension {
		return nil, mathutil.ErrDimensionMismatch
	}
	rotated := make([]float32, paddedDimension)
	if err := rotator.Rotate(vector, rotated); err != nil {
		return nil, err
	}
	return rotated, nil
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
