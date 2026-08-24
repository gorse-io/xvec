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

	"github.com/gorse-io/xvec/internal/ailego/algorithm"
	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/parallel"
)

const (
	DefaultKMeansIterations = 20
	DefaultKMeansTolerance  = 1.1920928955078125e-7
)

var (
	ErrInvalidKMeansOptions = errors.New("core: invalid k-means options")
	ErrEmptyTrainingSet     = errors.New("core: k-means training set is empty")
	ErrInvalidCentroid      = errors.New("core: invalid k-means centroid")
)

// KMeansInitializer selects initial centroids.
type KMeansInitializer uint8

const (
	// KMeansInitReservoir selects samples uniformly without replacement,
	// matching the pinned baseline's default initialization family.
	KMeansInitReservoir KMeansInitializer = iota + 1
	// KMeansInitPlusPlus uses squared-L2 weighted sampling after its first
	// uniformly selected sample.
	KMeansInitPlusPlus
)

// KMeansEmptyPolicy controls an empty centroid after an update.
type KMeansEmptyPolicy uint8

const (
	// KMeansEmptyKeep retains the previous centroid.
	KMeansEmptyKeep KMeansEmptyPolicy = iota + 1
	// KMeansEmptyReseedFarthest moves each empty centroid to the worst assigned
	// training vector, without reusing a vector in the same update.
	KMeansEmptyReseedFarthest
	// KMeansEmptyDrop removes empty centroids.
	KMeansEmptyDrop
)

// KMeansOptions configures deterministic Lloyd training. InitialCentroids, if
// present, replaces random initialization and must match the effective cluster
// count min(Clusters,len(training set)).
type KMeansOptions struct {
	Clusters         int
	MaxIterations    int
	Tolerance        float64
	Metric           Metric
	Workers          int
	Seed             uint64
	Initializer      KMeansInitializer
	EmptyPolicy      KMeansEmptyPolicy
	Spherical        bool
	InitialCentroids [][]float32
}

// DefaultKMeansOptions returns baseline-oriented defaults plus deterministic
// empty-cluster recovery suitable for index construction.
func DefaultKMeansOptions(clusters int, metric Metric) KMeansOptions {
	return KMeansOptions{
		Clusters:      clusters,
		MaxIterations: DefaultKMeansIterations,
		Tolerance:     DefaultKMeansTolerance,
		Metric:        metric,
		Initializer:   KMeansInitReservoir,
		EmptyPolicy:   KMeansEmptyReseedFarthest,
	}
}

// KMeansModel is an immutable trained centroid set.
type KMeansModel struct {
	metric     Metric
	dimension  int
	centroids  [][]float32
	counts     []int
	cost       float64
	iterations int
	converged  bool
}

// Metric returns the assignment metric.
func (m *KMeansModel) Metric() Metric {
	if m == nil {
		return 0
	}
	return m.metric
}

// Dimension returns the vector dimension.
func (m *KMeansModel) Dimension() int {
	if m == nil {
		return 0
	}
	return m.dimension
}

// Len returns the final centroid count.
func (m *KMeansModel) Len() int {
	if m == nil {
		return 0
	}
	return len(m.centroids)
}

// Centroids returns a deep copy in deterministic centroid order.
func (m *KMeansModel) Centroids() [][]float32 {
	if m == nil {
		return nil
	}
	return cloneVectors(m.centroids)
}

// Counts returns final assignment counts in centroid order.
func (m *KMeansModel) Counts() []int {
	if m == nil {
		return nil
	}
	return slices.Clone(m.counts)
}

// Cost returns the final lower-is-better objective. It is the score sum for
// distance metrics and the negated similarity sum for inner product.
func (m *KMeansModel) Cost() float64 {
	if m == nil {
		return 0
	}
	return m.cost
}

// Iterations returns the number of completed Lloyd update rounds.
func (m *KMeansModel) Iterations() int {
	if m == nil {
		return 0
	}
	return m.iterations
}

// Converged reports whether training stopped on the configured tolerance.
func (m *KMeansModel) Converged() bool {
	return m != nil && m.converged
}

// Nearest returns the best centroid and metric score for one vector. Equal
// scores choose the lower centroid index.
func (m *KMeansModel) Nearest(vector []float32) (int, float32, error) {
	if err := m.validate(); err != nil {
		return 0, 0, err
	}
	if err := validateTrainingVector(vector, m.dimension); err != nil {
		return 0, 0, err
	}
	return nearestCentroid(m.metric, m.centroids, vector)
}

// Classify assigns vectors concurrently while preserving input order and
// deterministic lower-index tie breaking.
func (m *KMeansModel) Classify(ctx context.Context, vectors [][]float32, workers int) ([]int, []float32, error) {
	if err := m.validate(); err != nil {
		return nil, nil, err
	}
	if ctx == nil {
		return nil, nil, errors.New("core: nil k-means classify context")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if err := validateTrainingVectors(ctx, vectors, m.dimension, false); err != nil {
		return nil, nil, err
	}
	return assignKMeans(ctx, m.metric, vectors, m.centroids, workers)
}

// TrainKMeans runs deterministic Lloyd iterations over finite FP32 vectors.
// Assignment is parallel; accumulation is performed in input order so results
// are bit-for-bit stable across worker counts.
func TrainKMeans(ctx context.Context, vectors [][]float32, options KMeansOptions) (*KMeansModel, error) {
	model, _, err := trainKMeansWithAssignments(ctx, vectors, options)
	return model, err
}

// trainKMeansWithAssignments returns the final assignment computed while
// producing model statistics. Index builders can consume it without repeating
// the most expensive phase of Lloyd training.
func trainKMeansWithAssignments(ctx context.Context, vectors [][]float32, options KMeansOptions) (*KMeansModel, []int, error) {
	if ctx == nil {
		return nil, nil, errors.New("core: nil k-means training context")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if err := validateKMeansOptions(options); err != nil {
		return nil, nil, err
	}
	if len(vectors) == 0 {
		return nil, nil, ErrEmptyTrainingSet
	}
	dimension := len(vectors[0])
	if err := validateTrainingVectors(ctx, vectors, dimension, true); err != nil {
		return nil, nil, err
	}
	effectiveClusters := min(options.Clusters, len(vectors))
	centroids, err := initializeKMeans(ctx, vectors, effectiveClusters, dimension, options)
	if err != nil {
		return nil, nil, err
	}
	if options.Spherical {
		for index := range centroids {
			if index&255 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, nil, err
				}
			}
			mathutil.NormalizeL2(centroids[index])
		}
	}

	previousCost := float64(0)
	iterations := 0
	converged := false
	for iteration := 0; iteration < options.MaxIterations && len(centroids) > 0; iteration++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		labels, scores, err := assignKMeans(ctx, options.Metric, vectors, centroids, options.Workers)
		if err != nil {
			return nil, nil, err
		}
		cost, next, changedShape, err := updateKMeans(ctx, vectors, centroids, labels, scores, options)
		if err != nil {
			return nil, nil, err
		}
		centroids = next
		iterations = iteration + 1
		if !changedShape && math.Abs(cost-previousCost) < options.Tolerance {
			converged = true
			break
		}
		previousCost = cost
	}
	if len(centroids) == 0 {
		return nil, nil, ErrInvalidCentroid
	}

	labels, scores, err := assignKMeans(ctx, options.Metric, vectors, centroids, options.Workers)
	if err != nil {
		return nil, nil, err
	}
	counts := make([]int, len(centroids))
	for index, label := range labels {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, nil, err
			}
		}
		counts[label]++
	}
	modelCentroids, err := cloneVectorsContext(ctx, centroids)
	if err != nil {
		return nil, nil, err
	}
	finalCost, err := kMeansObjective(ctx, options.Metric, scores)
	if err != nil {
		return nil, nil, err
	}
	return &KMeansModel{
		metric:     options.Metric,
		dimension:  dimension,
		centroids:  modelCentroids,
		counts:     counts,
		cost:       finalCost,
		iterations: iterations,
		converged:  converged,
	}, labels, nil
}

func validateKMeansOptions(options KMeansOptions) error {
	if options.Clusters <= 0 {
		return fmt.Errorf("%w: Clusters must be positive", ErrInvalidKMeansOptions)
	}
	if options.MaxIterations <= 0 {
		return fmt.Errorf("%w: MaxIterations must be positive", ErrInvalidKMeansOptions)
	}
	if options.Tolerance < 0 || math.IsNaN(options.Tolerance) || math.IsInf(options.Tolerance, 0) {
		return fmt.Errorf("%w: Tolerance must be finite and non-negative", ErrInvalidKMeansOptions)
	}
	if !options.Metric.Valid() {
		return fmt.Errorf("%w: invalid metric", ErrInvalidKMeansOptions)
	}
	if options.Initializer != KMeansInitReservoir && options.Initializer != KMeansInitPlusPlus {
		return fmt.Errorf("%w: invalid initializer", ErrInvalidKMeansOptions)
	}
	if options.EmptyPolicy < KMeansEmptyKeep || options.EmptyPolicy > KMeansEmptyDrop {
		return fmt.Errorf("%w: invalid empty-cluster policy", ErrInvalidKMeansOptions)
	}
	return nil
}

func validateTrainingVectors(ctx context.Context, vectors [][]float32, dimension int, requireNonEmpty bool) error {
	if requireNonEmpty && len(vectors) == 0 {
		return ErrEmptyTrainingSet
	}
	if dimension <= 0 {
		return mathutil.ErrEmptyVector
	}
	for index, vector := range vectors {
		if ctx != nil && index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if err := validateTrainingVector(vector, dimension); err != nil {
			return fmt.Errorf("core: training vector %d: %w", index, err)
		}
	}
	return nil
}

func validateTrainingVector(vector []float32, dimension int) error {
	if len(vector) != dimension {
		return mathutil.ErrDimensionMismatch
	}
	for _, value := range vector {
		if !finiteFloat32(value) {
			return mathutil.ErrNonFiniteVector
		}
	}
	return nil
}

func initializeKMeans(ctx context.Context, vectors [][]float32, clusters, dimension int, options KMeansOptions) ([][]float32, error) {
	if options.InitialCentroids != nil {
		if len(options.InitialCentroids) != clusters {
			return nil, fmt.Errorf("%w: got %d initial centroids, want %d", ErrInvalidCentroid, len(options.InitialCentroids), clusters)
		}
		if err := validateTrainingVectors(ctx, options.InitialCentroids, dimension, true); err != nil {
			return nil, fmt.Errorf("%w: %w", ErrInvalidCentroid, err)
		}
		return cloneVectorsContext(ctx, options.InitialCentroids)
	}
	random := splitMix64{state: options.Seed}
	switch options.Initializer {
	case KMeansInitReservoir:
		return algorithm.InitializeReservoir(ctx, vectors, clusters, random.intn)
	case KMeansInitPlusPlus:
		return algorithm.InitializePlusPlus(
			ctx,
			vectors,
			clusters,
			random.intn,
			random.float64,
			squaredL2Float64,
		)
	default:
		return nil, ErrInvalidKMeansOptions
	}
}

func assignKMeans(
	ctx context.Context,
	metric Metric,
	vectors, centroids [][]float32,
	workers int,
) ([]int, []float32, error) {
	if metric == MetricCosine {
		return assignKMeansCosine(ctx, vectors, centroids, workers)
	}
	distance, err := metric.PrevalidatedDistance()
	if err != nil {
		return nil, nil, err
	}
	labels := make([]int, len(vectors))
	scores := make([]float32, len(vectors))
	err = parallel.ParallelFor(ctx, len(vectors), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		label, score, err := nearestCentroidPrevalidatedContext(ctx, metric, distance, centroids, vectors[index])
		if err != nil {
			return fmt.Errorf("core: assign k-means vector %d: %w", index, err)
		}
		labels[index], scores[index] = label, score
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return labels, scores, nil
}

// assignKMeansCosine hoists both sides' magnitudes out of the centroid loop.
// The centroid update remains unchanged, so this is score-equivalent to the
// generic cosine path without copying or normalizing the training set.
func assignKMeansCosine(ctx context.Context, vectors, centroids [][]float32, workers int) ([]int, []float32, error) {
	centroidMagnitudes := make([]float32, len(centroids))
	for index, centroid := range centroids {
		magnitude, err := mathutil.L2MagnitudePrevalidated(centroid)
		if err != nil {
			return nil, nil, fmt.Errorf("core: prepare k-means centroid %d: %w", index, err)
		}
		centroidMagnitudes[index] = magnitude
	}
	labels := make([]int, len(vectors))
	scores := make([]float32, len(vectors))
	err := parallel.ParallelFor(ctx, len(vectors), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		vectorMagnitude, err := mathutil.L2MagnitudePrevalidated(vectors[index])
		if err != nil {
			return fmt.Errorf("core: prepare k-means vector %d: %w", index, err)
		}
		label, score, err := nearestCosineCentroidContext(ctx, centroids, centroidMagnitudes, vectors[index], vectorMagnitude)
		if err != nil {
			return fmt.Errorf("core: assign k-means vector %d: %w", index, err)
		}
		labels[index], scores[index] = label, score
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return labels, scores, nil
}

func nearestCosineCentroidContext(
	ctx context.Context,
	centroids [][]float32,
	centroidMagnitudes []float32,
	vector []float32,
	vectorMagnitude float32,
) (int, float32, error) {
	if len(centroids) == 0 {
		return 0, 0, ErrInvalidCentroid
	}
	bestIndex := 0
	bestScore, err := mathutil.CosineDistanceWithMagnitudesPrevalidated(
		centroids[0], vector, centroidMagnitudes[0], vectorMagnitude,
	)
	if err != nil {
		return 0, 0, err
	}
	for index := 1; index < len(centroids); index++ {
		if ctx != nil && index&63 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, 0, err
			}
		}
		score, err := mathutil.CosineDistanceWithMagnitudesPrevalidated(
			centroids[index], vector, centroidMagnitudes[index], vectorMagnitude,
		)
		if err != nil {
			return 0, 0, err
		}
		if score < bestScore {
			bestIndex, bestScore = index, score
		}
	}
	return bestIndex, bestScore, nil
}

func nearestCentroid(metric Metric, centroids [][]float32, vector []float32) (int, float32, error) {
	return nearestCentroidContext(context.Background(), metric, centroids, vector)
}

func nearestCentroidContext(ctx context.Context, metric Metric, centroids [][]float32, vector []float32) (int, float32, error) {
	distance, err := metric.PrevalidatedDistance()
	if err != nil {
		return 0, 0, err
	}
	return nearestCentroidPrevalidatedContext(ctx, metric, distance, centroids, vector)
}

func nearestCentroidPrevalidatedContext(
	ctx context.Context,
	metric Metric,
	distance mathutil.DenseDistance,
	centroids [][]float32,
	vector []float32,
) (int, float32, error) {
	if len(centroids) == 0 {
		return 0, 0, ErrInvalidCentroid
	}
	bestIndex := 0
	bestScore, err := distance(centroids[0], vector)
	if err != nil {
		return 0, 0, err
	}
	for index := 1; index < len(centroids); index++ {
		if ctx != nil && index&63 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, 0, err
			}
		}
		score, err := distance(centroids[index], vector)
		if err != nil {
			return 0, 0, err
		}
		if metric.Better(score, bestScore) {
			bestIndex, bestScore = index, score
		}
	}
	return bestIndex, bestScore, nil
}

func updateKMeans(
	ctx context.Context,
	vectors, centroids [][]float32,
	labels []int,
	scores []float32,
	options KMeansOptions,
) (float64, [][]float32, bool, error) {
	emptyPolicy := algorithm.EmptyKeep
	switch options.EmptyPolicy {
	case KMeansEmptyReseedFarthest:
		emptyPolicy = algorithm.EmptyReseedFarthest
	case KMeansEmptyDrop:
		emptyPolicy = algorithm.EmptyDrop
	}
	return algorithm.LloydUpdate(ctx, vectors, centroids, labels, scores, algorithm.LloydOptions{
		Spherical:   options.Spherical,
		EmptyPolicy: emptyPolicy,
		Worse: func(current, candidate float32) bool {
			return options.Metric.Better(current, candidate)
		},
		Objective: func(ctx context.Context, scores []float32) (float64, error) {
			return kMeansObjective(ctx, options.Metric, scores)
		},
	})
}

func kMeansObjective(ctx context.Context, metric Metric, scores []float32) (float64, error) {
	var cost float64
	for index, score := range scores {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		if metric == MetricIP {
			cost -= float64(score)
		} else {
			cost += float64(score)
		}
	}
	return cost, nil
}

func squaredL2Float64(left, right []float32) (float64, error) {
	if len(left) != len(right) {
		return 0, mathutil.ErrDimensionMismatch
	}
	var sum float64
	for index, value := range left {
		difference := float64(value) - float64(right[index])
		sum += difference * difference
	}
	if math.IsInf(sum, 0) || math.IsNaN(sum) {
		return 0, mathutil.ErrNonFiniteVector
	}
	return sum, nil
}

func cloneVectors(vectors [][]float32) [][]float32 {
	if vectors == nil {
		return nil
	}
	cloned := make([][]float32, len(vectors))
	for index := range vectors {
		cloned[index] = slices.Clone(vectors[index])
	}
	return cloned
}

func cloneVectorsContext(ctx context.Context, vectors [][]float32) ([][]float32, error) {
	if vectors == nil {
		return nil, nil
	}
	cloned := make([][]float32, len(vectors))
	for index := range vectors {
		if index&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		cloned[index] = slices.Clone(vectors[index])
	}
	return cloned, nil
}

func (m *KMeansModel) validate() error {
	if m == nil || !m.metric.Valid() || m.dimension <= 0 || len(m.centroids) == 0 {
		return errors.New("core: invalid k-means model")
	}
	return nil
}

// splitMix64 is a small fixed algorithm used so seeds are reproducible across
// Go versions and platforms.
type splitMix64 struct{ state uint64 }

func (r *splitMix64) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	value := r.state
	value = (value ^ value>>30) * 0xbf58476d1ce4e5b9
	value = (value ^ value>>27) * 0x94d049bb133111eb
	return value ^ value>>31
}

func (r *splitMix64) intn(limit int) int {
	if limit <= 1 {
		return 0
	}
	// Rejection avoids modulo bias without relying on a platform-sized int.
	bound := uint64(limit)
	threshold := -bound % bound
	for {
		value := r.next()
		if value >= threshold {
			return int(value % bound)
		}
	}
}

func (r *splitMix64) float64() float64 {
	return float64(r.next()>>11) * (1.0 / (1 << 53))
}
