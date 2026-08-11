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
	"math"
	"slices"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTrainKMeansTwoClusters(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{0, 0}, {0, 1}, {10, 10}, {10, 11}}
	options := DefaultKMeansOptions(2, MetricL2)
	options.InitialCentroids = [][]float32{{0, 0}, {10, 10}}
	model, err := TrainKMeans(context.Background(), vectors, options)
	require.NoError(t, err)
	require.True(t, model.Dimension() == 2)
	require.True(t, model.Len() == 2)
	require.Equal(t, MetricL2, model.Metric())

	wantCentroids := [][]float32{{0, .5}, {10, 10.5}}
	require.Equal(t, wantCentroids, model.Centroids())
	require.True(t, slices.Equal(model.Counts(), []int{2, 2}))
	require.True(t, model.Cost() == 1)
	require.True(t, model.Converged())
	require.True(t, model.Iterations() >= 1)
	require.True(t, model.Iterations() <= options.MaxIterations)

	labels, scores, err := model.Classify(context.Background(), vectors, 4)
	require.NoError(t, err)
	require.True(t, slices.Equal(labels, []int{0, 0, 1, 1}))
	require.True(t, slices.Equal(scores, []float32{.25, .25, .25, .25}))

	label, score, err := model.Nearest([]float32{9, 10})
	require.NoError(t, err)
	require.True(t, label == 1)
	require.True(t, score == 1.25)
}

func TestKMeansDeterministicAcrossWorkers(t *testing.T) {
	t.Parallel()
	vectors := make([][]float32, 300)
	for index := range vectors {
		vectors[index] = []float32{
			float32(index%17) - 8,
			float32(index%13) / 3,
			float32(index%7) * .25,
		}
	}
	for _, initializer := range []KMeansInitializer{KMeansInitReservoir, KMeansInitPlusPlus} {
		options := DefaultKMeansOptions(11, MetricL2)
		options.Initializer = initializer
		options.Seed = 0x123456789abcdef0
		options.Workers = 1
		one, err := TrainKMeans(context.Background(), vectors, options)
		require.NoError(t, err)

		options.Workers = 8
		many, err := TrainKMeans(context.Background(), vectors, options)
		require.NoError(t, err)
		require.Equal(t, many.Centroids(), one.Centroids())
		require.True(t, slices.Equal(one.Counts(), many.Counts()))
		require.Equal(t, many.Cost(), one.Cost())
		require.Equal(t, many.Iterations(), one.Iterations())
		require.Equal(t, many.Converged(), one.Converged())
	}
}

func TestKMeansReservoirSeedFixture(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{0}, {1}, {2}, {3}, {4}, {5}}
	options := DefaultKMeansOptions(3, MetricL2)
	centroids, err := initializeKMeans(context.Background(), vectors, 3, 1, options)
	require.NoError(t, err)
	require.Equal(t, [][]float32{{4}, {5}, {2}}, centroids)
}

func TestKMeansClusterCountCapsAtSamples(t *testing.T) {
	t.Parallel()
	options := DefaultKMeansOptions(10, MetricL2)
	model, err := TrainKMeans(context.Background(), [][]float32{{1}, {2}, {3}}, options)
	require.NoError(t, err)
	require.True(t, model.Len() == 3)
	require.True(t, slices.Equal(model.Counts(), []int{1, 1, 1}))
}

func TestKMeansEmptyPolicies(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{0}, {10}, {20}}
	initial := [][]float32{{0}, {0}, {20}}

	keepOptions := DefaultKMeansOptions(3, MetricL2)
	keepOptions.InitialCentroids = initial
	keepOptions.EmptyPolicy = KMeansEmptyKeep
	keepOptions.MaxIterations = 1
	keep, err := TrainKMeans(context.Background(), vectors, keepOptions)
	require.NoError(t, err)
	require.True(t, keep.Len() == 3)
	require.Equal(t, [][]float32{{5}, {0}, {20}}, keep.Centroids())

	dropOptions := keepOptions
	dropOptions.EmptyPolicy = KMeansEmptyDrop
	drop, err := TrainKMeans(context.Background(), vectors, dropOptions)
	require.NoError(t, err)
	require.True(t, drop.Len() == 2)
	require.True(t, slices.Equal(drop.Counts(), []int{2, 1}))

	reseedOptions := keepOptions
	reseedOptions.EmptyPolicy = KMeansEmptyReseedFarthest
	reseed, err := TrainKMeans(context.Background(), vectors, reseedOptions)
	require.NoError(t, err)
	require.True(t, reseed.Len() == 3)
	require.True(t, slices.Equal(reseed.Counts(), []int{1, 1, 1}))

	centroids := reseed.Centroids()
	require.Equal(t, [][]float32{{5}, {10}, {20}}, centroids)
}

func TestKMeansInnerProductAndSpherical(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{-2, 0}, {-1, 0}, {1, 0}, {3, 0}}
	options := DefaultKMeansOptions(2, MetricIP)
	options.InitialCentroids = [][]float32{{-1, 0}, {1, 0}}
	options.Spherical = true
	model, err := TrainKMeans(context.Background(), vectors, options)
	require.NoError(t, err)

	centroids := model.Centroids()
	assertFloatSlicesClose(t, centroids[0], []float32{-1, 0}, 1e-7)
	assertFloatSlicesClose(t, centroids[1], []float32{1, 0}, 1e-7)
	label, score, err := model.Nearest([]float32{2, 0})
	require.NoError(t, err)
	require.True(t, label == 1)
	require.True(t, score == 2)
	require.Equal(t, float64(-7), model.Cost())
}

func TestKMeansModelOwnsState(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{0}, {10}}
	initial := [][]float32{{0}, {10}}
	options := DefaultKMeansOptions(2, MetricL2)
	options.InitialCentroids = initial
	model, err := TrainKMeans(context.Background(), vectors, options)
	require.NoError(t, err)

	vectors[0][0] = 99
	initial[0][0] = 88
	centroids := model.Centroids()
	centroids[0][0] = 77
	counts := model.Counts()
	counts[0] = 99
	require.True(t, model.Centroids()[0][0] == 0,
		"model state aliases caller or accessor slices")
	require.True(t, model.Counts()[0] == 1,
		"model state aliases caller or accessor slices")
}

func TestKMeansValidation(t *testing.T) {
	t.Parallel()
	valid := DefaultKMeansOptions(2, MetricL2)
	{
		_, err := TrainKMeans(nil, [][]float32{{1}}, valid)
		require.Error(t, err,
			"nil context accepted")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := TrainKMeans(ctx, [][]float32{{1}}, valid)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := TrainKMeans(context.Background(), nil, valid)
		require.ErrorIs(t, err, ErrEmptyTrainingSet)
	}
	{
		_, err := TrainKMeans(context.Background(), [][]float32{{}}, valid)
		require.ErrorIs(t, err, ailego.ErrEmptyVector)
	}
	{
		_, err := TrainKMeans(context.Background(), [][]float32{{1}, {1, 2}}, valid)
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		_, err := TrainKMeans(context.Background(), [][]float32{{float32(math.NaN())}}, valid)
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}

	invalidOptions := []KMeansOptions{
		{},
		func() KMeansOptions { value := valid; value.Clusters = 0; return value }(),
		func() KMeansOptions { value := valid; value.MaxIterations = 0; return value }(),
		func() KMeansOptions { value := valid; value.Tolerance = -1; return value }(),
		func() KMeansOptions { value := valid; value.Tolerance = math.NaN(); return value }(),
		func() KMeansOptions { value := valid; value.Metric = 0; return value }(),
		func() KMeansOptions { value := valid; value.Initializer = 99; return value }(),
		func() KMeansOptions { value := valid; value.EmptyPolicy = 99; return value }(),
	}
	for _, options := range invalidOptions {
		{
			_, err := TrainKMeans(context.Background(), [][]float32{{1}}, options)
			assert.ErrorIs(t, err, ErrInvalidKMeansOptions)
		}
	}

	badInitial := valid
	badInitial.InitialCentroids = [][]float32{{1}}
	{
		_, err := TrainKMeans(context.Background(), [][]float32{{1}, {2}}, badInitial)
		require.ErrorIs(t, err, ErrInvalidCentroid)
	}
}

func TestKMeansModelValidation(t *testing.T) {
	t.Parallel()
	var model *KMeansModel
	{
		_, _, err := model.Nearest([]float32{1})
		require.Error(t, err,
			"nil model accepted")
	}

	options := DefaultKMeansOptions(1, MetricL2)
	model, err := TrainKMeans(context.Background(), [][]float32{{1}, {2}}, options)
	require.NoError(t, err)
	{
		_, _, err := model.Nearest([]float32{1, 2})
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		_, _, err := model.Classify(nil, nil, 1)
		require.Error(t, err,
			"nil classify context accepted")
	}

	labels, scores, err := model.Classify(context.Background(), nil, 1)
	require.NoError(t, err)
	require.NotNil(t, labels)
	require.NotNil(t, scores)
	require.Len(t, labels, 0)
	require.Len(t, scores, 0)
}

func TestKMeansTieBreaksByCentroidIndex(t *testing.T) {
	t.Parallel()
	model := &KMeansModel{
		metric: MetricL2, dimension: 1,
		centroids: [][]float32{{-1}, {1}}, counts: []int{0, 0},
	}
	label, score, err := model.Nearest([]float32{0})
	require.NoError(t, err)
	require.True(t, label == 0)
	require.True(t, score == 1)
}

func FuzzTrainKMeans(f *testing.F) {
	f.Add(uint8(7), uint8(3), uint64(42), float32(1), float32(-2))
	f.Add(uint8(2), uint8(8), uint64(0), float32(0), float32(0))
	f.Fuzz(func(t *testing.T, rawCount, rawClusters uint8, seed uint64, left, right float32) {
		count := int(rawCount%32) + 1
		clusters := int(rawClusters%8) + 1
		vectors := make([][]float32, count)
		for index := range vectors {
			vectors[index] = []float32{left + float32(index%3), right - float32(index%5)}
		}
		options := DefaultKMeansOptions(clusters, MetricL2)
		options.MaxIterations = 5
		options.Seed = seed
		options.Initializer = KMeansInitializer(seed%2 + 1)
		model, err := TrainKMeans(context.Background(), vectors, options)
		if errors.Is(err, ailego.ErrNonFiniteVector) {
			return
		}
		require.NoError(t, err)
		require.False(t, model.Len() == 0)
		require.True(t, model.Len() <= min(count, clusters))

		labels, scores, err := model.Classify(context.Background(), vectors, 3)
		require.NoError(t, err)
		require.Len(t, labels, count)
		require.Len(t, scores, count)
	})
}

func BenchmarkTrainKMeans(b *testing.B) {
	vectors := make([][]float32, 2048)
	for index := range vectors {
		vectors[index] = make([]float32, 32)
		for coordinate := range vectors[index] {
			vectors[index][coordinate] = float32((index*17+coordinate*31)%997) / 997
		}
	}
	options := DefaultKMeansOptions(32, MetricL2)
	options.MaxIterations = 10
	b.ResetTimer()
	for range b.N {
		{
			_, err := TrainKMeans(context.Background(), vectors, options)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}
