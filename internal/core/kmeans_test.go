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
	"reflect"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestTrainKMeansTwoClusters(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{0, 0}, {0, 1}, {10, 10}, {10, 11}}
	options := DefaultKMeansOptions(2, MetricL2)
	options.InitialCentroids = [][]float32{{0, 0}, {10, 10}}
	model, err := TrainKMeans(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	if model.Dimension() != 2 || model.Len() != 2 || model.Metric() != MetricL2 {
		t.Fatalf("metadata = (%d, %d, %d)", model.Dimension(), model.Len(), model.Metric())
	}
	wantCentroids := [][]float32{{0, .5}, {10, 10.5}}
	if !reflect.DeepEqual(model.Centroids(), wantCentroids) {
		t.Fatalf("centroids = %v, want %v", model.Centroids(), wantCentroids)
	}
	if !slices.Equal(model.Counts(), []int{2, 2}) {
		t.Fatalf("counts = %v", model.Counts())
	}
	if model.Cost() != 1 {
		t.Fatalf("cost = %g, want 1", model.Cost())
	}
	if !model.Converged() || model.Iterations() < 1 || model.Iterations() > options.MaxIterations {
		t.Fatalf("training state = iterations %d, converged %v", model.Iterations(), model.Converged())
	}
	labels, scores, err := model.Classify(context.Background(), vectors, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(labels, []int{0, 0, 1, 1}) || !slices.Equal(scores, []float32{.25, .25, .25, .25}) {
		t.Fatalf("classification = %v, %v", labels, scores)
	}
	label, score, err := model.Nearest([]float32{9, 10})
	if err != nil || label != 1 || score != 1.25 {
		t.Fatalf("nearest = %d, %g, %v", label, score, err)
	}
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
		if err != nil {
			t.Fatal(err)
		}
		options.Workers = 8
		many, err := TrainKMeans(context.Background(), vectors, options)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(one.Centroids(), many.Centroids()) ||
			!slices.Equal(one.Counts(), many.Counts()) || one.Cost() != many.Cost() ||
			one.Iterations() != many.Iterations() || one.Converged() != many.Converged() {
			t.Fatalf("initializer %d differs across workers", initializer)
		}
	}
}

func TestKMeansReservoirSeedFixture(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{0}, {1}, {2}, {3}, {4}, {5}}
	options := DefaultKMeansOptions(3, MetricL2)
	centroids, err := initializeKMeans(context.Background(), vectors, 3, 1, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(centroids, [][]float32{{4}, {5}, {2}}) {
		t.Fatalf("seed-zero reservoir centroids = %v", centroids)
	}
}

func TestKMeansClusterCountCapsAtSamples(t *testing.T) {
	t.Parallel()
	options := DefaultKMeansOptions(10, MetricL2)
	model, err := TrainKMeans(context.Background(), [][]float32{{1}, {2}, {3}}, options)
	if err != nil {
		t.Fatal(err)
	}
	if model.Len() != 3 {
		t.Fatalf("centroid count = %d, want 3", model.Len())
	}
	if !slices.Equal(model.Counts(), []int{1, 1, 1}) {
		t.Fatalf("counts = %v", model.Counts())
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if keep.Len() != 3 || !reflect.DeepEqual(keep.Centroids(), [][]float32{{5}, {0}, {20}}) {
		t.Fatalf("kept model = %v, %v", keep.Centroids(), keep.Counts())
	}

	dropOptions := keepOptions
	dropOptions.EmptyPolicy = KMeansEmptyDrop
	drop, err := TrainKMeans(context.Background(), vectors, dropOptions)
	if err != nil {
		t.Fatal(err)
	}
	if drop.Len() != 2 || !slices.Equal(drop.Counts(), []int{2, 1}) {
		t.Fatalf("dropped model = %v, %v", drop.Centroids(), drop.Counts())
	}

	reseedOptions := keepOptions
	reseedOptions.EmptyPolicy = KMeansEmptyReseedFarthest
	reseed, err := TrainKMeans(context.Background(), vectors, reseedOptions)
	if err != nil {
		t.Fatal(err)
	}
	if reseed.Len() != 3 || !slices.Equal(reseed.Counts(), []int{1, 1, 1}) {
		t.Fatalf("reseeded model = %v, %v", reseed.Centroids(), reseed.Counts())
	}
	centroids := reseed.Centroids()
	if !reflect.DeepEqual(centroids, [][]float32{{5}, {10}, {20}}) {
		t.Fatalf("reseeded centroids = %v", centroids)
	}
}

func TestKMeansInnerProductAndSpherical(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{-2, 0}, {-1, 0}, {1, 0}, {3, 0}}
	options := DefaultKMeansOptions(2, MetricIP)
	options.InitialCentroids = [][]float32{{-1, 0}, {1, 0}}
	options.Spherical = true
	model, err := TrainKMeans(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	centroids := model.Centroids()
	assertFloatSlicesClose(t, centroids[0], []float32{-1, 0}, 1e-7)
	assertFloatSlicesClose(t, centroids[1], []float32{1, 0}, 1e-7)
	label, score, err := model.Nearest([]float32{2, 0})
	if err != nil || label != 1 || score != 2 {
		t.Fatalf("nearest = %d, %g, %v", label, score, err)
	}
	if model.Cost() != -7 {
		t.Fatalf("IP objective = %g, want -7", model.Cost())
	}
}

func TestKMeansModelOwnsState(t *testing.T) {
	t.Parallel()
	vectors := [][]float32{{0}, {10}}
	initial := [][]float32{{0}, {10}}
	options := DefaultKMeansOptions(2, MetricL2)
	options.InitialCentroids = initial
	model, err := TrainKMeans(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	vectors[0][0] = 99
	initial[0][0] = 88
	centroids := model.Centroids()
	centroids[0][0] = 77
	counts := model.Counts()
	counts[0] = 99
	if model.Centroids()[0][0] != 0 || model.Counts()[0] != 1 {
		t.Fatal("model state aliases caller or accessor slices")
	}
}

func TestKMeansValidation(t *testing.T) {
	t.Parallel()
	valid := DefaultKMeansOptions(2, MetricL2)
	if _, err := TrainKMeans(nil, [][]float32{{1}}, valid); err == nil {
		t.Fatal("nil context accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := TrainKMeans(ctx, [][]float32{{1}}, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	if _, err := TrainKMeans(context.Background(), nil, valid); !errors.Is(err, ErrEmptyTrainingSet) {
		t.Fatalf("empty data error = %v", err)
	}
	if _, err := TrainKMeans(context.Background(), [][]float32{{}}, valid); !errors.Is(err, ailego.ErrEmptyVector) {
		t.Fatalf("empty vector error = %v", err)
	}
	if _, err := TrainKMeans(context.Background(), [][]float32{{1}, {1, 2}}, valid); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err := TrainKMeans(context.Background(), [][]float32{{float32(math.NaN())}}, valid); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("non-finite error = %v", err)
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
		if _, err := TrainKMeans(context.Background(), [][]float32{{1}}, options); !errors.Is(err, ErrInvalidKMeansOptions) {
			t.Errorf("options %#v error = %v", options, err)
		}
	}

	badInitial := valid
	badInitial.InitialCentroids = [][]float32{{1}}
	if _, err := TrainKMeans(context.Background(), [][]float32{{1}, {2}}, badInitial); !errors.Is(err, ErrInvalidCentroid) {
		t.Fatalf("initial count error = %v", err)
	}
}

func TestKMeansModelValidation(t *testing.T) {
	t.Parallel()
	var model *KMeansModel
	if _, _, err := model.Nearest([]float32{1}); err == nil {
		t.Fatal("nil model accepted")
	}
	options := DefaultKMeansOptions(1, MetricL2)
	model, err := TrainKMeans(context.Background(), [][]float32{{1}, {2}}, options)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := model.Nearest([]float32{1, 2}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("nearest dimension error = %v", err)
	}
	if _, _, err := model.Classify(nil, nil, 1); err == nil {
		t.Fatal("nil classify context accepted")
	}
	labels, scores, err := model.Classify(context.Background(), nil, 1)
	if err != nil || labels == nil || scores == nil || len(labels) != 0 || len(scores) != 0 {
		t.Fatalf("empty classify = %v, %v, %v", labels, scores, err)
	}
}

func TestKMeansTieBreaksByCentroidIndex(t *testing.T) {
	t.Parallel()
	model := &KMeansModel{
		metric: MetricL2, dimension: 1,
		centroids: [][]float32{{-1}, {1}}, counts: []int{0, 0},
	}
	label, score, err := model.Nearest([]float32{0})
	if err != nil || label != 0 || score != 1 {
		t.Fatalf("tie = %d, %g, %v", label, score, err)
	}
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
		if err != nil {
			if errors.Is(err, ailego.ErrNonFiniteVector) {
				return
			}
			t.Fatal(err)
		}
		if model.Len() == 0 || model.Len() > min(count, clusters) {
			t.Fatalf("centroid count = %d", model.Len())
		}
		labels, scores, err := model.Classify(context.Background(), vectors, 3)
		if err != nil {
			t.Fatal(err)
		}
		if len(labels) != count || len(scores) != count {
			t.Fatalf("classification lengths = %d, %d", len(labels), len(scores))
		}
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
		if _, err := TrainKMeans(context.Background(), vectors, options); err != nil {
			b.Fatal(err)
		}
	}
}
