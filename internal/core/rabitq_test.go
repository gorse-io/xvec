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

func TestRaBitQPinnedLibraryFixture(t *testing.T) {
	centroid := make([]float32, 64)
	vector := make([]float32, 64)
	for index := range vector {
		vector[index] = float32(int(index*17%29)-14) * .125
		centroid[index] = float32(int(index*5%11)-5) * .05
	}
	wantValues := []uint16{
		61, 64, 61, 65, 62, 67, 63, 61, 65, 62, 66, 64, 60, 64, 62, 65,
		63, 66, 64, 61, 66, 62, 67, 0, 60, 65, 61, 66, 63, 61, 64, 62,
		65, 63, 67, 63, 61, 65, 62, 66, 64, 60, 65, 62, 66, 63, 66, 64,
		61, 65, 62, 67, 63, 61, 65, 62, 66, 63, 60, 64, 62, 65, 63, 66,
	}
	tests := []struct {
		metric                                Metric
		coarseAdd, coarseRescale, coarseError float64
		fullAdd, fullRescale                  float64
	}{
		{MetricL2, 70.4307632, -4.88890362, 2.37279153, 69.6409988, -1.05726922},
		{MetricIP, 1.12069368, -2.44445181, 1.18639576, .725809216, -.528634608},
	}
	for _, test := range tests {
		code, err := quantizeRaBitQVector(vector, centroid, 0, 7, 15.75, test.metric == MetricIP)
		if err != nil {
			t.Fatal(err)
		}
		code.modelFingerprint = 1
		values, err := code.QuantizedValues()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(values, wantValues) {
			t.Fatalf("metric %d values = %v", test.metric, values)
		}
		assertRaBitQClose(t, "coarse add", code.coarseAdd, test.coarseAdd, 2e-6)
		assertRaBitQClose(t, "coarse rescale", code.coarseRescale, test.coarseRescale, 2e-6)
		assertRaBitQClose(t, "coarse error", code.coarseError, test.coarseError, 2e-6)
		assertRaBitQClose(t, "full add", code.fullAdd, test.fullAdd, 2e-6)
		assertRaBitQClose(t, "full rescale", code.fullRescale, test.fullRescale, 2e-6)
	}
}

func TestRaBitQTrainingDeterministicAcrossWorkers(t *testing.T) {
	vectors := raBitQTestVectors(120, 70)
	options := DefaultRaBitQOptions(MetricL2)
	options.TotalBits = 4
	options.Clusters = 6
	options.SampleCount = 90
	options.MaxIterations = 8
	options.Seed = 0x123456789abcdef
	options.Workers = 1
	serial, err := TrainRaBitQ(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	options.Workers = 4
	parallel, err := TrainRaBitQ(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(serial.State(), parallel.State()) {
		t.Fatal("RaBitQ model changed across worker counts")
	}
	if serial.Dimension() != 70 || serial.PaddedDimension() != 128 || serial.Len() != 6 || serial.TotalBits() != 4 {
		t.Fatalf("model metadata = dim %d/%d clusters %d bits %d", serial.Dimension(), serial.PaddedDimension(), serial.Len(), serial.TotalBits())
	}
	restored, err := RestoreRaBitQModel(serial.State())
	if err != nil {
		t.Fatal(err)
	}
	left, err := serial.Encode(vectors[37])
	if err != nil {
		t.Fatal(err)
	}
	right, err := restored.Encode(vectors[37])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(left, right) {
		t.Fatal("restored model produced a different code")
	}
	state := serial.State()
	state.Centroids[0][0]++
	state.RotationSigns[0]++
	if reflect.DeepEqual(state, serial.State()) {
		t.Fatal("State exposed mutable model storage")
	}
}

func TestRaBitQFullEstimateImprovesCoarseQuality(t *testing.T) {
	vectors := raBitQGaussianVectors(320, 64, 101)
	options := DefaultRaBitQOptions(MetricL2)
	options.Clusters = 8
	options.TotalBits = 7
	options.MaxIterations = 10
	options.Seed = 77
	model, err := TrainRaBitQ(context.Background(), vectors, options)
	if err != nil {
		t.Fatal(err)
	}
	codes, err := model.EncodeBatch(context.Background(), vectors, 4)
	if err != nil {
		t.Fatal(err)
	}
	queryVector := raBitQGaussianVectors(1, 64, 202)[0]
	for index := range queryVector {
		queryVector[index] += float32(index%5) * .03125
	}
	query, err := model.PrepareQuery(queryVector)
	if err != nil {
		t.Fatal(err)
	}
	var coarseError, fullError float64
	var lowerCovered, upperCovered int
	for index, code := range codes {
		coarse, err := query.EstimateCoarse(code)
		if err != nil {
			t.Fatal(err)
		}
		full, err := query.Estimate(code)
		if err != nil {
			t.Fatal(err)
		}
		exact, err := MetricL2.Compute(vectors[index], queryVector)
		if err != nil {
			t.Fatal(err)
		}
		coarseError += math.Abs(float64(coarse.Distance - exact))
		fullError += math.Abs(float64(full.Distance - exact))
		if exact >= full.LowerBound {
			lowerCovered++
		}
		if exact <= full.UpperBound {
			upperCovered++
		}
	}
	if fullError >= coarseError*.35 {
		t.Fatalf("full absolute error %.6f is not materially below coarse %.6f", fullError, coarseError)
	}
	if lowerCoverage, upperCoverage := float64(lowerCovered)/float64(len(codes)), float64(upperCovered)/float64(len(codes)); lowerCoverage < .80 || upperCoverage < .80 {
		t.Fatalf("full probabilistic-envelope coverage = lower %.3f, upper %.3f, want both >= .80", lowerCoverage, upperCoverage)
	}
}

func TestRaBitQIPCosineAndZeroResidual(t *testing.T) {
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		vectors := raBitQTestVectors(80, 64)
		options := DefaultRaBitQOptions(metric)
		options.TotalBits = 5
		options.Clusters = 4
		options.MaxIterations = 6
		options.Seed = 19
		model, err := TrainRaBitQ(context.Background(), vectors, options)
		if err != nil {
			t.Fatalf("metric %d train: %v", metric, err)
		}
		code, err := model.Encode(vectors[23])
		if err != nil {
			t.Fatal(err)
		}
		query, err := model.PrepareQuery(vectors[23])
		if err != nil {
			t.Fatal(err)
		}
		estimate, err := query.Estimate(code)
		if err != nil {
			t.Fatal(err)
		}
		exact, err := metric.Compute(vectors[23], vectors[23])
		if err != nil {
			t.Fatal(err)
		}
		if metric == MetricIP {
			exact = 1 - exact
		}
		if math.Abs(float64(estimate.Distance-exact)) > .08 {
			t.Fatalf("metric %d self estimate %v, exact %v", metric, estimate.Distance, exact)
		}
	}

	centroid := raBitQTestVectors(1, 64)[0]
	model, err := RestoreRaBitQModel(RaBitQModelState{
		Dimension: 64, Metric: MetricL2, TotalBits: 1,
		Centroids: [][]float32{centroid}, RotationSigns: make([]byte, 32),
	})
	if err != nil {
		t.Fatal(err)
	}
	code, err := model.Encode(centroid)
	if err != nil {
		t.Fatal(err)
	}
	query, _ := model.PrepareQuery(centroid)
	estimate, err := query.Estimate(code)
	if err != nil || estimate.Distance != 0 || estimate.LowerBound != 0 || estimate.UpperBound != 0 {
		t.Fatalf("zero residual estimate = %#v, %v", estimate, err)
	}
}

func TestRaBitQBitWidthsAndDimensionBoundaries(t *testing.T) {
	for _, dimension := range []int{64, 65, 127, 128, 129, MaxRaBitQDimension} {
		centroid := make([]float32, dimension)
		vector := raBitQTestVectors(1, dimension)[0]
		padded := roundUpRaBitQDimension(dimension)
		for totalBits := MinRaBitQTotalBits; totalBits <= MaxRaBitQTotalBits; totalBits++ {
			extraScale := float64(0)
			if totalBits > 1 {
				extraScale = 15.75
			}
			model, err := RestoreRaBitQModel(RaBitQModelState{
				Dimension: dimension, Metric: MetricL2, TotalBits: totalBits,
				Centroids: [][]float32{centroid}, RotationSigns: make([]byte, 4*padded/8),
				ExtraScale: extraScale,
			})
			if err != nil {
				t.Fatalf("restore dimension %d bits %d: %v", dimension, totalBits, err)
			}
			code, err := model.Encode(vector)
			if err != nil {
				t.Fatalf("encode dimension %d bits %d: %v", dimension, totalBits, err)
			}
			if len(code.BinaryCode()) != padded/8 || len(code.ExtraCode()) != padded*(totalBits-1)/8 {
				t.Fatalf("dimension %d bits %d code lengths = %d/%d", dimension, totalBits, len(code.BinaryCode()), len(code.ExtraCode()))
			}
			query, err := model.PrepareQuery(vector)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := query.Estimate(code); err != nil {
				t.Fatalf("estimate dimension %d bits %d: %v", dimension, totalBits, err)
			}
		}
	}
}

func TestRaBitQValidationCancellationAndOwnership(t *testing.T) {
	invalidOptions := []RaBitQOptions{
		{},
		{Metric: MetricMIPSL2, TotalBits: 7, Clusters: 1, MaxIterations: 1},
		{Metric: MetricL2, TotalBits: 10, Clusters: 1, MaxIterations: 1},
		{Metric: MetricL2, TotalBits: 7, Clusters: 0, MaxIterations: 1},
		{Metric: MetricL2, TotalBits: 7, Clusters: 1, SampleCount: -1, MaxIterations: 1},
		{Metric: MetricL2, TotalBits: 7, Clusters: 1, MaxIterations: 1, Workers: -1},
	}
	for _, options := range invalidOptions {
		if err := options.Validate(); !errors.Is(err, ErrInvalidRaBitQOptions) {
			t.Fatalf("invalid options %#v: %v", options, err)
		}
	}
	options := DefaultRaBitQOptions(MetricL2)
	if _, err := TrainRaBitQ(nil, raBitQTestVectors(2, 64), options); err == nil {
		t.Fatal("nil training context succeeded")
	}
	if _, err := TrainRaBitQ(context.Background(), raBitQTestVectors(2, 63), options); !errors.Is(err, ErrInvalidRaBitQOptions) {
		t.Fatalf("dimension error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := TrainRaBitQ(canceled, raBitQTestVectors(2, 64), options); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled train error = %v", err)
	}

	model := fixtureRaBitQModel(t, MetricL2)
	vector := raBitQTestVectors(1, 64)[0]
	code, err := model.Encode(vector)
	if err != nil {
		t.Fatal(err)
	}
	binaryCode := code.BinaryCode()
	binaryCode[0] ^= 0xff
	if slices.Equal(binaryCode, code.BinaryCode()) {
		t.Fatal("BinaryCode exposed mutable storage")
	}
	if _, err := model.Encode(vector[:63]); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("encode dimension error = %v", err)
	}
	if _, err := model.EncodeBatch(nil, [][]float32{vector}, 1); err == nil {
		t.Fatal("nil batch context succeeded")
	}
	if _, err := model.EncodeBatch(canceled, [][]float32{vector}, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled batch error = %v", err)
	}

	other := fixtureRaBitQModelWithSigns(t, MetricL2, 1)
	query, _ := other.PrepareQuery(vector)
	if _, err := query.Estimate(code); !errors.Is(err, ErrRaBitQModelMismatch) {
		t.Fatalf("model mismatch error = %v", err)
	}
	badState := model.State()
	badState.RotationSigns = badState.RotationSigns[:len(badState.RotationSigns)-1]
	if _, err := RestoreRaBitQModel(badState); !errors.Is(err, ErrInvalidRaBitQModel) {
		t.Fatalf("bad state error = %v", err)
	}
}

func FuzzRaBitQEncodeEstimate(f *testing.F) {
	f.Add([]byte("portable-rabitq-fixture"))
	f.Add(make([]byte, 64))
	model := fixtureRaBitQModel(f, MetricL2)
	f.Fuzz(func(t *testing.T, input []byte) {
		vector := make([]float32, 64)
		for index := range vector {
			if len(input) > 0 {
				vector[index] = float32(int8(input[index%len(input)])) / 16
			}
		}
		code, err := model.Encode(vector)
		if err != nil {
			t.Fatal(err)
		}
		query, err := model.PrepareQuery(vector)
		if err != nil {
			t.Fatal(err)
		}
		estimate, err := query.Estimate(code)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(float64(estimate.Distance)) || math.IsInf(float64(estimate.Distance), 0) {
			t.Fatalf("non-finite estimate %#v", estimate)
		}
	})
}

func BenchmarkRaBitQEncodeEstimate(b *testing.B) {
	model := fixtureRaBitQModel(b, MetricL2)
	vector := raBitQTestVectors(1, 64)[0]
	query, err := model.PrepareQuery(vector)
	if err != nil {
		b.Fatal(err)
	}
	code, err := model.Encode(vector)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("Encode", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := model.Encode(vector); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("Estimate", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := query.Estimate(code); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func fixtureRaBitQModel(t testing.TB, metric Metric) *RaBitQModel {
	t.Helper()
	return fixtureRaBitQModelWithSigns(t, metric, 0)
}

func fixtureRaBitQModelWithSigns(t testing.TB, metric Metric, sign byte) *RaBitQModel {
	t.Helper()
	centroid := make([]float32, 64)
	for index := range centroid {
		centroid[index] = float32(int(index*5%11)-5) * .05
	}
	signs := make([]byte, 32)
	for index := range signs {
		signs[index] = sign
	}
	model, err := RestoreRaBitQModel(RaBitQModelState{
		Dimension: 64, Metric: metric, TotalBits: 7,
		Centroids: [][]float32{centroid}, RotationSigns: signs, ExtraScale: 15.75,
	})
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func raBitQTestVectors(count, dimension int) [][]float32 {
	vectors := make([][]float32, count)
	for row := range vectors {
		vectors[row] = make([]float32, dimension)
		for column := range vectors[row] {
			vectors[row][column] = float32(int((row*37+column*17+row*column*3)%101)-50) / 17
		}
	}
	return vectors
}

func raBitQGaussianVectors(count, dimension int, seed uint64) [][]float32 {
	random := splitMix64{state: seed}
	vectors := make([][]float32, count)
	for row := range vectors {
		vectors[row] = make([]float32, dimension)
		for column := 0; column < dimension; column += 2 {
			left, right := raBitQNormalPair(&random)
			vectors[row][column] = float32(left)
			if column+1 < dimension {
				vectors[row][column+1] = float32(right)
			}
		}
	}
	return vectors
}

func assertRaBitQClose(t testing.TB, name string, got, want, tolerance float64) {
	t.Helper()
	difference := math.Abs(got - want)
	if difference > tolerance*max(1, math.Abs(want)) {
		t.Fatalf("%s = %.9g, want %.9g (difference %.9g)", name, got, want, difference)
	}
}
