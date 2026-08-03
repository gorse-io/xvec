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

func TestQuantizeFP16(t *testing.T) {
	t.Parallel()
	vector, err := QuantizeVector(QuantizationFP16, []float32{1, -2, 1.0003, 0})
	if err != nil {
		t.Fatal(err)
	}
	if vector.Kind() != QuantizationFP16 || vector.Dimension() != 4 {
		t.Fatalf("metadata = (%d, %d)", vector.Kind(), vector.Dimension())
	}
	if got := vector.Codes(); !slices.Equal(got, []byte{0x00, 0x3c, 0x00, 0xc0, 0x00, 0x3c, 0x00, 0x00}) {
		t.Fatalf("codes = % x", got)
	}
	decoded, err := vector.Decode()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(decoded, []float32{1, -2, 1, 0}) {
		t.Fatalf("decoded = %v", decoded)
	}
	codes := vector.Codes()
	codes[0] = 0xff
	if vector.Codes()[0] != 0 {
		t.Fatal("Codes exposed mutable storage")
	}
}

func TestQuantizeInt8(t *testing.T) {
	t.Parallel()
	vector, err := QuantizeVector(QuantizationInt8, []float32{-1, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got := vector.Codes(); !slices.Equal(got, []byte{129, 0, 127}) {
		t.Fatalf("codes = %v", got)
	}
	if difference := math.Abs(float64(vector.InverseScale() - 1.0/127)); difference > 1e-8 {
		t.Fatalf("inverse scale = %g", vector.InverseScale())
	}
	if vector.Offset() != 0 {
		t.Fatalf("offset = %g", vector.Offset())
	}
	decoded, err := vector.Decode()
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesClose(t, decoded, []float32{-1, 0, 1}, 1e-6)
}

func TestQuantizeInt4Packing(t *testing.T) {
	t.Parallel()
	vector, err := QuantizeVector(QuantizationInt4, []float32{-1, -.5, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got := vector.Codes(); !slices.Equal(got, []byte{0xc8, 0x7f}) {
		t.Fatalf("packed codes = % x, want c8 7f", got)
	}
	decoded, err := vector.Decode()
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesClose(t, decoded, []float32{-1, -.46666667, -.06666667, 1}, 1e-6)
}

func TestQuantizeConstantVector(t *testing.T) {
	t.Parallel()
	for _, kind := range []Quantization{QuantizationInt8, QuantizationInt4} {
		vector, err := QuantizeVector(kind, []float32{3.5, 3.5})
		if err != nil {
			t.Fatal(err)
		}
		if vector.InverseScale() != 0 || vector.Offset() != 3.5 {
			t.Fatalf("kind %d params = (%g, %g)", kind, vector.InverseScale(), vector.Offset())
		}
		decoded, err := vector.Decode()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(decoded, []float32{3.5, 3.5}) {
			t.Fatalf("kind %d decoded = %v", kind, decoded)
		}
	}
}

func TestQuantizedDistancesMatchDecodedVectors(t *testing.T) {
	t.Parallel()
	leftInput := []float32{-3, -.25, .5, 5}
	rightInput := []float32{2, -1.5, 1.25, 4}
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		left, err := QuantizeVector(kind, leftInput)
		if err != nil {
			t.Fatal(err)
		}
		right, err := QuantizeVector(kind, rightInput)
		if err != nil {
			t.Fatal(err)
		}
		leftDecoded, _ := left.Decode()
		rightDecoded, _ := right.Decode()
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			want, err := metric.Compute(leftDecoded, rightDecoded)
			if err != nil {
				t.Fatal(err)
			}
			got, err := QuantizedDistance(metric, left, right)
			if err != nil {
				t.Fatalf("kind %d metric %d: %v", kind, metric, err)
			}
			if difference := math.Abs(float64(got - want)); difference > 2e-5*max(1, math.Abs(float64(want))) {
				t.Fatalf("kind %d metric %d score = %g, decoded score = %g", kind, metric, got, want)
			}
		}
	}
}

func TestQuantizedDistanceZeroVectors(t *testing.T) {
	t.Parallel()
	zero, err := QuantizeVector(QuantizationInt4, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	unit, err := QuantizeVector(QuantizationInt4, []float32{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		metric Metric
		left   QuantizedVector
		right  QuantizedVector
		want   float32
	}{
		{MetricCosine, zero, zero, 0},
		{MetricCosine, zero, unit, 1},
		{MetricMIPSL2, zero, zero, 0},
		{MetricMIPSL2, zero, unit, 2},
	} {
		got, err := QuantizedDistance(test.metric, test.left, test.right)
		if err != nil || got != test.want {
			t.Fatalf("metric %d score = %g, err = %v, want %g", test.metric, got, err, test.want)
		}
	}
}

func TestQuantizedDistanceToFloat(t *testing.T) {
	t.Parallel()
	candidate, err := QuantizeVector(QuantizationInt8, []float32{-2, 1, 4})
	if err != nil {
		t.Fatal(err)
	}
	query := []float32{3, 2, -1}
	got, err := QuantizedDistanceToFloat(MetricL2, candidate, query)
	if err != nil {
		t.Fatal(err)
	}
	quantizedQuery, _ := QuantizeVector(QuantizationInt8, query)
	want, _ := QuantizedDistance(MetricL2, candidate, quantizedQuery)
	if got != want {
		t.Fatalf("score = %g, want %g", got, want)
	}
}

func TestQuantizeBatch(t *testing.T) {
	t.Parallel()
	input := [][]float32{{-1, 1}, {2, 4}, {9, 9}}
	got, err := QuantizeBatch(context.Background(), QuantizationInt4, input, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(input) {
		t.Fatalf("batch length = %d", len(got))
	}
	for index, vector := range got {
		decoded, err := vector.Decode()
		if err != nil {
			t.Fatal(err)
		}
		assertFloatSlicesClose(t, decoded, input[index], 1e-6)
	}
	input[0][0] = 100
	decoded, _ := got[0].Decode()
	if decoded[0] != -1 {
		t.Fatal("batch result aliases caller input")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := QuantizeBatch(ctx, QuantizationInt8, input, 2); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled batch error = %v", err)
	}
	if _, err := QuantizeBatch(nil, QuantizationInt8, input, 2); err == nil {
		t.Fatal("nil context accepted")
	}
}

func TestQuantizationValidation(t *testing.T) {
	t.Parallel()
	if _, err := QuantizeVector(0, []float32{1}); !errors.Is(err, ErrInvalidQuantization) {
		t.Fatalf("invalid kind error = %v", err)
	}
	if _, err := QuantizeVector(QuantizationFP16, nil); !errors.Is(err, ailego.ErrEmptyVector) {
		t.Fatalf("empty error = %v", err)
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := QuantizeVector(QuantizationInt8, []float32{value}); !errors.Is(err, ailego.ErrNonFiniteVector) {
			t.Fatalf("non-finite error = %v", err)
		}
	}
	if _, err := QuantizeVector(QuantizationFP16, []float32{math.MaxFloat32}); !errors.Is(err, ErrQuantizationOverflow) {
		t.Fatalf("FP16 overflow error = %v", err)
	}
	if _, err := QuantizeVector(QuantizationInt4, []float32{1}); !errors.Is(err, ErrOddInt4Dimension) {
		t.Fatalf("odd INT4 error = %v", err)
	}

	int8Vector, _ := QuantizeVector(QuantizationInt8, []float32{1, 2})
	int4Vector, _ := QuantizeVector(QuantizationInt4, []float32{1, 2})
	if _, err := QuantizedDistance(MetricL2, int8Vector, int4Vector); !errors.Is(err, ErrInvalidQuantizedVector) {
		t.Fatalf("encoding mismatch error = %v", err)
	}
	short, _ := QuantizeVector(QuantizationInt8, []float32{1})
	if _, err := QuantizedDistance(MetricL2, int8Vector, short); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension mismatch error = %v", err)
	}
	corrupt := int8Vector
	corrupt.codes = corrupt.codes[:1]
	if _, err := corrupt.Decode(); !errors.Is(err, ErrInvalidQuantizedVector) {
		t.Fatalf("corrupt code error = %v", err)
	}
	if _, err := QuantizedDistance(0, int8Vector, int8Vector); err == nil {
		t.Fatal("invalid metric accepted")
	}
	if _, err := QuantizedDistanceToFloat(MetricL2, int8Vector, []float32{1}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("query dimension error = %v", err)
	}
	if !reflect.DeepEqual(QuantizedVector{}.Codes(), []byte(nil)) {
		t.Fatal("zero vector codes should be nil")
	}
}

func FuzzQuantizedVector(f *testing.F) {
	f.Add(uint8(2), float32(-1), float32(0), float32(1), float32(2))
	f.Add(uint8(3), float32(3.5), float32(3.5), float32(3.5), float32(3.5))
	f.Fuzz(func(t *testing.T, rawKind uint8, a, b, c, d float32) {
		kind := Quantization(rawKind%3 + 1)
		input := []float32{a, b, c, d}
		vector, err := QuantizeVector(kind, input)
		if err != nil {
			if errors.Is(err, ailego.ErrNonFiniteVector) || errors.Is(err, ErrQuantizationOverflow) {
				return
			}
			t.Fatal(err)
		}
		decoded, err := vector.Decode()
		if err != nil {
			t.Fatal(err)
		}
		for _, value := range decoded {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("non-finite decoded value %g", value)
			}
		}
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			if _, err := QuantizedDistance(metric, vector, vector); err != nil && !errors.Is(err, ailego.ErrNonFiniteVector) {
				t.Fatalf("metric %d: %v", metric, err)
			}
		}
	})
}

func assertFloatSlicesClose(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for index := range got {
		if difference := math.Abs(float64(got[index] - want[index])); difference > tolerance {
			t.Fatalf("element %d = %g, want %g (difference %g)", index, got[index], want[index], difference)
		}
	}
}
