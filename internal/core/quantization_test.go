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
	"math"
	"slices"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/stretchr/testify/require"
)

func TestQuantizeFP16(t *testing.T) {
	t.Parallel()
	vector, err := QuantizeVector(QuantizationFP16, []float32{1, -2, 1.0003, 0})
	require.NoError(t, err)
	require.Equal(t, QuantizationFP16, vector.Kind())
	require.True(t, vector.Dimension() == 4)
	{
		got := vector.Codes()
		require.True(t, slices.Equal(got, []byte{0x00, 0x3c, 0x00, 0xc0, 0x00, 0x3c, 0x00, 0x00}))
	}

	decoded, err := vector.Decode()
	require.NoError(t, err)
	require.True(t, slices.Equal(decoded, []float32{1, -2, 1, 0}))

	codes := vector.Codes()
	codes[0] = 0xff
	require.True(t, vector.Codes()[0] == 0,
		"Codes exposed mutable storage")
}

func TestQuantizeInt8(t *testing.T) {
	t.Parallel()
	vector, err := QuantizeVector(QuantizationInt8, []float32{-1, 0, 1})
	require.NoError(t, err)
	{
		got := vector.Codes()
		require.True(t, slices.Equal(got, []byte{129, 0, 127}))
	}
	require.InDelta(t, 1.0/127, vector.InverseScale(), 1e-8)
	require.True(t, vector.Offset() == 0)

	decoded, err := vector.Decode()
	require.NoError(t, err)

	assertFloatSlicesClose(t, decoded, []float32{-1, 0, 1}, 1e-6)
}

func TestQuantizeInt4Packing(t *testing.T) {
	t.Parallel()
	vector, err := QuantizeVector(QuantizationInt4, []float32{-1, -.5, 0, 1})
	require.NoError(t, err)
	{
		got := vector.Codes()
		require.True(t, slices.Equal(got, []byte{0xc8, 0x7f}))
	}

	decoded, err := vector.Decode()
	require.NoError(t, err)

	assertFloatSlicesClose(t, decoded, []float32{-1, -.46666667, -.06666667, 1}, 1e-6)
}

func TestQuantizeConstantVector(t *testing.T) {
	t.Parallel()
	for _, kind := range []Quantization{QuantizationInt8, QuantizationInt4} {
		vector, err := QuantizeVector(kind, []float32{3.5, 3.5})
		require.NoError(t, err)
		require.True(t, vector.InverseScale() == 0)
		require.True(t, vector.Offset() == 3.5)

		decoded, err := vector.Decode()
		require.NoError(t, err)
		require.True(t, slices.Equal(decoded, []float32{3.5, 3.5}))
	}
}

func TestQuantizedDistancesMatchDecodedVectors(t *testing.T) {
	t.Parallel()
	leftInput := []float32{-3, -.25, .5, 5}
	rightInput := []float32{2, -1.5, 1.25, 4}
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		left, err := QuantizeVector(kind, leftInput)
		require.NoError(t, err)

		right, err := QuantizeVector(kind, rightInput)
		require.NoError(t, err)

		leftDecoded, _ := left.Decode()
		rightDecoded, _ := right.Decode()
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			want, err := metric.Compute(leftDecoded, rightDecoded)
			require.NoError(t, err)

			got, err := QuantizedDistance(metric, left, right)
			require.NoError(t, err)
			require.InDelta(t, want, got, 2e-5*max(1, math.Abs(float64(want))))
		}
	}
}

func TestQuantizedDistanceZeroVectors(t *testing.T) {
	t.Parallel()
	zero, err := QuantizeVector(QuantizationInt4, []float32{0, 0})
	require.NoError(t, err)

	unit, err := QuantizeVector(QuantizationInt4, []float32{1, 0})
	require.NoError(t, err)

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
		require.NoError(t, err)
		require.Equal(t, test.want, got)
	}
}

func TestQuantizedDistanceToFloat(t *testing.T) {
	t.Parallel()
	candidate, err := QuantizeVector(QuantizationInt8, []float32{-2, 1, 4})
	require.NoError(t, err)

	query := []float32{3, 2, -1}
	got, err := QuantizedDistanceToFloat(MetricL2, candidate, query)
	require.NoError(t, err)

	quantizedQuery, _ := QuantizeVector(QuantizationInt8, query)
	want, _ := QuantizedDistance(MetricL2, candidate, quantizedQuery)
	require.Equal(t, want, got)
}

func TestQuantizeBatch(t *testing.T) {
	t.Parallel()
	input := [][]float32{{-1, 1}, {2, 4}, {9, 9}}
	got, err := QuantizeBatch(context.Background(), QuantizationInt4, input, 2)
	require.NoError(t, err)
	require.Len(t, got, len(input))

	for index, vector := range got {
		decoded, err := vector.Decode()
		require.NoError(t, err)

		assertFloatSlicesClose(t, decoded, input[index], 1e-6)
	}
	input[0][0] = 100
	decoded, _ := got[0].Decode()
	require.Equal(t, float32(-1), decoded[0],
		"batch result aliases caller input")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := QuantizeBatch(ctx, QuantizationInt8, input, 2)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := QuantizeBatch(nil, QuantizationInt8, input, 2)
		require.Error(t, err,
			"nil context accepted")
	}
}

func TestQuantizationValidation(t *testing.T) {
	t.Parallel()
	{
		_, err := QuantizeVector(0, []float32{1})
		require.ErrorIs(t, err, ErrInvalidQuantization)
	}
	{
		_, err := QuantizeVector(QuantizationFP16, nil)
		require.ErrorIs(t, err, mathutil.ErrEmptyVector)
	}

	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		{
			_, err := QuantizeVector(QuantizationInt8, []float32{value})
			require.ErrorIs(t, err, mathutil.ErrNonFiniteVector)
		}
	}
	{
		_, err := QuantizeVector(QuantizationFP16, []float32{math.MaxFloat32})
		require.ErrorIs(t, err, ErrQuantizationOverflow)
	}
	{
		_, err := QuantizeVector(QuantizationInt4, []float32{1})
		require.ErrorIs(t, err, ErrOddInt4Dimension)
	}

	int8Vector, _ := QuantizeVector(QuantizationInt8, []float32{1, 2})
	int4Vector, _ := QuantizeVector(QuantizationInt4, []float32{1, 2})
	{
		_, err := QuantizedDistance(MetricL2, int8Vector, int4Vector)
		require.ErrorIs(t, err, ErrInvalidQuantizedVector)
	}

	short, _ := QuantizeVector(QuantizationInt8, []float32{1})
	{
		_, err := QuantizedDistance(MetricL2, int8Vector, short)
		require.ErrorIs(t, err, mathutil.ErrDimensionMismatch)
	}

	corrupt := int8Vector
	corrupt.codes = corrupt.codes[:1]
	{
		_, err := corrupt.Decode()
		require.ErrorIs(t, err, ErrInvalidQuantizedVector)
	}
	{
		_, err := QuantizedDistance(0, int8Vector, int8Vector)
		require.Error(t, err,
			"invalid metric accepted")
	}
	{
		_, err := QuantizedDistanceToFloat(MetricL2, int8Vector, []float32{1})
		require.ErrorIs(t, err, mathutil.ErrDimensionMismatch)
	}
	require.Equal(t, []byte(nil), QuantizedVector{}.Codes(),
		"zero vector codes should be nil")
}

func FuzzQuantizedVector(f *testing.F) {
	f.Add(uint8(2), float32(-1), float32(0), float32(1), float32(2))
	f.Add(uint8(3), float32(3.5), float32(3.5), float32(3.5), float32(3.5))
	f.Fuzz(func(t *testing.T, rawKind uint8, a, b, c, d float32) {
		kind := Quantization(rawKind%3 + 1)
		input := []float32{a, b, c, d}
		vector, err := QuantizeVector(kind, input)
		if errors.Is(err, mathutil.ErrNonFiniteVector) || errors.Is(err, ErrQuantizationOverflow) {
			return
		}
		require.NoError(t, err)
		decoded, err := vector.Decode()
		require.NoError(t, err)

		for _, value := range decoded {
			require.False(t, math.IsNaN(float64(value)))
			require.False(t, math.IsInf(float64(value), 0))
		}
		for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
			{
				_, err := QuantizedDistance(metric, vector, vector)
				require.False(t, err != nil && !errors.Is(err, mathutil.ErrNonFiniteVector))
			}
		}
	})
}

func assertFloatSlicesClose(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	require.Len(t, got, len(want))

	for index := range got {
		require.InDelta(t, want[index], got[index], tolerance)
	}
}
