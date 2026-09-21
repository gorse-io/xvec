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

package mathutil

import (
	"math"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/utility"
	"github.com/stretchr/testify/require"
)

func TestDenseMetricsMatchPinnedBaseline(t *testing.T) {
	t.Parallel()

	type metricCase struct {
		name      string
		compute   DenseDistance
		left      []float32
		right     []float32
		expected  float32
		tolerance float32
	}
	cases := []metricCase{
		{name: "l2 same", compute: L2Squared, left: []float32{1, 0}, right: []float32{1, 0}, expected: 0},
		{name: "l2 opposite", compute: L2Squared, left: []float32{1, 0}, right: []float32{-1, 0}, expected: 4},
		{name: "l2 general", compute: L2Squared, left: []float32{1, 2, 3}, right: []float32{4, 6, 3}, expected: 25},
		{name: "ip same", compute: InnerProduct, left: []float32{1, 0}, right: []float32{1, 0}, expected: 1},
		{name: "ip opposite", compute: InnerProduct, left: []float32{1, 0}, right: []float32{-1, 0}, expected: -1},
		{name: "ip general", compute: InnerProduct, left: []float32{1, 2, 3}, right: []float32{4, 6, 3}, expected: 25},
		{name: "cosine same", compute: CosineDistance, left: []float32{1, 0}, right: []float32{1, 0}, expected: 0},
		{name: "cosine orthogonal", compute: CosineDistance, left: []float32{1, 0}, right: []float32{0, 1}, expected: 1},
		{name: "cosine opposite", compute: CosineDistance, left: []float32{1, 0}, right: []float32{-1, 0}, expected: 2},
		{name: "cosine scaled", compute: CosineDistance, left: []float32{2, 0}, right: []float32{1, 0}, expected: 0},
		{name: "mips same", compute: MIPSL2Squared, left: []float32{1, 0}, right: []float32{1, 0}, expected: 0},
		{name: "mips orthogonal", compute: MIPSL2Squared, left: []float32{1, 0}, right: []float32{0, 1}, expected: 2},
		{name: "mips opposite", compute: MIPSL2Squared, left: []float32{1, 0}, right: []float32{-1, 0}, expected: 4},
		{name: "mips scaled", compute: MIPSL2Squared, left: []float32{2, 0}, right: []float32{1, 0}, expected: 1},
		{name: "cosine non-integer", compute: CosineDistance, left: []float32{1, 1}, right: []float32{1, 0}, expected: float32(1 - 1/math.Sqrt2), tolerance: 1e-6},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			actual := testCase.compute(testCase.left, testCase.right)
			tolerance := testCase.tolerance
			if tolerance == 0 {
				tolerance = 1e-7
			}
			require.InDelta(t, testCase.expected, actual, float64(tolerance))
		})
	}
}

func TestFP16DenseMetricsMatchPinnedBaseline(t *testing.T) {
	t.Parallel()

	encode := func(values ...float32) []uint16 {
		encoded := make([]uint16, len(values))
		for index, value := range values {
			encoded[index] = utility.Float32ToFloat16Bits(value)
		}
		return encoded
	}
	left := encode(1, 2, 3)
	right := encode(4, 6, 3)
	require.Equal(t, float32(25), L2SquaredFP16(left, right))
	require.Equal(t, float32(25), InnerProductFP16(left, right))
	require.InDelta(t, CosineDistance(
		[]float32{1, 2, 3}, []float32{4, 6, 3},
	), CosineDistanceFP16(left, right), 1e-6)
	require.InDelta(t, MIPSL2Squared(
		[]float32{1, 2, 3}, []float32{4, 6, 3},
	), MIPSL2SquaredFP16(left, right), 1e-6)
	require.InDelta(t, float32(math.Sqrt(14)), L2MagnitudeFP16(left), 1e-6)
	require.InDelta(t, CosineDistanceFP16(left, right), CosineDistanceWithMagnitudesFP16(
		left, right, L2MagnitudeFP16(left), L2MagnitudeFP16(right),
	), 1e-6)
}

func TestFP16DenseMetricZeroVectors(t *testing.T) {
	t.Parallel()

	zero := []uint16{0, 0}
	unit := []uint16{utility.Float32ToFloat16Bits(1), 0}
	require.Equal(t, float32(0), CosineDistanceFP16(zero, zero))
	require.Equal(t, float32(1), CosineDistanceFP16(zero, unit))
	require.Equal(t, float32(1), CosineDistanceFP16(unit, zero))
	require.Equal(t, float32(0), MIPSL2SquaredFP16(zero, zero))
	require.Equal(t, float32(2), MIPSL2SquaredFP16(zero, unit))
	require.Equal(t, float32(2), MIPSL2SquaredFP16(unit, zero))
}

func TestFP16DenseDistancesDoNotAllocate(t *testing.T) {
	left := []uint16{0x3266, 0x3b33, 0xb666, 0x399a}
	right := []uint16{0x34cd, 0x3800, 0x3a66, 0xae66}
	for _, test := range []struct {
		name     string
		distance DenseDistanceFP16
	}{
		{name: "l2", distance: L2SquaredFP16},
		{name: "inner product", distance: InnerProductFP16},
		{name: "cosine", distance: CosineDistanceFP16},
		{name: "mips-l2", distance: MIPSL2SquaredFP16},
	} {
		t.Run(test.name, func(t *testing.T) {
			require.Zero(t, testing.AllocsPerRun(100, func() {
				benchmarkDenseScore = test.distance(left, right)
			}))
		})
	}
}

func TestFP16MagnitudeClampsNegativeRoundoff(t *testing.T) {
	original := kernelsFP16.dot
	kernelsFP16.dot = func(_, _ []uint16) float32 { return -1 }
	t.Cleanup(func() { kernelsFP16.dot = original })
	require.Zero(t, L2MagnitudeFP16([]uint16{0}))
}

func TestDenseMetricZeroVectors(t *testing.T) {
	t.Parallel()

	zero := []float32{0, 0}
	unit := []float32{1, 0}
	assertScore(t, CosineDistance, zero, zero, 0)
	assertScore(t, CosineDistance, zero, unit, 1)
	assertScore(t, CosineDistance, unit, zero, 1)
	assertScore(t, MIPSL2Squared, zero, zero, 0)
	assertScore(t, MIPSL2Squared, zero, unit, 2)
	assertScore(t, MIPSL2Squared, unit, zero, 2)
}

func TestUncheckedDenseDistanceAPI(t *testing.T) {
	t.Parallel()

	var distance DenseDistance = L2Squared
	require.Equal(t, float32(25), distance([]float32{1, 2, 3}, []float32{4, 6, 3}))
	require.Equal(t, float32(5), L2Magnitude([]float32{3, 4}))
	require.Equal(t, float32(0), CosineDistanceWithMagnitudes(
		[]float32{1, 0}, []float32{2, 0}, 1, 2,
	))
	large := []float32{math.MaxFloat32, math.MaxFloat32}
	require.True(t, math.IsInf(float64(InnerProduct(large, large)), 1))
}

func TestValidateDense(t *testing.T) {
	t.Parallel()

	require.NoError(t, ValidateDense([]float32{1, 2}, 2))
	require.ErrorIs(t, ValidateDense([]float32{1}, 2), ErrDimensionMismatch)
	require.ErrorIs(t, ValidateDense(nil, 0), ErrEmptyVector)
	require.ErrorIs(t, ValidateDense([]float32{float32(math.NaN())}, 1), ErrNonFiniteVector)
	require.ErrorIs(t, ValidateDense([]float32{float32(math.Inf(1))}, 1), ErrNonFiniteVector)
}

func TestCosineDistanceAvoidsNormProductOverflowAndUnderflow(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		left     []float32
		right    []float32
		expected float32
	}{
		{name: "large identical vectors", left: []float32{1e10, 1e10}, right: []float32{1e10, 1e10}, expected: 0},
		{name: "tiny orthogonal vectors", left: []float32{1e-20, 0}, right: []float32{0, 1e-20}, expected: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.InDelta(t, test.expected, CosineDistance(test.left, test.right), 1e-6)
		})
	}
}

func TestDenseMetricsUseFloat32Accumulation(t *testing.T) {
	t.Parallel()

	left := []float32{1e8, 1, -1e8}
	right := []float32{1, 1, 1}
	require.Zero(t, InnerProduct(left, right))
}

func TestDenseDistanceKernelsDoNotAllocate(t *testing.T) {
	left := []float32{0.2, 0.9, -0.4, 0.7}
	right := []float32{0.3, 0.5, 0.8, -0.1}
	for _, test := range []struct {
		name     string
		distance DenseDistance
	}{
		{name: "l2", distance: L2Squared},
		{name: "inner product", distance: InnerProduct},
		{name: "cosine", distance: CosineDistance},
		{name: "mips-l2", distance: MIPSL2Squared},
	} {
		t.Run(test.name, func(t *testing.T) {
			require.Zero(t, testing.AllocsPerRun(100, func() {
				benchmarkDenseScore = test.distance(left, right)
			}))
		})
	}
}

func TestCosineDistanceWithCachedMagnitudes(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		left  []float32
		right []float32
	}{
		{left: []float32{1, 2, 3}, right: []float32{4, 5, 6}},
		{left: []float32{0, 0, 0}, right: []float32{0, 0, 0}},
		{left: []float32{0, 0, 0}, right: []float32{1, 0, 0}},
	} {
		leftMagnitude := L2Magnitude(test.left)
		rightMagnitude := L2Magnitude(test.right)
		got := CosineDistanceWithMagnitudes(test.left, test.right, leftMagnitude, rightMagnitude)
		require.InDelta(t, CosineDistance(test.left, test.right), got, 1e-6)
	}
}

var benchmarkDenseScore float32

func TestSparseInnerProduct(t *testing.T) {
	t.Parallel()

	score, err := SparseInnerProduct(
		[]uint32{1, 3, 7}, []float32{2, 4, -1},
		[]uint32{0, 3, 7, 9}, []float32{8, 3, 5, 2},
	)
	require.NoError(t, err)
	require.True(t, score == 7)
	{
		score, err = SparseInnerProduct(nil, nil, nil, nil)
		require.NoError(t, err)
		require.True(t, score == 0)
	}
	{
		_, err = SparseInnerProduct([]uint32{1}, nil, nil, nil)
		require.ErrorIs(t, err, ErrDimensionMismatch)
	}
	{
		_, err = SparseInnerProduct([]uint32{1, 1}, []float32{1, 2}, nil, nil)
		require.ErrorIs(t, err, ErrInvalidSparseOrder)
	}
	{
		_, err = SparseInnerProduct([]uint32{2, 1}, []float32{1, 2}, nil, nil)
		require.ErrorIs(t, err, ErrInvalidSparseOrder)
	}
	{
		_, err = SparseInnerProduct([]uint32{1}, []float32{float32(math.NaN())}, nil, nil)
		require.ErrorIs(t, err, ErrNonFiniteVector)
	}
}

func assertScore(t *testing.T, compute DenseDistance, left, right []float32, expected float32) {
	t.Helper()
	require.Equal(t, expected, compute(left, right))
}
