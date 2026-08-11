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

package ailego

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDenseMetricsMatchPinnedBaseline(t *testing.T) {
	t.Parallel()

	type metricCase struct {
		name      string
		compute   func([]float32, []float32) (float32, error)
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
			actual, err := testCase.compute(testCase.left, testCase.right)
			require.NoError(t, err)

			tolerance := testCase.tolerance
			if tolerance == 0 {
				tolerance = 1e-7
			}
			require.InDelta(t, testCase.expected, actual, float64(tolerance))
		})
	}
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

func TestDenseMetricValidation(t *testing.T) {
	t.Parallel()

	metrics := []struct {
		name    string
		compute func([]float32, []float32) (float32, error)
	}{
		{name: "l2", compute: L2Squared},
		{name: "ip", compute: InnerProduct},
		{name: "cosine", compute: CosineDistance},
		{name: "mips-l2", compute: MIPSL2Squared},
	}
	for _, metric := range metrics {
		t.Run(metric.name, func(t *testing.T) {
			{
				_, err := metric.compute([]float32{1}, []float32{1, 2})
				require.ErrorIs(t, err, ErrDimensionMismatch)
			}
			{
				_, err := metric.compute(nil, nil)
				require.ErrorIs(t, err, ErrEmptyVector)
			}
			{
				_, err := metric.compute([]float32{float32(math.NaN())}, []float32{1})
				require.ErrorIs(t, err, ErrNonFiniteVector)
			}
			{
				_, err := metric.compute([]float32{1}, []float32{float32(math.Inf(1))})
				require.ErrorIs(t, err, ErrNonFiniteVector)
			}
		})
	}

	large := []float32{math.MaxFloat32, math.MaxFloat32}
	{
		_, err := InnerProduct(large, large)
		require.ErrorIs(t, err, ErrNonFiniteVector)
	}
}

func TestPrevalidatedDenseDistanceKernelsMatchCheckedMetrics(t *testing.T) {
	left := []float32{0.2, 0.9, -0.4, 0.7}
	right := []float32{0.3, 0.5, 0.8, -0.1}
	tests := []struct {
		name       string
		checked    func([]float32, []float32) (float32, error)
		prechecked DenseDistance
	}{
		{name: "l2", checked: L2Squared, prechecked: L2SquaredPrevalidated},
		{name: "inner product", checked: InnerProduct, prechecked: InnerProductPrevalidated},
		{name: "cosine", checked: CosineDistance, prechecked: CosineDistancePrevalidated},
		{name: "mips-l2", checked: MIPSL2Squared, prechecked: MIPSL2SquaredPrevalidated},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			expected, err := test.checked(left, right)
			require.NoError(t, err)
			actual, err := test.prechecked(left, right)
			require.NoError(t, err)
			require.Equal(t, expected, actual)
			require.Zero(t, testing.AllocsPerRun(100, func() {
				benchmarkDenseScore, benchmarkDenseErr = test.prechecked(left, right)
			}))
		})
	}
	large := []float32{math.MaxFloat32, math.MaxFloat32}
	_, err := InnerProductPrevalidated(large, large)
	require.ErrorIs(t, err, ErrNonFiniteVector)
}

var benchmarkDenseScore float32
var benchmarkDenseErr error

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

func assertScore(
	t *testing.T,
	compute func([]float32, []float32) (float32, error),
	left, right []float32,
	expected float32,
) {
	t.Helper()
	score, err := compute(left, right)
	require.NoError(t, err)
	require.Equal(t, expected, score)
}
