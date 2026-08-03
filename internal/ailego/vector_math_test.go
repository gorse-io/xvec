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

package ailego

import (
	"errors"
	"math"
	"testing"
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
			if err != nil {
				t.Fatalf("compute: %v", err)
			}
			tolerance := testCase.tolerance
			if tolerance == 0 {
				tolerance = 1e-7
			}
			if difference := float32(math.Abs(float64(actual - testCase.expected))); difference > tolerance {
				t.Fatalf("score = %g, want %g (difference %g)", actual, testCase.expected, difference)
			}
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
			if _, err := metric.compute([]float32{1}, []float32{1, 2}); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("dimension error = %v", err)
			}
			if _, err := metric.compute(nil, nil); !errors.Is(err, ErrEmptyVector) {
				t.Fatalf("empty error = %v", err)
			}
			if _, err := metric.compute([]float32{float32(math.NaN())}, []float32{1}); !errors.Is(err, ErrNonFiniteVector) {
				t.Fatalf("NaN error = %v", err)
			}
			if _, err := metric.compute([]float32{1}, []float32{float32(math.Inf(1))}); !errors.Is(err, ErrNonFiniteVector) {
				t.Fatalf("infinity error = %v", err)
			}
		})
	}

	large := []float32{math.MaxFloat32, math.MaxFloat32}
	if _, err := InnerProduct(large, large); !errors.Is(err, ErrNonFiniteVector) {
		t.Fatalf("overflow error = %v", err)
	}
}

func TestSparseInnerProduct(t *testing.T) {
	t.Parallel()

	score, err := SparseInnerProduct(
		[]uint32{1, 3, 7}, []float32{2, 4, -1},
		[]uint32{0, 3, 7, 9}, []float32{8, 3, 5, 2},
	)
	if err != nil {
		t.Fatalf("compute sparse inner product: %v", err)
	}
	if score != 7 {
		t.Fatalf("score = %g, want 7", score)
	}

	if score, err = SparseInnerProduct(nil, nil, nil, nil); err != nil || score != 0 {
		t.Fatalf("empty sparse score = %g, error = %v", score, err)
	}
	if _, err = SparseInnerProduct([]uint32{1}, nil, nil, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err = SparseInnerProduct([]uint32{1, 1}, []float32{1, 2}, nil, nil); !errors.Is(err, ErrInvalidSparseOrder) {
		t.Fatalf("duplicate index error = %v", err)
	}
	if _, err = SparseInnerProduct([]uint32{2, 1}, []float32{1, 2}, nil, nil); !errors.Is(err, ErrInvalidSparseOrder) {
		t.Fatalf("descending index error = %v", err)
	}
	if _, err = SparseInnerProduct([]uint32{1}, []float32{float32(math.NaN())}, nil, nil); !errors.Is(err, ErrNonFiniteVector) {
		t.Fatalf("non-finite error = %v", err)
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
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	if score != expected {
		t.Fatalf("score = %g, want %g", score, expected)
	}
}
