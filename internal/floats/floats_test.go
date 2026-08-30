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

package floats

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDistanceKernelsMatchFloat32Oracle(t *testing.T) {
	t.Parallel()

	for _, dimension := range []int{1, 3, 4, 7, 8, 15, 16, 17, 127, 128, 129, 768, 1536} {
		t.Run(fmt.Sprintf("dimension_%d", dimension), func(t *testing.T) {
			random := rand.New(rand.NewSource(int64(dimension)))
			left := make([]float32, dimension+1)
			right := make([]float32, dimension+1)
			for index := 1; index <= dimension; index++ {
				left[index] = random.Float32()*2 - 1
				right[index] = random.Float32()*2 - 1
			}
			// Offset both slices to exercise unaligned addresses.
			left = left[1:]
			right = right[1:]

			wantL2, wantDot, wantLeftNorm, wantRightNorm := distanceOracle(left, right)
			requireFloat32Close(t, wantL2, L2Squared(left, right))
			requireFloat32Close(t, wantDot, InnerProduct(left, right))
			dot, leftNorm, rightNorm := DotNorms(left, right)
			requireFloat32Close(t, wantDot, dot)
			requireFloat32Close(t, wantLeftNorm, leftNorm)
			requireFloat32Close(t, wantRightNorm, rightNorm)
		})
	}
}

func TestDistanceKernelsUseFloat32Accumulation(t *testing.T) {
	t.Parallel()

	// At float32 precision, adding one after 2^24 no longer changes the sum.
	left := []float32{4096, 1}
	right := []float32{4096, 1}
	require.Equal(t, float32(1<<24), InnerProduct(left, right))
	dot, leftNorm, rightNorm := DotNorms(left, right)
	require.Equal(t, float32(1<<24), dot)
	require.Equal(t, float32(1<<24), leftNorm)
	require.Equal(t, float32(1<<24), rightNorm)
}

func TestInnerProducts2MatchesFloat32Oracle(t *testing.T) {
	t.Parallel()
	for _, dimension := range []int{1, 7, 8, 15, 16, 17, 127, 128, 129, 768, 1536} {
		t.Run(fmt.Sprintf("dimension_%d", dimension), func(t *testing.T) {
			random := rand.New(rand.NewSource(int64(dimension * 3)))
			query := make([]float32, dimension+1)
			first := make([]float32, dimension+1)
			second := make([]float32, dimension+1)
			for index := 1; index <= dimension; index++ {
				query[index] = random.Float32()*2 - 1
				first[index] = random.Float32()*2 - 1
				second[index] = random.Float32()*2 - 1
			}
			query, first, second = query[1:], first[1:], second[1:]
			_, wantFirst, _, _ := distanceOracle(query, first)
			_, wantSecond, _, _ := distanceOracle(query, second)
			gotFirst, gotSecond := InnerProducts2(query, first, second)
			requireFloat32Close(t, wantFirst, gotFirst)
			requireFloat32Close(t, wantSecond, gotSecond)
			require.Equal(t, InnerProduct(query, first), gotFirst)
			require.Equal(t, InnerProduct(query, second), gotSecond)
		})
	}
}

func TestDistanceKernelsDoNotAllocateOrMutate(t *testing.T) {
	left := []float32{0.2, 0.9, -0.4, 0.7}
	right := []float32{0.3, 0.5, 0.8, -0.1}
	leftCopy := append([]float32(nil), left...)
	rightCopy := append([]float32(nil), right...)

	require.Zero(t, testing.AllocsPerRun(100, func() {
		benchmarkL2 = L2Squared(left, right)
		benchmarkInnerProduct = InnerProduct(left, right)
		benchmarkDot, benchmarkDot2 = InnerProducts2(left, right, right)
		benchmarkDot, benchmarkLeftNorm, benchmarkRightNorm = DotNorms(left, right)
	}))
	require.Equal(t, leftCopy, left)
	require.Equal(t, rightCopy, right)
}

func BenchmarkDistanceKernels(b *testing.B) {
	for _, dimension := range []int{128, 768, 1536} {
		left := make([]float32, dimension)
		right := make([]float32, dimension)
		for index := range left {
			left[index] = float32(index%17)/17 - 0.5
			right[index] = float32(index%23)/23 - 0.5
		}
		b.Run(fmt.Sprintf("L2/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkL2 = L2Squared(left, right)
			}
		})
		b.Run(fmt.Sprintf("InnerProduct/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkDot = InnerProduct(left, right)
			}
		})
		b.Run(fmt.Sprintf("InnerProductSequential2/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkDot = InnerProduct(left, right)
				benchmarkDot2 = InnerProduct(left, right)
			}
		})
		b.Run(fmt.Sprintf("InnerProducts2/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkDot, benchmarkDot2 = InnerProducts2(left, right, right)
			}
		})
		b.Run(fmt.Sprintf("DotNorms/%d", dimension), func(b *testing.B) {
			for b.Loop() {
				benchmarkDot, benchmarkLeftNorm, benchmarkRightNorm = DotNorms(left, right)
			}
		})
	}
}

func testArchitectureKernels(
	t *testing.T,
	l2Kernel binaryKernel,
	dotKernel binaryKernel,
	productsKernel productsKernel,
) {
	t.Helper()
	for _, dimension := range []int{1, 3, 4, 7, 8, 15, 16, 17, 127, 128, 129} {
		left := make([]float32, dimension+1)
		right := make([]float32, dimension+1)
		for index := 1; index <= dimension; index++ {
			left[index] = float32((index*7)%19)/19 - 0.5
			right[index] = float32((index*11)%23)/23 - 0.5
		}
		left, right = left[1:], right[1:]
		wantL2, wantDot, wantLeftNorm, wantRightNorm := distanceOracle(left, right)
		requireFloat32Close(t, wantL2, l2Kernel(left, right))
		requireFloat32Close(t, wantDot, dotKernel(left, right))
		dot, leftNorm, rightNorm := productsKernel(left, right)
		requireFloat32Close(t, wantDot, dot)
		requireFloat32Close(t, wantLeftNorm, leftNorm)
		requireFloat32Close(t, wantRightNorm, rightNorm)
	}
}

func distanceOracle(left, right []float32) (l2, dot, leftNorm, rightNorm float32) {
	for index, leftValue := range left {
		rightValue := right[index]
		difference := leftValue - rightValue
		l2 += difference * difference
		dot += leftValue * rightValue
		leftNorm += leftValue * leftValue
		rightNorm += rightValue * rightValue
	}
	return
}

func requireFloat32Close(t *testing.T, expected, actual float32) {
	t.Helper()
	tolerance := float32(1e-5) * max(1, float32(math.Abs(float64(expected))))
	require.InDelta(t, expected, actual, float64(tolerance))
}

var (
	benchmarkL2           float32
	benchmarkInnerProduct float32
	benchmarkDot          float32
	benchmarkDot2         float32
	benchmarkLeftNorm     float32
	benchmarkRightNorm    float32
)
