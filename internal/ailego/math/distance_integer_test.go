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
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInt4Distances(t *testing.T) {
	testInt4DistanceKernels(t, InnerProductInt4, L2SquaredInt4, dotNormsInt4)
}

func TestInt4DistanceScalars(t *testing.T) {
	testInt4DistanceKernels(t, innerProductInt4Scalar, squaredEuclideanInt4Scalar, dotNormsInt4Scalar)
}

func testInt4DistanceKernels(
	t *testing.T,
	dot func([]byte, []byte) int64,
	l2 func([]byte, []byte) int64,
	products func([]byte, []byte) (int64, int64, int64),
) {
	t.Helper()
	rng := rand.New(rand.NewPCG(13, 17))
	lengths := []int{0, 1, 2, 15, 16, 17, 31, 32, 33, 255, 256, 257, 65535, 65536, 65537}
	for _, length := range lengths {
		for _, offset := range []int{0, 1, 15, 31} {
			t.Run(fmt.Sprintf("bytes=%d/offset=%d", length, offset), func(t *testing.T) {
				left := make([]byte, length+offset)[offset:]
				right := make([]byte, length+32-offset)[32-offset:]
				var wantDot, wantL2, wantLeftNorm, wantRightNorm int64
				for i := range left {
					lo, hi := int8(rng.IntN(16)-8), int8(rng.IntN(16)-8)
					rlo, rhi := int8(rng.IntN(16)-8), int8(rng.IntN(16)-8)
					left[i] = packInt4(lo, hi)
					right[i] = packInt4(rlo, rhi)
					for _, pair := range [][2]int8{{lo, rlo}, {hi, rhi}} {
						l, r := int64(pair[0]), int64(pair[1])
						wantDot += l * r
						wantL2 += (l - r) * (l - r)
						wantLeftNorm += l * l
						wantRightNorm += r * r
					}
				}
				require.Equal(t, wantDot, dot(left, right))
				require.Equal(t, wantL2, l2(left, right))
				gotDot, gotLeftNorm, gotRightNorm := products(left, right)
				require.Equal(t, wantDot, gotDot)
				require.Equal(t, wantLeftNorm, gotLeftNorm)
				require.Equal(t, wantRightNorm, gotRightNorm)
			})
		}
	}
}

func TestInt8Distances(t *testing.T) {
	testInt8DistanceKernels(t, InnerProductInt8, L2SquaredInt8, dotNormsInt8)
}

func TestInt8DistanceScalars(t *testing.T) {
	testInt8DistanceKernels(t, innerProductInt8Scalar, squaredEuclideanInt8Scalar, dotNormsInt8Scalar)
}

func testInt8DistanceKernels(
	t *testing.T,
	dot func([]byte, []byte) int64,
	l2 func([]byte, []byte) int64,
	products func([]byte, []byte) (int64, int64, int64),
) {
	t.Helper()
	rng := rand.New(rand.NewPCG(19, 23))
	lengths := []int{0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 65535, 65536, 65537, 131073}
	for _, length := range lengths {
		for _, offset := range []int{0, 1, 15, 31} {
			t.Run(fmt.Sprintf("bytes=%d/offset=%d", length, offset), func(t *testing.T) {
				left := make([]byte, length+offset)[offset:]
				right := make([]byte, length+32-offset)[32-offset:]
				var wantDot, wantL2, wantLeftNorm, wantRightNorm int64
				for i := range left {
					l, r := int64(rng.IntN(256)-128), int64(rng.IntN(256)-128)
					left[i], right[i] = byte(l), byte(r)
					wantDot += l * r
					wantL2 += (l - r) * (l - r)
					wantLeftNorm += l * l
					wantRightNorm += r * r
				}
				require.Equal(t, wantDot, dot(left, right))
				require.Equal(t, wantL2, l2(left, right))
				gotDot, gotLeftNorm, gotRightNorm := products(left, right)
				require.Equal(t, wantDot, gotDot)
				require.Equal(t, wantLeftNorm, gotLeftNorm)
				require.Equal(t, wantRightNorm, gotRightNorm)
			})
		}
	}
}

func TestIntegerMIPSDistances(t *testing.T) {
	int4Left := []byte{packInt4(-8, 7), packInt4(3, -2)}
	int4Right := []byte{packInt4(7, -8), packInt4(-4, 5)}
	int8Left := []byte{signedByte(-128), 127, 3, signedByte(-2)}
	int8Right := []byte{127, signedByte(-128), signedByte(-4), 5}

	for _, test := range []struct {
		name      string
		left      []byte
		right     []byte
		products  func([]byte, []byte) (int64, int64, int64)
		spherical func([]byte, []byte, float32) float32
		repeated  func([]byte, []byte, int, float32) float32
	}{
		{"int4", int4Left, int4Right, dotNormsInt4Scalar, MIPSSphericalL2SquaredInt4, MIPSRepeatedQuadraticL2SquaredInt4},
		{"int8", int8Left, int8Right, dotNormsInt8Scalar, MIPSSphericalL2SquaredInt8, MIPSRepeatedQuadraticL2SquaredInt8},
	} {
		t.Run(test.name, func(t *testing.T) {
			dot, leftNorm, rightNorm := test.products(test.left, test.right)
			for _, e2 := range []float32{0, 1 / float32(max(leftNorm, rightNorm)+1)} {
				require.InDelta(t, sphericalMIPSReference(dot, leftNorm, rightNorm, e2), test.spherical(test.left, test.right, e2), 1e-6)
				for repetitions := 0; repetitions <= 3; repetitions++ {
					require.InDelta(t, repeatedMIPSReference(dot, leftNorm, rightNorm, repetitions, e2), test.repeated(test.left, test.right, repetitions, e2), 1e-6)
				}
			}
		})
	}

	require.Equal(t, float32(0), MIPSSphericalL2SquaredInt4([]byte{0}, []byte{0}, 0))
	require.Equal(t, float32(0), MIPSSphericalL2SquaredInt8([]byte{0}, []byte{0}, 0))
}

func signedByte(value int) byte {
	return byte(int8(value))
}

func packInt4(low, high int8) byte {
	return byte(low)&0x0f | byte(high<<4)
}

func sphericalMIPSReference(dot, leftNorm, rightNorm int64, e2 float32) float32 {
	if e2 == 0 {
		denominator := max(leftNorm, rightNorm)
		if denominator == 0 {
			return 0
		}
		return 2 - 2*float32(dot)/float32(denominator)
	}
	left, right := float64(leftNorm), float64(rightNorm)
	scale := float64(e2)
	v := (1 - scale*left) * (1 - scale*right)
	score := 1 - scale*float64(dot)
	if v > 0 {
		score -= math.Sqrt(v)
	}
	return float32(2 * score)
}

func repeatedMIPSReference(dot, leftNorm, rightNorm int64, repetitions int, e2 float32) float32 {
	left := float32(leftNorm) * e2
	right := float32(rightNorm) * e2
	sum := float32(leftNorm+rightNorm-2*dot) * e2
	for range repetitions {
		difference := left - right
		sum += difference * difference
		left *= left
		right *= right
	}
	return sum
}
