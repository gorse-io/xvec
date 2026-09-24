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
	testInt4Kernels(t, innerProductInt4Scalar, squaredEuclideanInt4Scalar, dotNormsInt4Scalar)
	testInt4Kernels(t, InnerProductInt4, L2SquaredInt4, dotNormsInt4)
}

func testInt4Kernels(
	t *testing.T,
	dot func([]byte, []byte) int64,
	l2 func([]byte, []byte) int64,
	products func([]byte, []byte) (int64, int64, int64),
) {
	t.Helper()
	rng := rand.New(rand.NewPCG(13, 17))
	lengths := []int{255, 256, 257, 511, 512, 513, 1<<20 + 3}
	for length := 0; length <= 129; length++ {
		lengths = append(lengths, length)
	}
	for _, length := range lengths {
		for _, offset := range []int{0, 1, 15, 31} {
			t.Run(fmt.Sprintf("bytes=%d/offset=%d", length, offset), func(t *testing.T) {
				left := make([]byte, length+offset)[offset:]
				right := make([]byte, length+32-offset)[32-offset:]
				for i := range left {
					left[i], right[i] = byte(rng.Uint32()), byte(rng.Uint32())
				}
				wantDot, wantLeftNorm, wantRightNorm := int4Oracle(left, right)
				require.Equal(t, wantDot, dot(left, right))
				require.Equal(t, wantLeftNorm+wantRightNorm-2*wantDot, l2(left, right))
				actualDot, actualLeftNorm, actualRightNorm := products(left, right)
				require.Equal(t, wantDot, actualDot)
				require.Equal(t, wantLeftNorm, actualLeftNorm)
				require.Equal(t, wantRightNorm, actualRightNorm)

				for _, pair := range [][2]byte{{0x88, 0x88}, {0x88, 0x77}, {0x77, 0x77}, {0x00, 0x88}} {
					for i := range left {
						left[i], right[i] = pair[0], pair[1]
					}
					wantDot, wantLeftNorm, wantRightNorm = int4Oracle(left, right)
					require.Equal(t, wantDot, dot(left, right))
					require.Equal(t, wantLeftNorm+wantRightNorm-2*wantDot, l2(left, right))
					actualDot, actualLeftNorm, actualRightNorm = products(left, right)
					require.Equal(t, wantDot, actualDot)
					require.Equal(t, wantLeftNorm, actualLeftNorm)
					require.Equal(t, wantRightNorm, actualRightNorm)
				}
			})
		}
	}
}

func TestInt4DerivedDistances(t *testing.T) {
	left := packInt4(-8, 7, -3, 2, 0, 1)
	right := packInt4(7, -8, 2, -3, 0, -1)
	dot, leftNorm, rightNorm := int4Oracle(left, right)
	l2 := leftNorm + rightNorm - 2*dot

	require.Equal(t, -dot, MinusInnerProductInt4(left, right))
	require.Equal(t, float32(math.Sqrt(float64(l2))), L2Int4(left, right))

	const e2 = float32(0.01)
	wantSpherical := float32(2) - 2*e2*float32(dot) - 2*float32(math.Sqrt(
		float64((1-e2*float32(leftNorm))*(1-e2*float32(rightNorm))),
	))
	require.InDelta(t, wantSpherical, MIPSSphericalInt4(left, right, e2), 1e-6)

	wantRepeated := e2 * float32(l2)
	leftScaled, rightScaled := e2*float32(leftNorm), e2*float32(rightNorm)
	for range 3 {
		difference := leftScaled - rightScaled
		wantRepeated += difference * difference
		leftScaled *= leftScaled
		rightScaled *= rightScaled
	}
	require.InDelta(t, wantRepeated, MIPSRepeatedQuadraticInt4(left, right, 3, e2), 1e-6)
}

func TestMIPSSphericalInt4Localized(t *testing.T) {
	left := packInt4(2, 0)
	right := packInt4(1, 0)
	require.Equal(t, float32(1), MIPSSphericalInt4(left, right, 0))
	require.Equal(t, float32(0), MIPSSphericalInt4(packInt4(0, 0), packInt4(0, 0), 0))
}

func int4Oracle(left, right []byte) (dot, leftNorm, rightNorm int64) {
	for index, packedLeft := range left {
		packedRight := right[index]
		for shift := uint(0); shift <= 4; shift += 4 {
			leftNibble := int8((packedLeft >> shift) & 0x0f)
			rightNibble := int8((packedRight >> shift) & 0x0f)
			leftValue := int64((leftNibble ^ 8) - 8)
			rightValue := int64((rightNibble ^ 8) - 8)
			dot += leftValue * rightValue
			leftNorm += leftValue * leftValue
			rightNorm += rightValue * rightValue
		}
	}
	return
}

func packInt4(values ...int8) []byte {
	packed := make([]byte, len(values)/2)
	for index, value := range values {
		packed[index/2] |= (byte(value) & 0x0f) << (4 * (index % 2))
	}
	return packed
}
