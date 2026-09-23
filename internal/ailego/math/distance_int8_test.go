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
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInnerProductInt8(t *testing.T) {
	testInnerProductInt8(t, InnerProductInt8)
}

func TestInnerProductInt8Scalar(t *testing.T) {
	testInnerProductInt8(t, innerProductInt8Scalar)
}

func testInnerProductInt8(t *testing.T, dot func([]byte, []byte) int64) {
	t.Helper()
	rng := rand.New(rand.NewPCG(7, 11))
	dimensions := []int{255, 256, 257, 767, 768, 769, 1536, 65535, 65536, 65537, 131073, 1<<20 + 3}
	for dimension := 0; dimension <= 129; dimension++ {
		dimensions = append(dimensions, dimension)
	}
	for _, dimension := range dimensions {
		for _, offset := range []int{0, 1, 15, 31} {
			t.Run(fmt.Sprintf("dimension=%d/offset=%d", dimension, offset), func(t *testing.T) {
				left := make([]byte, dimension+offset)[offset:]
				right := make([]byte, dimension+32-offset)[32-offset:]
				var want int64
				for i := range left {
					l, r := rng.IntN(256)-128, rng.IntN(256)-128
					left[i], right[i] = byte(l), byte(r)
					want += int64(l * r)
				}
				require.Equal(t, want, dot(left, right))
				// Extreme inputs exercise signed widening, saturation hazards,
				// SIMD reduction and totals beyond both int32 and float32 precision.
				for _, pair := range [][2]int{{-128, -128}, {-128, 127}, {127, 127}, {0, -128}} {
					for i := range left {
						left[i], right[i] = byte(pair[0]), byte(pair[1])
					}
					require.Equal(t, int64(dimension)*int64(pair[0]*pair[1]), dot(left, right))
				}
			})
		}
	}
}
