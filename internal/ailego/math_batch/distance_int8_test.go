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

package mathbatch

import (
	"fmt"
	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestInnerProductsInt8(t *testing.T) {
	testInnerProductsInt8(t, InnerProductsInt8)
}

func TestInnerProductsInt8Scalar4(t *testing.T) {
	testInnerProductsInt8(t, int8BatchWithKernel(innerProductsInt8Scalar4))
}

func int8BatchWithKernel(kernel func([]byte, []byte, []byte, []byte, []byte, []int64)) func([]byte, [][]byte, []int64) {
	return func(query []byte, candidates [][]byte, output []int64) {
		i := 0
		for ; i+4 <= len(candidates); i += 4 {
			kernel(query, candidates[i], candidates[i+1], candidates[i+2], candidates[i+3], output[i:i+4])
		}
		for ; i < len(candidates); i++ {
			output[i] = mathutil.InnerProductInt8(query, candidates[i])
		}
	}
}

func testInnerProductsInt8(t *testing.T, batch func([]byte, [][]byte, []int64)) {
	t.Helper()
	for _, dimension := range []int{0, 1, 15, 31, 32, 33, 63, 64, 65, 767, 768, 769, 65535, 65536, 65537, 131073} {
		for _, count := range []int{0, 1, 3, 4, 5, 7, 8, 9} {
			t.Run(fmt.Sprintf("dim=%d/count=%d", dimension, count), func(t *testing.T) {
				query := make([]byte, dimension+1)[1:]
				candidates := make([][]byte, count)
				for j := range candidates {
					offset := (j*7 + 3) % 32
					candidates[j] = make([]byte, dimension+offset)[offset:]
				}
				for _, extreme := range []bool{false, true} {
					for i := range query {
						query[i] = byte(i*13 + 5)
						if extreme {
							query[i] = 128
						}
					}
					want := make([]int64, count)
					for j := range candidates {
						for i := range query {
							candidates[j][i] = byte(i*29 + j*7)
							if extreme {
								if j%2 == 0 {
									candidates[j][i] = 128
								} else {
									candidates[j][i] = 127
								}
							}
							want[j] += int64(int8(query[i])) * int64(int8(candidates[j][i]))
						}
					}
					output := make([]int64, count+2)
					output[count], output[count+1] = 71, 73
					batch(query, candidates, output[:count])
					require.Equal(t, want, output[:count])
					require.Equal(t, []int64{71, 73}, output[count:])
				}
			})
		}
	}
}

func BenchmarkInnerProductsInt8(b *testing.B) {
	query := make([]byte, 768)
	candidates := make([][]byte, 32)
	for j := range candidates {
		candidates[j] = make([]byte, len(query))
		for i := range query {
			query[i], candidates[j][i] = byte(i), byte(i+j)
		}
	}
	output := make([]int64, len(candidates))
	for _, name := range []string{"single", "batch"} {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if name == "batch" {
					InnerProductsInt8(query, candidates, output)
				} else {
					for i := range candidates {
						output[i] = mathutil.InnerProductInt8(query, candidates[i])
					}
				}
			}
		})
	}
}
