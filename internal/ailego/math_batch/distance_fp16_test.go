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
	"math"
	"testing"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/utility"
	"github.com/stretchr/testify/require"
)

func fp16BatchFixture(dimension int) [5][]uint16 {
	var vectors [5][]uint16
	for j := range vectors {
		vectors[j] = make([]uint16, dimension)
		for d := range dimension {
			vectors[j][d] = utility.Float32ToFloat16Bits(float32(math.Sin(float64(j*17+d*3))) * 8)
		}
	}
	return vectors
}

func TestFP16Batch4(t *testing.T) {
	for _, dimension := range []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 768, 769} {
		t.Run(fmt.Sprint(dimension), func(t *testing.T) {
			v := fp16BatchFixture(dimension)
			for _, special := range []bool{false, true} {
				if special {
					clear(v[0])
					for j := 1; j < len(v); j++ {
						for d := range v[j] {
							v[j][d] = []uint16{0, 0x8000, 1, 0x7bff, 0xfbff}[(j+d)%5]
						}
					}
				}
				for _, tc := range []struct {
					name            string
					batch, fallback fp16Batch4Kernel
					single          func([]uint16, []uint16) float32
				}{
					{"l2", SquaredEuclideanDistances4FP16, fp16L2Scalar4, mathutil.L2SquaredFP16},
					{"ip", InnerProducts4FP16, fp16DotScalar4, mathutil.InnerProductFP16},
					{"cosine", CosineDistances4FP16, fp16CosineScalar4, mathutil.CosineDistanceFP16},
					{"mips", MIPSL2SquaredDistances4FP16, fp16MIPSScalar4, mathutil.MIPSL2SquaredFP16},
				} {
					var out [5]float32
					out[4] = 42
					for _, kernel := range []fp16Batch4Kernel{tc.batch, tc.fallback} {
						kernel(v[0], v[1], v[2], v[3], v[4], out[:4])
						for j := range 4 {
							want := tc.single(v[0], v[j+1])
							require.InDelta(t, want, out[j], 2e-5*max(1, math.Abs(float64(want))), tc.name)
						}
						require.Equal(t, float32(42), out[4])
						require.Zero(t, testing.AllocsPerRun(10, func() { kernel(v[0], v[1], v[2], v[3], v[4], out[:4]) }))
					}
				}
			}
		})
	}
}

func BenchmarkFP16CosineBatch4(b *testing.B) {
	v := fp16BatchFixture(768)
	var out [4]float32
	for _, batch := range []bool{false, true} {
		b.Run(fmt.Sprintf("batch=%v", batch), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if batch {
					CosineDistances4FP16(v[0], v[1], v[2], v[3], v[4], out[:])
				} else {
					fp16CosineScalar4(v[0], v[1], v[2], v[3], v[4], out[:])
				}
			}
		})
	}
}
