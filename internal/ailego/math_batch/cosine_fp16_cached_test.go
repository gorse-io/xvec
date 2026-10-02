// SPDX-License-Identifier: Apache-2.0

package mathbatch

import (
	"fmt"
	"math"
	"testing"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/utility"
	"github.com/stretchr/testify/require"
)

func TestCosineDistances4WithMagnitudesFP16(t *testing.T) {
	for _, dimension := range []int{0, 1, 7, 8, 17, 33, 768} {
		for _, zeroQuery := range []bool{false, true} {
			t.Run(fmt.Sprintf("dim=%d/zero=%t", dimension, zeroQuery), func(t *testing.T) {
				query := make([]uint16, dimension)
				var candidates [4][]uint16
				var magnitudes [4]float32
				for j := range candidates {
					candidates[j] = make([]uint16, dimension)
					for d := range query {
						if !zeroQuery {
							query[d] = utility.Float32ToFloat16Bits(float32(math.Sin(float64(d))))
						}
						if j != 0 {
							candidates[j][d] = utility.Float32ToFloat16Bits(float32(math.Cos(float64(d + j))))
						}
					}
					magnitudes[j] = mathutil.L2MagnitudeFP16(candidates[j])
				}
				queryMagnitude := mathutil.L2MagnitudeFP16(query)
				output := make([]float32, 4)
				CosineDistances4WithMagnitudesFP16(query, candidates[0], candidates[1], candidates[2], candidates[3],
					queryMagnitude, magnitudes[0], magnitudes[1], magnitudes[2], magnitudes[3], output)
				for j := range output {
					want := mathutil.CosineDistanceWithMagnitudesFP16(query, candidates[j], queryMagnitude, magnitudes[j])
					require.InDelta(t, want, output[j], 2e-6)
				}
			})
		}
	}
}
