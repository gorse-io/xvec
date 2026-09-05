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

package simd

import (
	"math/bits"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestWarmupScalar(t *testing.T) {
	testWarmup(t, warmupIPX0Q512Scalar)
}

func TestWarmupDispatch(t *testing.T) {
	testWarmup(t, warmupIPX0Q512)
}

func testWarmup(t *testing.T, kernel func(unsafe.Pointer, unsafe.Pointer, float32, float32, int64, int64) float32) {
	t.Helper()
	coefficients := [][2]float32{
		{0.25, -0.75},
		{-0.6307932, 1.8377749},
	}
	for _, dimension := range []int{64, 448, 512, 576, 1024} {
		for _, queryBits := range []int{1, 4, 8} {
			data, query := makeWarmupInput(dimension, queryBits)
			for _, coefficient := range coefficients {
				delta, vl := coefficient[0], coefficient[1]
				want := warmupOracle(data, query, delta, vl, dimension, queryBits)
				got := kernel(unsafe.Pointer(&data[0]), unsafe.Pointer(&query[0]), delta, vl, int64(dimension), int64(queryBits))
				require.Equal(t, want, got, "dimension %d query bits %d delta %v vl %v", dimension, queryBits, delta, vl)
			}
		}
	}
}

func makeWarmupInput(dimension, queryBits int) ([]uint64, []uint64) {
	chunks := dimension / 64
	data := make([]uint64, chunks)
	query := make([]uint64, chunks*queryBits)
	for i := range data {
		data[i] = uint64(0x9e3779b97f4a7c15) * uint64(i+1)
	}
	for i := range query {
		query[i] = uint64(0xd6e8feb86659fd93) * uint64(i+3)
	}
	return data, query
}

func warmupOracle(data, query []uint64, delta, vl float32, dimension, queryBits int) float32 {
	var ip, ppc uint64
	chunks := dimension / 64
	queryBase := 0
	for block := 0; block < chunks; {
		blockChunks := min(8, chunks-block)
		for chunk := range blockChunks {
			x := data[block+chunk]
			ppc += uint64(bits.OnesCount64(x))
			for bit := range queryBits {
				y := query[queryBase+bit*blockChunks+chunk]
				ip += uint64(bits.OnesCount64(x&y)) << bit
			}
		}
		block += blockChunks
		queryBase += blockChunks * queryBits
	}
	return delta*float32(ip) + vl*float32(ppc)
}
