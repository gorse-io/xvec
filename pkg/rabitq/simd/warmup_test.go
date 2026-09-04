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

func TestWarmupIPX0Q512(t *testing.T) {
	for _, dimension := range []int{64, 448, 512, 576, 1024} {
		for _, queryBits := range []int{1, 4, 8} {
			data, query := makeWarmupInput(dimension, queryBits)
			const delta = float32(0.25)
			const vl = float32(-0.75)
			want := warmupOracle(data, query, delta, vl, dimension, queryBits)
			got := WarmupIPX0Q512(data, query, delta, vl, dimension, queryBits)
			require.Equal(t, want, got, "dimension %d query bits %d", dimension, queryBits)
		}
	}
}

func TestWarmupIPX0Q512RejectsInvalidInput(t *testing.T) {
	data, query := makeWarmupInput(64, 8)
	tests := []struct {
		name      string
		data      []uint64
		query     []uint64
		dimension int
		queryBits int
	}{
		{name: "zero dimension", data: data, query: query, queryBits: 8},
		{name: "unaligned dimension", data: data, query: query, dimension: 63, queryBits: 8},
		{name: "zero query bits", data: data, query: query, dimension: 64},
		{name: "too many query bits", data: data, query: query, dimension: 64, queryBits: 9},
		{name: "short data", query: query, dimension: 64, queryBits: 8},
		{name: "short query", data: data, dimension: 64, queryBits: 8},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.Panics(t, func() {
				WarmupIPX0Q512(test.data, test.query, 1, 1, test.dimension, test.queryBits)
			})
		})
	}
}

func testWarmup(t *testing.T, kernel func(unsafe.Pointer, unsafe.Pointer, float32, float32, int64, int64) float32) {
	t.Helper()
	for _, dimension := range []int{64, 448, 512, 576, 1024} {
		for _, queryBits := range []int{1, 4, 8} {
			data, query := makeWarmupInput(dimension, queryBits)
			const delta = float32(0.25)
			const vl = float32(-0.75)
			want := warmupOracle(data, query, delta, vl, dimension, queryBits)
			got := kernel(unsafe.Pointer(&data[0]), unsafe.Pointer(&query[0]), delta, vl, int64(dimension), int64(queryBits))
			require.Equal(t, want, got, "dimension %d query bits %d", dimension, queryBits)
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
