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
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestSpaceExcodeScalar(t *testing.T) {
	testSpaceExcode(t, []excodeIPFunc{
		ip16FxU1Scalar,
		ip64FxU2Scalar,
		ip64FxU3Scalar,
		ip16FxU4Scalar,
		ip64FxU5Scalar,
		ip64FxU6Scalar,
		ip64FxU7Scalar,
		ip16FxU8Scalar,
	})
}

func TestSpaceExcodeDispatch(t *testing.T) {
	testSpaceExcode(t, []excodeIPFunc{
		ip16FxU1,
		ip64FxU2,
		ip64FxU3,
		ip16FxU4,
		ip64FxU5,
		ip64FxU6,
		ip64FxU7,
		ip16FxU8,
	})
}

func TestSpaceExcodeInvalidInput(t *testing.T) {
	require.Panics(t, func() { IP16FxU1(nil, nil) })
	require.Panics(t, func() { IP16FxU1(make([]float32, 15), make([]uint8, 2)) })
	require.Panics(t, func() { IP16FxU1(make([]float32, 16), make([]uint8, 1)) })
	require.Panics(t, func() { IP64FxU2(make([]float32, 63), make([]uint8, 16)) })
	require.Panics(t, func() { IP64FxU3(make([]float32, 64), make([]uint8, 23)) })
	require.Panics(t, func() { IP16FxU4(make([]float32, 16), make([]uint8, 7)) })
	require.Panics(t, func() { IP64FxU5(make([]float32, 64), make([]uint8, 39)) })
	require.Panics(t, func() { IP64FxU6(make([]float32, 64), make([]uint8, 47)) })
	require.Panics(t, func() { IP64FxU7(make([]float32, 64), make([]uint8, 55)) })
	require.Panics(t, func() { IP16FxU8(make([]float32, 16), make([]uint8, 15)) })
}

func testSpaceExcode(t *testing.T, functions []excodeIPFunc) {
	t.Helper()

	for bits, function := range functions {
		bits++
		blockSize := 64
		if bits == 1 || bits == 4 || bits == 8 {
			blockSize = 16
		}
		for _, dimension := range []int{blockSize, blockSize * 2, 768} {
			query := make([]float32, dimension)
			for i := range query {
				query[i] = float32((i*37)%101-50) / 13
			}
			raw := make([]uint8, dimension)
			for i := range raw {
				raw[i] = uint8((i*29 + bits*17 + i/7) % (1 << bits))
			}
			compact := packExcodeForTest(raw, bits)
			var want float32
			for i := range query {
				want += query[i] * float32(raw[i])
			}
			got := function(unsafe.Pointer(&query[0]), unsafe.Pointer(&compact[0]), int64(dimension))
			require.InDelta(t, want, got, max(1e-4, float64(abs32(want))*1e-5),
				"dimension %d bits %d", dimension, bits)
		}
	}
}

func packExcodeForTest(raw []uint8, bits int) []uint8 {
	compact := make([]uint8, len(raw)*bits/8)
	switch bits {
	case 1:
		for i, value := range raw {
			compact[i/8] |= (value & 1) << uint(i%8)
		}
	case 2:
		Pack2BitExcode(raw, compact)
	case 3:
		Pack3BitExcode(raw, compact)
	case 4:
		Pack4BitExcode(raw, compact)
	case 5:
		Pack5BitExcode(raw, compact)
	case 6:
		Pack6BitExcode(raw, compact)
	case 7:
		Pack7BitExcode(raw, compact)
	case 8:
		copy(compact, raw)
	}
	return compact
}

func abs32(value float32) float32 {
	if value < 0 {
		return -value
	}
	return value
}
