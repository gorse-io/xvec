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

func TestSpaceScalar(t *testing.T) {
	testSpace(t, scalarQuantizeUint8Scalar, scalarQuantizeUint16Scalar,
		transposeBinScalar, transposeBin512Scalar, maskIPX0QScalar)
}

func TestSpaceDispatch(t *testing.T) {
	testSpace(t, scalarQuantizeUint8, scalarQuantizeUint16,
		transposeBin, transposeBin512, maskIPX0Q)
}

func testSpace(
	t *testing.T,
	quantizeUint8 func(unsafe.Pointer, unsafe.Pointer, int64, float32, float32),
	quantizeUint16 func(unsafe.Pointer, unsafe.Pointer, int64, float32, float32),
	transpose16 func(unsafe.Pointer, unsafe.Pointer, int64, int64),
	transpose8 func(unsafe.Pointer, unsafe.Pointer, int64, int64),
	maskIP func(unsafe.Pointer, unsafe.Pointer, int64) float32,
) {
	t.Helper()

	for _, dimension := range []int{7, 17, 64} {
		values := make([]float32, dimension)
		want8 := make([]uint8, dimension)
		want16 := make([]uint16, dimension)
		for i := range dimension {
			want8[i] = uint8((i*13 + 7) % 251)
			want16[i] = uint16((i*997 + 101) % 65521)
			values[i] = -3.5 + float32(want8[i])*0.25
		}
		got8 := make([]uint8, dimension)
		quantizeUint8(unsafe.Pointer(&got8[0]), unsafe.Pointer(&values[0]), int64(dimension), -3.5, 0.25)
		require.Equal(t, want8, got8, "uint8 dimension %d", dimension)

		for i := range dimension {
			values[i] = 11.25 + float32(want16[i])*0.125
		}
		got16 := make([]uint16, dimension)
		quantizeUint16(unsafe.Pointer(&got16[0]), unsafe.Pointer(&values[0]), int64(dimension), 11.25, 0.125)
		require.Equal(t, want16, got16, "uint16 dimension %d", dimension)
	}

	tieValues := make([]float32, 17)
	wantTies8 := make([]uint8, len(tieValues))
	wantTies16 := make([]uint16, len(tieValues))
	for i := range tieValues {
		tieValues[i] = float32(i) + 0.5
		wantTies8[i] = uint8(i + 1)
		wantTies16[i] = uint16(i + 1)
	}
	gotTies8 := make([]uint8, len(tieValues))
	quantizeUint8(unsafe.Pointer(&gotTies8[0]), unsafe.Pointer(&tieValues[0]), int64(len(tieValues)), 0, 1)
	require.Equal(t, wantTies8, gotTies8)
	gotTies16 := make([]uint16, len(tieValues))
	quantizeUint16(unsafe.Pointer(&gotTies16[0]), unsafe.Pointer(&tieValues[0]), int64(len(tieValues)), 0, 1)
	require.Equal(t, wantTies16, gotTies16)

	for _, dimension := range []int{64, 128} {
		for _, bits := range []int{1, 7, 16} {
			values := make([]uint16, dimension)
			for i := range values {
				values[i] = uint16(i*1009 + 19)
			}
			want := make([]uint64, dimension/64*bits)
			transposeBinScalar(unsafe.Pointer(&values[0]), unsafe.Pointer(&want[0]), int64(dimension), int64(bits))
			got := make([]uint64, len(want))
			transpose16(unsafe.Pointer(&values[0]), unsafe.Pointer(&got[0]), int64(dimension), int64(bits))
			require.Equal(t, want, got, "uint16 dimension %d bits %d", dimension, bits)
		}
	}

	for _, dimension := range []int{64, 128, 512, 576} {
		for _, bits := range []int{1, 6, 8} {
			values := make([]uint8, dimension)
			for i := range values {
				values[i] = uint8(i*29 + 11)
			}
			want := make([]uint64, dimension/64*bits)
			transposeBin512Scalar(unsafe.Pointer(&values[0]), unsafe.Pointer(&want[0]), int64(dimension), int64(bits))
			got := make([]uint64, len(want))
			transpose8(unsafe.Pointer(&values[0]), unsafe.Pointer(&got[0]), int64(dimension), int64(bits))
			require.Equal(t, want, got, "uint8 dimension %d bits %d", dimension, bits)
		}
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = float32(i+1) / 8
	}
	mask := []uint64{0x8000000000000001, 0x00ff00ff00ff00ff}
	var wantIP float32
	for i, value := range query {
		if mask[i/64]&(uint64(1)<<uint(63-i%64)) != 0 {
			wantIP += value
		}
	}
	gotIP := maskIP(unsafe.Pointer(&query[0]), unsafe.Pointer(&mask[0]), int64(len(query)))
	require.InDelta(t, wantIP, gotIP, 1e-5)
}
