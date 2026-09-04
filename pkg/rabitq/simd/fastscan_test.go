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

func TestFastScanScalar(t *testing.T) {
	testFastScan(t, accumulateScalar, transferLUTHACCScalar, accumulateHACCScalar)
}

func TestFastScanDispatch(t *testing.T) {
	testFastScan(t, accumulate, transferLUTHACC, accumulateHACC)
}

func TestFastScanAPI(t *testing.T) {
	const dimension = 16
	codes := make([]uint8, dimension*4)
	lut8 := make([]uint8, dimension*4)
	lut16 := make([]uint16, dimension*4)
	for i := range codes {
		codes[i] = uint8((i*7+3)%16) | uint8((i*11+5)%16)<<4
		lut8[i] = uint8((i*13 + 17) % 251)
		lut16[i] = uint16((i*997 + 101) % 65521)
	}

	want8 := make([]uint16, 32)
	accumulateScalar(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut8[0]), unsafe.Pointer(&want8[0]), dimension)
	got8 := make([]uint16, 32)
	Accumulate(codes, lut8, got8, dimension)
	require.Equal(t, want8, got8)

	scalarLUT := make([]uint8, dimension*8)
	transferLUTHACCScalar(unsafe.Pointer(&lut16[0]), dimension, unsafe.Pointer(&scalarLUT[0]))
	want16 := make([]int32, 32)
	accumulateHACCScalar(unsafe.Pointer(&codes[0]), unsafe.Pointer(&scalarLUT[0]), unsafe.Pointer(&want16[0]), dimension)
	transferred := make([]uint8, dimension*8)
	TransferLUTHACC(lut16, transferred, dimension)
	got16 := make([]int32, 32)
	AccumulateHACC(codes, transferred, got16, dimension)
	require.Equal(t, want16, got16)
}

func TestFastScanInvalidDimension(t *testing.T) {
	for _, dimension := range []int{-16, 0, 1, 15, 1040} {
		codes := make([]uint8, max(1, dimension*4))
		lut8 := make([]uint8, max(1, dimension*4))
		lut16 := make([]uint16, max(1, dimension*4))
		transferred := make([]uint8, max(1, dimension*8))
		result8 := make([]uint16, 32)
		result16 := make([]int32, 32)

		require.Panics(t, func() { Accumulate(codes, lut8, result8, dimension) })
		require.Panics(t, func() { TransferLUTHACC(lut16, transferred, dimension) })
		require.Panics(t, func() { AccumulateHACC(codes, transferred, result16, dimension) })
	}
}

func TestFastScanInvalidBufferLength(t *testing.T) {
	codes := make([]uint8, 64)
	lut8 := make([]uint8, 64)
	lut16 := make([]uint16, 64)
	transferred := make([]uint8, 128)
	result8 := make([]uint16, 32)
	result16 := make([]int32, 32)

	require.Panics(t, func() { Accumulate(codes[:63], lut8, result8, 16) })
	require.Panics(t, func() { Accumulate(codes, lut8[:63], result8, 16) })
	require.Panics(t, func() { Accumulate(codes, lut8, result8[:31], 16) })
	require.Panics(t, func() { TransferLUTHACC(lut16[:63], transferred, 16) })
	require.Panics(t, func() { TransferLUTHACC(lut16, transferred[:127], 16) })
	require.Panics(t, func() { AccumulateHACC(codes[:63], transferred, result16, 16) })
	require.Panics(t, func() { AccumulateHACC(codes, transferred[:127], result16, 16) })
	require.Panics(t, func() { AccumulateHACC(codes, transferred, result16[:31], 16) })
}

func testFastScan(
	t *testing.T,
	accumulateFn func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int64),
	transferLUTHACCFn func(unsafe.Pointer, int64, unsafe.Pointer),
	accumulateHACCFn func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int64),
) {
	t.Helper()

	perm := [...]int{0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}
	for _, dimension := range []int{16, 32, 64, 1024} {
		codes := make([]uint8, dimension*4)
		lut8 := make([]uint8, dimension*4)
		lut16 := make([]uint16, dimension*4)
		for i := range codes {
			codes[i] = uint8((i*7+3)%16) | uint8((i*11+5)%16)<<4
			if dimension == maxFastScanDimension {
				lut8[i] = 255
				lut16[i] = 65535
			} else {
				lut8[i] = uint8((i*13 + 17) % 251)
				lut16[i] = uint16((i*997 + 101) % 65521)
			}
		}

		want8 := make([]uint16, 32)
		want16 := make([]int32, 32)
		for codebook := 0; codebook < dimension/4; codebook++ {
			base := codebook * 16
			for packed := 0; packed < 16; packed++ {
				code := codes[base+packed]
				loVector := perm[packed]
				hiVector := loVector + 16
				want8[loVector] += uint16(lut8[base+int(code&0x0f)])
				want8[hiVector] += uint16(lut8[base+int(code>>4)])
				want16[loVector] += int32(lut16[base+int(code&0x0f)])
				want16[hiVector] += int32(lut16[base+int(code>>4)])
			}
		}

		got8 := make([]uint16, 32)
		accumulateFn(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut8[0]), unsafe.Pointer(&got8[0]), int64(dimension))
		require.Equal(t, want8, got8, "8-bit dimension %d", dimension)

		transferred := make([]uint8, len(lut16)*2)
		transferLUTHACCFn(unsafe.Pointer(&lut16[0]), int64(dimension), unsafe.Pointer(&transferred[0]))
		got16 := make([]int32, 32)
		accumulateHACCFn(unsafe.Pointer(&codes[0]), unsafe.Pointer(&transferred[0]), unsafe.Pointer(&got16[0]), int64(dimension))
		require.Equal(t, want16, got16, "16-bit dimension %d", dimension)
	}
}
