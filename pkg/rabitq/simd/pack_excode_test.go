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

type packExcodeFunc func(unsafe.Pointer, unsafe.Pointer, int64)

func TestPackExcodeScalar(t *testing.T) {
	testPackExcode(t, []packExcodeFunc{
		pack2BitExcodeScalar,
		pack3BitExcodeScalar,
		pack4BitExcodeScalar,
		pack5BitExcodeScalar,
		pack6BitExcodeScalar,
		pack7BitExcodeScalar,
	})
}

func TestPackExcodeDispatch(t *testing.T) {
	testPackExcode(t, []packExcodeFunc{
		pack2BitExcode,
		pack3BitExcode,
		pack4BitExcode,
		pack5BitExcode,
		pack6BitExcode,
		pack7BitExcode,
	})
}

func testPackExcode(t *testing.T, functions []packExcodeFunc) {
	t.Helper()

	for bits := 2; bits <= 7; bits++ {
		dimensions := []int{64, 128}
		if bits == 4 {
			dimensions = []int{16, 32, 64, 128}
		}
		for _, dimension := range dimensions {
			raw := make([]uint8, dimension)
			for i := range raw {
				raw[i] = uint8((i*37 + bits*11) & ((1 << bits) - 1))
			}
			want := referencePackExcode(raw, bits)
			got := make([]uint8, len(want))
			for i := range got {
				got[i] = 0xff
			}
			functions[bits-2](unsafe.Pointer(&raw[0]), unsafe.Pointer(&got[0]), int64(dimension))
			require.Equal(t, want, got, "bits %d dimension %d", bits, dimension)
		}
	}
}

func referencePackExcode(raw []uint8, bits int) []uint8 {
	result := make([]uint8, len(raw)*bits/8)
	inputOffset, outputOffset := 0, 0
	for inputOffset < len(raw) {
		blockSize := 64
		if bits == 4 {
			blockSize = 16
		}
		block := raw[inputOffset : inputOffset+blockSize]
		switch bits {
		case 2:
			for i := range 16 {
				result[outputOffset+i] = block[i] | block[i+16]<<2 | block[i+32]<<4 | block[i+48]<<6
			}
		case 3:
			for i := range 16 {
				result[outputOffset+i] = block[i]&3 | (block[i+16]&3)<<2 |
					(block[i+32]&3)<<4 | (block[i+48]&3)<<6
			}
			packTopBit(block, result[outputOffset+16:], 2)
		case 4:
			for i := range 8 {
				result[outputOffset+i] = block[i] | block[i+8]<<4
			}
		case 5:
			for i := range 16 {
				result[outputOffset+i] = block[i]&15 | (block[i+16]&15)<<4
				result[outputOffset+16+i] = block[i+32]&15 | (block[i+48]&15)<<4
			}
			packTopBit(block, result[outputOffset+32:], 4)
		case 6, 7:
			for i := range 16 {
				result[outputOffset+i] = block[i]&63 | (block[i+48]&3)<<6
				result[outputOffset+16+i] = block[i+16]&63 | ((block[i+48]>>2)&3)<<6
				result[outputOffset+32+i] = block[i+32]&63 | ((block[i+48]>>4)&3)<<6
			}
			if bits == 7 {
				packTopBit(block, result[outputOffset+48:], 6)
			}
		}
		inputOffset += blockSize
		outputOffset += blockSize * bits / 8
	}
	return result
}

func packTopBit(raw, result []uint8, bit uint) {
	for i, value := range raw {
		result[i/8] |= (value >> bit & 1) << (i % 8)
	}
}
