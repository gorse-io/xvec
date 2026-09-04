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

import "unsafe"

var (
	pack2BitExcode = pack2BitExcodeScalar
	pack3BitExcode = pack3BitExcodeScalar
	pack4BitExcode = pack4BitExcodeScalar
	pack5BitExcode = pack5BitExcodeScalar
	pack6BitExcode = pack6BitExcodeScalar
	pack7BitExcode = pack7BitExcodeScalar
)

// Pack2BitExcode packs 2-bit codes in 64-value blocks.
// Raw length must be a positive multiple of 64, and compact must contain at
// least len(raw)*2/8 bytes.
func Pack2BitExcode(raw, compact []uint8) {
	pack2BitExcode(unsafe.Pointer(&raw[0]), unsafe.Pointer(&compact[0]), int64(len(raw)))
}

// Pack3BitExcode packs 3-bit codes in 64-value blocks.
// Raw length must be a positive multiple of 64, and compact must contain at
// least len(raw)*3/8 bytes.
func Pack3BitExcode(raw, compact []uint8) {
	pack3BitExcode(unsafe.Pointer(&raw[0]), unsafe.Pointer(&compact[0]), int64(len(raw)))
}

// Pack4BitExcode packs 4-bit codes in 16-value blocks.
// Raw length must be a positive multiple of 16, and compact must contain at
// least len(raw)*4/8 bytes.
func Pack4BitExcode(raw, compact []uint8) {
	pack4BitExcode(unsafe.Pointer(&raw[0]), unsafe.Pointer(&compact[0]), int64(len(raw)))
}

// Pack5BitExcode packs 5-bit codes in 64-value blocks.
// Raw length must be a positive multiple of 64, and compact must contain at
// least len(raw)*5/8 bytes.
func Pack5BitExcode(raw, compact []uint8) {
	pack5BitExcode(unsafe.Pointer(&raw[0]), unsafe.Pointer(&compact[0]), int64(len(raw)))
}

// Pack6BitExcode packs 6-bit codes in 64-value blocks.
// Raw length must be a positive multiple of 64, and compact must contain at
// least len(raw)*6/8 bytes.
func Pack6BitExcode(raw, compact []uint8) {
	pack6BitExcode(unsafe.Pointer(&raw[0]), unsafe.Pointer(&compact[0]), int64(len(raw)))
}

// Pack7BitExcode packs 7-bit codes in 64-value blocks.
// Raw length must be a positive multiple of 64, and compact must contain at
// least len(raw)*7/8 bytes.
func Pack7BitExcode(raw, compact []uint8) {
	pack7BitExcode(unsafe.Pointer(&raw[0]), unsafe.Pointer(&compact[0]), int64(len(raw)))
}

func pack2BitExcodeScalar(raw, compact unsafe.Pointer, dimension int64) {
	input := unsafe.Slice((*uint8)(raw), dimension)
	output := unsafe.Slice((*uint8)(compact), dimension*2/8)
	for offset := int64(0); offset < dimension; offset += 64 {
		for i := int64(0); i < 16; i++ {
			output[offset/4+i] = input[offset+i] | input[offset+16+i]<<2 |
				input[offset+32+i]<<4 | input[offset+48+i]<<6
		}
	}
}

func pack3BitExcodeScalar(raw, compact unsafe.Pointer, dimension int64) {
	input := unsafe.Slice((*uint8)(raw), dimension)
	output := unsafe.Slice((*uint8)(compact), dimension*3/8)
	for inputOffset, outputOffset := int64(0), int64(0); inputOffset < dimension; inputOffset, outputOffset = inputOffset+64, outputOffset+24 {
		for i := int64(0); i < 16; i++ {
			output[outputOffset+i] = input[inputOffset+i]&3 | (input[inputOffset+16+i]&3)<<2 |
				(input[inputOffset+32+i]&3)<<4 | (input[inputOffset+48+i]&3)<<6
		}
		packExcodeTopBit(input[inputOffset:inputOffset+64], output[outputOffset+16:outputOffset+24], 2)
	}
}

func pack4BitExcodeScalar(raw, compact unsafe.Pointer, dimension int64) {
	input := unsafe.Slice((*uint8)(raw), dimension)
	output := unsafe.Slice((*uint8)(compact), dimension*4/8)
	for inputOffset, outputOffset := int64(0), int64(0); inputOffset < dimension; inputOffset, outputOffset = inputOffset+16, outputOffset+8 {
		for i := int64(0); i < 8; i++ {
			output[outputOffset+i] = input[inputOffset+i] | input[inputOffset+8+i]<<4
		}
	}
}

func pack5BitExcodeScalar(raw, compact unsafe.Pointer, dimension int64) {
	input := unsafe.Slice((*uint8)(raw), dimension)
	output := unsafe.Slice((*uint8)(compact), dimension*5/8)
	for inputOffset, outputOffset := int64(0), int64(0); inputOffset < dimension; inputOffset, outputOffset = inputOffset+64, outputOffset+40 {
		for i := int64(0); i < 16; i++ {
			output[outputOffset+i] = input[inputOffset+i]&15 | (input[inputOffset+16+i]&15)<<4
			output[outputOffset+16+i] = input[inputOffset+32+i]&15 | (input[inputOffset+48+i]&15)<<4
		}
		packExcodeTopBit(input[inputOffset:inputOffset+64], output[outputOffset+32:outputOffset+40], 4)
	}
}

func pack6BitExcodeScalar(raw, compact unsafe.Pointer, dimension int64) {
	pack67BitExcodeScalar(raw, compact, dimension, false)
}

func pack7BitExcodeScalar(raw, compact unsafe.Pointer, dimension int64) {
	pack67BitExcodeScalar(raw, compact, dimension, true)
}

func pack67BitExcodeScalar(raw, compact unsafe.Pointer, dimension int64, topBit bool) {
	bits := int64(6)
	if topBit {
		bits = 7
	}
	input := unsafe.Slice((*uint8)(raw), dimension)
	output := unsafe.Slice((*uint8)(compact), dimension*bits/8)
	outputBlockSize := bits * 8
	for inputOffset, outputOffset := int64(0), int64(0); inputOffset < dimension; inputOffset, outputOffset = inputOffset+64, outputOffset+outputBlockSize {
		for i := int64(0); i < 16; i++ {
			last := input[inputOffset+48+i]
			output[outputOffset+i] = input[inputOffset+i]&63 | (last&3)<<6
			output[outputOffset+16+i] = input[inputOffset+16+i]&63 | ((last>>2)&3)<<6
			output[outputOffset+32+i] = input[inputOffset+32+i]&63 | ((last>>4)&3)<<6
		}
		if topBit {
			packExcodeTopBit(input[inputOffset:inputOffset+64], output[outputOffset+48:outputOffset+56], 6)
		}
	}
}

func packExcodeTopBit(raw, compact []uint8, bit uint) {
	clear(compact)
	for i, value := range raw {
		compact[i/8] |= (value >> bit & 1) << (i % 8)
	}
}
