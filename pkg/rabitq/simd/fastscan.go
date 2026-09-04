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
	accumulate      = accumulateScalar
	transferLUTHACC = transferLUTHACCScalar
	accumulateHACC  = accumulateHACCScalar
)

var fastScanPerm = [...]int{0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}

const maxFastScanDimension = 1024

// Accumulate uses a packed 8-bit lookup table to accumulate scores for a
// 32-vector FastScan block. The dimension must be a positive multiple of 16
// no greater than 1024.
// Codes and lut must each contain dimension*4 elements, and result must contain
// at least 32 elements.
func Accumulate(codes, lut []uint8, result []uint16, dimension int) {
	tableLength := validateFastScanDimension(dimension)
	if len(codes) < tableLength || len(lut) < tableLength || len(result) < 32 {
		panic("simd: invalid FastScan buffer length")
	}
	accumulate(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut[0]), unsafe.Pointer(&result[0]), int64(dimension))
}

// TransferLUTHACC splits a packed 16-bit lookup table into the layout consumed
// by AccumulateHACC. The dimension must be a positive multiple of 16 no greater
// than 1024. Lut must contain dimension*4 elements, and result must contain
// dimension*8 bytes.
func TransferLUTHACC(lut []uint16, result []uint8, dimension int) {
	tableLength := validateFastScanDimension(dimension)
	if len(lut) < tableLength || len(result) < tableLength*2 {
		panic("simd: invalid FastScan buffer length")
	}
	transferLUTHACC(unsafe.Pointer(&lut[0]), int64(dimension), unsafe.Pointer(&result[0]))
}

// AccumulateHACC uses a lookup table produced by TransferLUTHACC to accumulate
// high-accuracy scores for a 32-vector FastScan block. The dimension must be a
// positive multiple of 16 no greater than 1024. Codes must contain dimension*4
// elements, lut must contain dimension*8 bytes, and result must contain at
// least 32 elements.
func AccumulateHACC(codes, lut []uint8, result []int32, dimension int) {
	tableLength := validateFastScanDimension(dimension)
	if len(codes) < tableLength || len(lut) < tableLength*2 || len(result) < 32 {
		panic("simd: invalid FastScan buffer length")
	}
	accumulateHACC(unsafe.Pointer(&codes[0]), unsafe.Pointer(&lut[0]), unsafe.Pointer(&result[0]), int64(dimension))
}

func validateFastScanDimension(dimension int) int {
	if dimension <= 0 || dimension > maxFastScanDimension || dimension%16 != 0 {
		panic("simd: FastScan dimension must be a positive multiple of 16 no greater than 1024")
	}
	return dimension * 4
}

func accumulateScalar(codes, lut, result unsafe.Pointer, dimension int64) {
	packedCodes := unsafe.Slice((*uint8)(codes), dimension*4)
	table := unsafe.Slice((*uint8)(lut), dimension*4)
	output := unsafe.Slice((*uint16)(result), 32)
	clear(output)
	for codebook := int64(0); codebook < dimension/4; codebook++ {
		base := codebook * 16
		for packed, vector := range fastScanPerm {
			code := packedCodes[base+int64(packed)]
			output[vector] += uint16(table[base+int64(code&0x0f)])
			output[vector+16] += uint16(table[base+int64(code>>4)])
		}
	}
}

func transferLUTHACCScalar(lut unsafe.Pointer, dimension int64, result unsafe.Pointer) {
	table := unsafe.Slice((*uint16)(lut), dimension*4)
	output := unsafe.Slice((*uint8)(result), dimension*8)
	for codebook := int64(0); codebook < dimension/4; codebook++ {
		groupBase := codebook / 4 * 128
		lowBase := groupBase + codebook%4*16
		highBase := lowBase + 64
		for i := int64(0); i < 16; i++ {
			value := table[codebook*16+i]
			output[lowBase+i] = uint8(value)
			output[highBase+i] = uint8(value >> 8)
		}
	}
}

func accumulateHACCScalar(codes, lut, result unsafe.Pointer, dimension int64) {
	packedCodes := unsafe.Slice((*uint8)(codes), dimension*4)
	table := unsafe.Slice((*uint8)(lut), dimension*8)
	output := unsafe.Slice((*int32)(result), 32)
	clear(output)
	for codebook := int64(0); codebook < dimension/4; codebook++ {
		codeBase := codebook * 16
		groupBase := codebook / 4 * 128
		lowBase := groupBase + codebook%4*16
		highBase := lowBase + 64
		for packed, vector := range fastScanPerm {
			code := packedCodes[codeBase+int64(packed)]
			loIndex := int64(code & 0x0f)
			hiIndex := int64(code >> 4)
			loValue := uint16(table[lowBase+loIndex]) | uint16(table[highBase+loIndex])<<8
			hiValue := uint16(table[lowBase+hiIndex]) | uint16(table[highBase+hiIndex])<<8
			output[vector] += int32(loValue)
			output[vector+16] += int32(hiValue)
		}
	}
}
