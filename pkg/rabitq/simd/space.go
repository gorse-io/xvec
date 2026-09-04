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
	"math"
	"unsafe"
)

var (
	scalarQuantizeUint8  = scalarQuantizeUint8Scalar
	scalarQuantizeUint16 = scalarQuantizeUint16Scalar
	transposeBin         = transposeBinScalar
	transposeBin512      = transposeBin512Scalar
	maskIPX0Q            = maskIPX0QScalar
)

// ScalarQuantizeUint8 quantizes data into result using round((value-lo)/delta).
// Scaled values must be in the uint8 range, and result must have at least
// len(data) elements.
func ScalarQuantizeUint8(result []uint8, data []float32, lo, delta float32) {
	if len(data) == 0 {
		return
	}
	scalarQuantizeUint8(unsafe.Pointer(&result[0]), unsafe.Pointer(&data[0]), int64(len(data)), lo, delta)
}

// ScalarQuantizeUint16 quantizes data into result using round((value-lo)/delta).
// Scaled values must be in the uint16 range, and result must have at least
// len(data) elements.
func ScalarQuantizeUint16(result []uint16, data []float32, lo, delta float32) {
	if len(data) == 0 {
		return
	}
	scalarQuantizeUint16(unsafe.Pointer(&result[0]), unsafe.Pointer(&data[0]), int64(len(data)), lo, delta)
}

// TransposeBin transposes the low bits of 16-bit values into 64-value bit planes.
// Data length must be a positive multiple of 64. Result must have at least
// len(data)/64*bits elements, and bits must not exceed 16.
func TransposeBin(data []uint16, result []uint64, bits int) {
	transposeBin(unsafe.Pointer(&data[0]), unsafe.Pointer(&result[0]), int64(len(data)), int64(bits))
}

// TransposeBin512 transposes the low bits of 8-bit values into bit planes,
// grouped into blocks of up to 512 values. Data length must be a positive
// multiple of 64. Result must have at least len(data)/64*bits elements, and
// bits must not exceed 8.
func TransposeBin512(data []uint8, result []uint64, bits int) {
	transposeBin512(unsafe.Pointer(&data[0]), unsafe.Pointer(&result[0]), int64(len(data)), int64(bits))
}

// MaskIPX0Q sums query elements selected by big-endian bits in data.
// Query length must be a positive multiple of 64, and data must contain at
// least len(query)/64 elements.
func MaskIPX0Q(query []float32, data []uint64) float32 {
	return maskIPX0Q(unsafe.Pointer(&query[0]), unsafe.Pointer(&data[0]), int64(len(query)))
}

func scalarQuantizeUint8Scalar(result, data unsafe.Pointer, dimension int64, lo, delta float32) {
	output := unsafe.Slice((*uint8)(result), dimension)
	values := unsafe.Slice((*float32)(data), dimension)
	oneOverDelta := float32(1) / delta
	for i, value := range values {
		output[i] = uint8(math.Round(float64((value - lo) * oneOverDelta)))
	}
}

func scalarQuantizeUint16Scalar(result, data unsafe.Pointer, dimension int64, lo, delta float32) {
	output := unsafe.Slice((*uint16)(result), dimension)
	values := unsafe.Slice((*float32)(data), dimension)
	oneOverDelta := float32(1) / delta
	for i, value := range values {
		output[i] = uint16(math.Round(float64((value - lo) * oneOverDelta)))
	}
}

func transposeBinScalar(data, result unsafe.Pointer, paddedDimension, bits int64) {
	values := unsafe.Slice((*uint16)(data), paddedDimension)
	output := unsafe.Slice((*uint64)(result), paddedDimension/64*bits)
	clear(output)
	for block := int64(0); block < paddedDimension/64; block++ {
		for bit := int64(0); bit < bits; bit++ {
			for i := int64(0); i < 64; i++ {
				if values[block*64+i]&(uint16(1)<<uint(bit)) != 0 {
					output[block*bits+bit] |= uint64(1) << uint(63-i)
				}
			}
		}
	}
}

func transposeBin512Scalar(data, result unsafe.Pointer, paddedDimension, bits int64) {
	values := unsafe.Slice((*uint8)(data), paddedDimension)
	output := unsafe.Slice((*uint64)(result), paddedDimension/64*bits)
	clear(output)
	for blockStart := int64(0); blockStart < paddedDimension; blockStart += 512 {
		blockSize := min(int64(512), paddedDimension-blockStart)
		chunks := blockSize / 64
		outputBase := blockStart / 64 * bits
		for bit := int64(0); bit < bits; bit++ {
			for chunk := int64(0); chunk < chunks; chunk++ {
				for i := int64(0); i < 64; i++ {
					if values[blockStart+chunk*64+i]&(uint8(1)<<uint(bit)) != 0 {
						output[outputBase+bit*chunks+chunk] |= uint64(1) << uint(63-i)
					}
				}
			}
		}
	}
}

func maskIPX0QScalar(query, data unsafe.Pointer, paddedDimension int64) float32 {
	values := unsafe.Slice((*float32)(query), paddedDimension)
	masks := unsafe.Slice((*uint64)(data), paddedDimension/64)
	var result float32
	for i, value := range values {
		if masks[i/64]&(uint64(1)<<uint(63-i%64)) != 0 {
			result += value
		}
	}
	return result
}
