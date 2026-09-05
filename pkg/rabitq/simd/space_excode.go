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

type excodeIPFunc func(unsafe.Pointer, unsafe.Pointer, int64) float32

var (
	ip16FxU1 = ip16FxU1Scalar
	ip64FxU2 = ip64FxU2Scalar
	ip64FxU3 = ip64FxU3Scalar
	ip16FxU4 = ip16FxU4Scalar
	ip64FxU5 = ip64FxU5Scalar
	ip64FxU6 = ip64FxU6Scalar
	ip64FxU7 = ip64FxU7Scalar
	ip16FxU8 = ip16FxU8Scalar
)

// IP16FxU1 computes the inner product between query and packed 1-bit unsigned
// codes. Query length must be a positive multiple of 16, and compactCode must
// contain at least len(query)/8 bytes.
func IP16FxU1(query []float32, compactCode []uint8) float32 {
	return excodeIP(query, compactCode, 1, 16, ip16FxU1)
}

// IP64FxU2 computes the inner product between query and packed 2-bit unsigned
// codes. Query length must be a positive multiple of 64, and compactCode must
// contain at least len(query)*2/8 bytes.
func IP64FxU2(query []float32, compactCode []uint8) float32 {
	return excodeIP(query, compactCode, 2, 64, ip64FxU2)
}

// IP64FxU3 computes the inner product between query and packed 3-bit unsigned
// codes. Query length must be a positive multiple of 64, and compactCode must
// contain at least len(query)*3/8 bytes.
func IP64FxU3(query []float32, compactCode []uint8) float32 {
	return excodeIP(query, compactCode, 3, 64, ip64FxU3)
}

// IP16FxU4 computes the inner product between query and packed 4-bit unsigned
// codes. Query length must be a positive multiple of 16, and compactCode must
// contain at least len(query)*4/8 bytes.
func IP16FxU4(query []float32, compactCode []uint8) float32 {
	return excodeIP(query, compactCode, 4, 16, ip16FxU4)
}

// IP64FxU5 computes the inner product between query and packed 5-bit unsigned
// codes. Query length must be a positive multiple of 64, and compactCode must
// contain at least len(query)*5/8 bytes.
func IP64FxU5(query []float32, compactCode []uint8) float32 {
	return excodeIP(query, compactCode, 5, 64, ip64FxU5)
}

// IP64FxU6 computes the inner product between query and packed 6-bit unsigned
// codes. Query length must be a positive multiple of 64, and compactCode must
// contain at least len(query)*6/8 bytes.
func IP64FxU6(query []float32, compactCode []uint8) float32 {
	return excodeIP(query, compactCode, 6, 64, ip64FxU6)
}

// IP64FxU7 computes the inner product between query and packed 7-bit unsigned
// codes. Query length must be a positive multiple of 64, and compactCode must
// contain at least len(query)*7/8 bytes.
func IP64FxU7(query []float32, compactCode []uint8) float32 {
	return excodeIP(query, compactCode, 7, 64, ip64FxU7)
}

// IP16FxU8 computes the inner product between query and 8-bit unsigned codes.
// Query length must be a positive multiple of 16, and code must contain at
// least len(query) bytes.
func IP16FxU8(query []float32, code []uint8) float32 {
	return excodeIP(query, code, 8, 16, ip16FxU8)
}

func excodeIP(query []float32, code []uint8, bits, blockSize int, function excodeIPFunc) float32 {
	if len(query) == 0 || len(query)%blockSize != 0 {
		panic("invalid excode query length")
	}
	if len(code) < len(query)/8*bits {
		panic("excode is too short")
	}
	return function(unsafe.Pointer(&query[0]), unsafe.Pointer(&code[0]), int64(len(query)))
}

func ip16FxU1Scalar(query, compactCode unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, compactCode, dimension, 1)
}

func ip64FxU2Scalar(query, compactCode unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, compactCode, dimension, 2)
}

func ip64FxU3Scalar(query, compactCode unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, compactCode, dimension, 3)
}

func ip16FxU4Scalar(query, compactCode unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, compactCode, dimension, 4)
}

func ip64FxU5Scalar(query, compactCode unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, compactCode, dimension, 5)
}

func ip64FxU6Scalar(query, compactCode unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, compactCode, dimension, 6)
}

func ip64FxU7Scalar(query, compactCode unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, compactCode, dimension, 7)
}

func ip16FxU8Scalar(query, code unsafe.Pointer, dimension int64) float32 {
	return excodeIPScalar(query, code, dimension, 8)
}

func excodeIPScalar(query, compactCode unsafe.Pointer, dimension int64, bits int) float32 {
	values := unsafe.Slice((*float32)(query), dimension)
	code := unsafe.Slice((*uint8)(compactCode), dimension*int64(bits)/8)
	var result float32
	for i, value := range values {
		result += value * float32(excodeValue(code, i, bits))
	}
	return result
}

func excodeValue(code []uint8, index, bits int) uint8 {
	switch bits {
	case 1:
		return code[index/8] >> uint(index%8) & 1
	case 2, 3:
		block, offset := index/64, index%64
		base := block * bits * 8
		value := code[base+offset%16] >> uint(offset/16*2) & 3
		if bits == 3 {
			value |= (code[base+16+offset/8] >> uint(offset%8) & 1) << 2
		}
		return value
	case 4:
		block, offset := index/16, index%16
		return code[block*8+offset%8] >> uint(offset/8*4) & 15
	case 5:
		block, offset := index/64, index%64
		base := block * 40
		value := code[base+offset/32*16+offset%16] >> uint(offset%32/16*4) & 15
		return value | (code[base+32+offset/8]>>uint(offset%8)&1)<<4
	case 6, 7:
		block, offset := index/64, index%64
		base := block * bits * 8
		var value uint8
		if offset < 48 {
			value = code[base+offset/16*16+offset%16] & 63
		} else {
			column := offset - 48
			value = code[base+column] >> 6
			value |= (code[base+16+column] >> 6) << 2
			value |= (code[base+32+column] >> 6) << 4
		}
		if bits == 7 {
			value |= (code[base+48+offset/8] >> uint(offset%8) & 1) << 6
		}
		return value
	case 8:
		return code[index]
	default:
		panic("unsupported excode bit width")
	}
}
