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

package utility

import "math"

// Float32ToFloat16Bits converts an IEEE 754 binary32 value to binary16 with
// round-to-nearest, ties-to-even. NaN payloads retain their high bits.
func Float32ToFloat16Bits(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16(bits>>16) & 0x8000
	exponent32 := int((bits >> 23) & 0xff)
	mantissa := bits & 0x7fffff

	if exponent32 == 0xff {
		if mantissa == 0 {
			return sign | 0x7c00
		}
		payload := uint16(mantissa >> 13)
		if payload == 0 {
			payload = 1
		}
		return sign | 0x7c00 | payload
	}

	exponent16 := exponent32 - 127 + 15
	if exponent16 >= 31 {
		return sign | 0x7c00
	}
	if exponent16 <= 0 {
		if exponent16 < -10 {
			return sign
		}
		mantissa |= 0x800000
		shift := uint32(14 - exponent16)
		rounded := mantissa >> shift
		remainder := mantissa & ((uint32(1) << shift) - 1)
		halfway := uint32(1) << (shift - 1)
		if remainder > halfway || remainder == halfway && rounded&1 != 0 {
			rounded++
		}
		return sign | uint16(rounded)
	}

	rounded := mantissa >> 13
	remainder := mantissa & 0x1fff
	if remainder > 0x1000 || remainder == 0x1000 && rounded&1 != 0 {
		rounded++
		if rounded == 0x400 {
			rounded = 0
			exponent16++
			if exponent16 >= 31 {
				return sign | 0x7c00
			}
		}
	}
	return sign | uint16(exponent16<<10) | uint16(rounded)
}

// Float16BitsToFloat32 expands an IEEE 754 binary16 bit pattern exactly.
func Float16BitsToFloat32(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exponent := uint32(bits>>10) & 0x1f
	mantissa := uint32(bits & 0x03ff)

	switch exponent {
	case 0:
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		exponent32 := int32(127 - 14)
		for mantissa&0x400 == 0 {
			mantissa <<= 1
			exponent32--
		}
		mantissa &= 0x3ff
		return math.Float32frombits(sign | uint32(exponent32)<<23 | mantissa<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mantissa<<13)
	default:
		exponent32 := exponent + (127 - 15)
		return math.Float32frombits(sign | exponent32<<23 | mantissa<<13)
	}
}
