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

// Package simd contains portable and SIMD kernels ported from RaBitQ-Library.
package simd

import "unsafe"

var (
	flipSign = flipSignScalar
	kacsWalk = kacsWalkScalar
)

// FlipSign negates data elements selected by little-endian bits in flip.
// The data length must be a positive multiple of 64, and flip must contain at
// least len(data)/8 bytes.
func FlipSign(flip []byte, data []float32) {
	flipSign(unsafe.Pointer(&flip[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

// KacsWalk applies one Kac transform step in place. The data length must be a
// positive multiple of 32.
func KacsWalk(data []float32) {
	kacsWalk(unsafe.Pointer(&data[0]), int64(len(data)))
}

func flipSignScalar(flip, data unsafe.Pointer, dimension int64) {
	flips := unsafe.Slice((*byte)(flip), (dimension+7)/8)
	values := unsafe.Slice((*float32)(data), dimension)
	for i := range values {
		if flips[i/8]&(1<<uint(i%8)) != 0 {
			values[i] = -values[i]
		}
	}
}

func kacsWalkScalar(data unsafe.Pointer, length int64) {
	values := unsafe.Slice((*float32)(data), length)
	half := len(values) / 2
	for i := range half {
		x, y := values[i], values[i+half]
		values[i] = x + y
		values[i+half] = x - y
	}
}
