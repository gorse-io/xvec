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

type flipSignKernel func(flip, data unsafe.Pointer, dimension int64)
type kacsWalkKernel func(data unsafe.Pointer, length int64)

type rotatorImplementation uint8

const (
	rotatorScalar rotatorImplementation = iota
	rotatorAVX2
	rotatorAVX512
	rotatorNEON
)

var (
	activeFlipSign = flipSignScalarKernel
	activeKacsWalk = kacsWalkScalarKernel
)

// FlipSign negates data elements selected by little-endian bits in flip.
// The data length must be a positive multiple of 64, and flip must contain at
// least len(data)/8 bytes.
func FlipSign(flip []byte, data []float32) {
	activeFlipSign(unsafe.Pointer(&flip[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

// KacsWalk applies one Kac transform step in place. The data length must be a
// positive multiple of 32.
func KacsWalk(data []float32) {
	activeKacsWalk(unsafe.Pointer(&data[0]), int64(len(data)))
}

func flipSignScalarKernel(flip, data unsafe.Pointer, dimension int64) {
	flipSignScalar(
		unsafe.Slice((*byte)(flip), (dimension+7)/8),
		unsafe.Slice((*float32)(data), dimension),
	)
}

func kacsWalkScalarKernel(data unsafe.Pointer, length int64) {
	kacsWalkScalar(unsafe.Slice((*float32)(data), length))
}

func flipSignScalar(flip []byte, data []float32) {
	for i := range data {
		if flip[i/8]&(1<<uint(i%8)) != 0 {
			data[i] = -data[i]
		}
	}
}

func kacsWalkScalar(data []float32) {
	half := len(data) / 2
	for i := range half {
		x, y := data[i], data[i+half]
		data[i] = x + y
		data[i+half] = x - y
	}
}
