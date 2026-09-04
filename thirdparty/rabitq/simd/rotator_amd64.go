//go:build !noasm && amd64

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

//go:generate make avx512

// FlipSignAVX512 negates data elements selected by little-endian bits in flip.
// The data length must be a positive multiple of 64, and flip must contain at
// least len(data)/8 bytes. The caller must verify AVX-512F and AVX-512DQ support.
func FlipSignAVX512(flip []byte, data []float32) {
	flip_sign_avx512(unsafe.Pointer(&flip[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

// KacsWalkAVX512 applies one Kac transform step in place. The data length must
// be a positive multiple of 32. The caller must verify AVX-512F support.
func KacsWalkAVX512(data []float32) {
	kacs_walk_avx512(unsafe.Pointer(&data[0]), int64(len(data)))
}
