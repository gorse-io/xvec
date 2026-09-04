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

//go:generate go tool goat src/rotator_avx2.c --target amd64 -O3 -mavx2 -o ../simd
//go:generate go tool goat src/rotator_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -o ../simd
//go:generate go tool goat src/space_avx2.c --target amd64 -O3 -mavx2 -o ../simd
//go:generate go tool goat src/space_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -mavx512bw -o ../simd
//go:generate go tool goat src/fastscan_avx2.c --target amd64 -O3 -mavx2 -o ../simd
//go:generate go tool goat src/fastscan_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -mavx512bw -o ../simd
//go:generate rm -f src/fastscan_avx2.o src/fastscan_avx2.s src/fastscan_avx512.o src/fastscan_avx512.s src/rotator_avx2.o src/rotator_avx2.s src/rotator_avx512.o src/rotator_avx512.s src/space_avx2.o src/space_avx2.s src/space_avx512.o src/space_avx512.s

import "golang.org/x/sys/cpu"

func initFastScan() {
	switch {
	case cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ && cpu.X86.HasAVX512BW:
		accumulate = accumulate_avx512
		transferLUTHACC = transfer_lut_hacc_avx512
		accumulateHACC = accumulate_hacc_avx512
	case cpu.X86.HasAVX2:
		accumulate = accumulate_avx2
		transferLUTHACC = transfer_lut_hacc_avx2
		accumulateHACC = accumulate_hacc_avx2
	}
}

func init() {
	initFastScan()

	switch {
	case cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ:
		flipSign = flip_sign_avx512
		kacsWalk = kacs_walk_avx512
	case cpu.X86.HasAVX2:
		flipSign = flip_sign_avx2
		kacsWalk = kacs_walk_avx2
	}

	switch {
	case cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ && cpu.X86.HasAVX512BW:
		scalarQuantizeUint8 = scalar_quantize_uint8_avx512
		scalarQuantizeUint16 = scalar_quantize_uint16_avx512
		transposeBin = new_transpose_bin_avx512
		transposeBin512 = new_transpose_bin_512_avx512
		maskIPX0Q = mask_ip_x0_q_avx512
	case cpu.X86.HasAVX2:
		scalarQuantizeUint8 = scalar_quantize_uint8_avx2
		scalarQuantizeUint16 = scalar_quantize_uint16_avx2
		transposeBin = new_transpose_bin_avx2
		transposeBin512 = new_transpose_bin_512_avx2
		maskIPX0Q = mask_ip_x0_q_avx2
	}
}
