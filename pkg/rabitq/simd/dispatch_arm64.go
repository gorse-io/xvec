//go:build !noasm && arm64

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

//go:generate go tool goat src/fastscan_neon.c --target arm64 -O3 -o ../simd
//go:generate go tool goat src/rotator_neon.c --target arm64 -O3 -o ../simd
//go:generate go tool goat src/space_neon.c --target arm64 -O3 -o ../simd
//go:generate rm -f src/fastscan_neon.o src/fastscan_neon.s src/rotator_neon.o src/rotator_neon.s src/space_neon.o src/space_neon.s

func init() {
	accumulate, transferLUTHACC, accumulateHACC = accumulate_neon, transfer_lut_hacc_neon, accumulate_hacc_neon
	flipSign = flip_sign_neon
	kacsWalk = kacs_walk_neon
	scalarQuantizeUint8 = scalar_quantize_uint8_neon
	scalarQuantizeUint16 = scalar_quantize_uint16_neon
	transposeBin = new_transpose_bin_neon
	transposeBin512 = new_transpose_bin_512_neon
	maskIPX0Q = mask_ip_x0_q_neon
}
