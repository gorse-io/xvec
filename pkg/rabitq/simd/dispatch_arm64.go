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
//go:generate go tool goat src/pack_excode_neon.c --target arm64 -O3 -o ../simd
//go:generate go tool goat src/rotator_neon.c --target arm64 -O3 -o ../simd
//go:generate go tool goat src/space_neon.c --target arm64 -O3 -o ../simd
//go:generate go tool goat src/space_excode_neon.c --target arm64 -O3 -o ../simd
//go:generate go tool goat src/warmup_neon.c --target arm64 -O3 -o ../simd
//go:generate rm -f src/fastscan_neon.o src/fastscan_neon.s src/pack_excode_neon.o src/pack_excode_neon.s src/rotator_neon.o src/rotator_neon.s src/space_neon.o src/space_neon.s src/space_excode_neon.o src/space_excode_neon.s src/warmup_neon.o src/warmup_neon.s

func init() {
	warmupIPX0Q512 = warmup_ip_x0_q_512_neon
	ip16FxU1, ip64FxU2, ip64FxU3, ip16FxU4 = ip16_fxu1_neon, ip64_fxu2_neon, ip64_fxu3_neon, ip16_fxu4_neon
	ip64FxU5, ip64FxU6, ip64FxU7, ip16FxU8 = ip64_fxu5_neon, ip64_fxu6_neon, ip64_fxu7_neon, ip16_fxu8_neon
	accumulate, transferLUTHACC, accumulateHACC = accumulate_neon, transfer_lut_hacc_neon, accumulate_hacc_neon
	flipSign = flip_sign_neon
	kacsWalk = kacs_walk_neon
	pack2BitExcode = packing_2bit_excode_neon
	pack3BitExcode = packing_3bit_excode_neon
	pack4BitExcode = packing_4bit_excode_neon
	pack5BitExcode = packing_5bit_excode_neon
	pack6BitExcode = packing_6bit_excode_neon
	pack7BitExcode = packing_7bit_excode_neon
	scalarQuantizeUint8 = scalar_quantize_uint8_neon
	scalarQuantizeUint16 = scalar_quantize_uint16_neon
	transposeBin = new_transpose_bin_neon
	transposeBin512 = new_transpose_bin_512_neon
	maskIPX0Q = mask_ip_x0_q_neon
}
