//go:build !noasm && loong64

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

//go:generate env OBJDUMP=loongarch64-linux-gnu-objdump go tool goat src/fastscan_lasx.c --target loong64 -O3 -mlasx -e=-fno-vectorize -e=-fno-slp-vectorize -o ../simd
//go:generate env OBJDUMP=loongarch64-linux-gnu-objdump go tool goat src/pack_excode_lasx.c --target loong64 -O3 -mlasx -e=-fno-vectorize -e=-fno-slp-vectorize -o ../simd
//go:generate env OBJDUMP=loongarch64-linux-gnu-objdump go tool goat src/rotator_lasx.c --target loong64 -O3 -mlasx -e=-fno-vectorize -e=-fno-slp-vectorize -o ../simd
//go:generate env OBJDUMP=loongarch64-linux-gnu-objdump go tool goat src/space_lasx.c --target loong64 -O3 -mlasx -e=-fno-vectorize -e=-fno-slp-vectorize -o ../simd
//go:generate env OBJDUMP=loongarch64-linux-gnu-objdump go tool goat src/space_excode_lasx.c --target loong64 -O3 -mlasx -e=-fno-vectorize -e=-fno-slp-vectorize -o ../simd
//go:generate env OBJDUMP=loongarch64-linux-gnu-objdump go tool goat src/warmup_lasx.c --target loong64 -O3 -mlasx -e=-fno-vectorize -e=-fno-slp-vectorize -o ../simd
//go:generate rm -f src/fastscan_lasx.o src/fastscan_lasx.s src/pack_excode_lasx.o src/pack_excode_lasx.s src/rotator_lasx.o src/rotator_lasx.s src/space_lasx.o src/space_lasx.s src/space_excode_lasx.o src/space_excode_lasx.s src/warmup_lasx.o src/warmup_lasx.s

import "golang.org/x/sys/cpu"

func init() {
	if !cpu.Loong64.HasLASX {
		return
	}

	warmupIPX0Q512 = warmup_ip_x0_q_512_lasx
	ip16FxU1, ip64FxU2, ip64FxU3, ip16FxU4 = ip16_fxu1_lasx, ip64_fxu2_lasx, ip64_fxu3_lasx, ip16_fxu4_lasx
	ip64FxU5, ip64FxU6, ip64FxU7, ip16FxU8 = ip64_fxu5_lasx, ip64_fxu6_lasx, ip64_fxu7_lasx, ip16_fxu8_lasx
	accumulate, transferLUTHACC, accumulateHACC = accumulate_lasx, transfer_lut_hacc_lasx, accumulate_hacc_lasx
	flipSign = flip_sign_lasx
	kacsWalk = kacs_walk_lasx
	pack2BitExcode = packing_2bit_excode_lasx
	pack3BitExcode = packing_3bit_excode_lasx
	pack4BitExcode = packing_4bit_excode_lasx
	pack5BitExcode = packing_5bit_excode_lasx
	pack6BitExcode = packing_6bit_excode_lasx
	pack7BitExcode = packing_7bit_excode_lasx
	scalarQuantizeUint8 = scalar_quantize_uint8_lasx
	scalarQuantizeUint16 = scalar_quantize_uint16_lasx
	transposeBin = new_transpose_bin_lasx
	transposeBin512 = new_transpose_bin_512_lasx
	maskIPX0Q = mask_ip_x0_q_lasx
}
