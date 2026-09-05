//go:build !noasm && riscv64

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

//go:generate env OBJDUMP=riscv64-linux-gnu-objdump go tool goat src/fastscan_rvv.c --target riscv64 -O3 -march=rv64imafdv
//go:generate env OBJDUMP=riscv64-linux-gnu-objdump go tool goat src/pack_excode_rvv.c --target riscv64 -O3 -march=rv64imafdv
//go:generate env OBJDUMP=riscv64-linux-gnu-objdump go tool goat src/rotator_rvv.c --target riscv64 -O3 -march=rv64imafdv
//go:generate env OBJDUMP=riscv64-linux-gnu-objdump go tool goat src/space_rvv.c --target riscv64 -O3 -march=rv64imafdv
//go:generate env OBJDUMP=riscv64-linux-gnu-objdump go tool goat src/space_excode_rvv.c --target riscv64 -O3 -march=rv64imafdv
//go:generate env OBJDUMP=riscv64-linux-gnu-objdump go tool goat src/warmup_rvv.c --target riscv64 -O3 -march=rv64imafdv
//go:generate rm -f src/fastscan_rvv.o src/fastscan_rvv.s src/pack_excode_rvv.o src/pack_excode_rvv.s src/rotator_rvv.o src/rotator_rvv.s src/space_rvv.o src/space_rvv.s src/space_excode_rvv.o src/space_excode_rvv.s src/warmup_rvv.o src/warmup_rvv.s

import "golang.org/x/sys/cpu"

func init() {
	if !cpu.RISCV64.HasV {
		return
	}
	warmupIPX0Q512 = warmup_ip_x0_q_512_rvv
	ip16FxU1, ip64FxU2, ip64FxU3, ip16FxU4 = ip16_fxu1_rvv, ip64_fxu2_rvv, ip64_fxu3_rvv, ip16_fxu4_rvv
	ip64FxU5, ip64FxU6, ip64FxU7, ip16FxU8 = ip64_fxu5_rvv, ip64_fxu6_rvv, ip64_fxu7_rvv, ip16_fxu8_rvv
	accumulate, transferLUTHACC, accumulateHACC = accumulate_rvv, transfer_lut_hacc_rvv, accumulate_hacc_rvv
	flipSign = flip_sign_rvv
	kacsWalk = kacs_walk_rvv
	pack2BitExcode = packing_2bit_excode_rvv
	pack3BitExcode = packing_3bit_excode_rvv
	pack4BitExcode = packing_4bit_excode_rvv
	pack5BitExcode = packing_5bit_excode_rvv
	pack6BitExcode = packing_6bit_excode_rvv
	pack7BitExcode = packing_7bit_excode_rvv
	scalarQuantizeUint8 = scalar_quantize_uint8_rvv
	scalarQuantizeUint16 = scalar_quantize_uint16_rvv
	transposeBin = new_transpose_bin_rvv
	transposeBin512 = new_transpose_bin_512_rvv
	maskIPX0Q = mask_ip_x0_q_rvv
}
