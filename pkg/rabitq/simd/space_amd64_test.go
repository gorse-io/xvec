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

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func TestSpaceAVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported")
	}
	testSpace(t, scalar_quantize_uint8_avx2, scalar_quantize_uint16_avx2,
		new_transpose_bin_avx2, new_transpose_bin_512_avx2, mask_ip_x0_q_avx2)
}

func TestSpaceAVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512DQ || !cpu.X86.HasAVX512BW {
		t.Skip("AVX-512F, AVX-512DQ, and AVX-512BW are not supported")
	}
	testSpace(t, scalar_quantize_uint8_avx512, scalar_quantize_uint16_avx512,
		new_transpose_bin_avx512, new_transpose_bin_512_avx512, mask_ip_x0_q_avx512)
}
