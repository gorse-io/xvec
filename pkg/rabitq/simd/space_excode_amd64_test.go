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

func TestSpaceExcodeAVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported")
	}
	testSpaceExcode(t, []excodeIPFunc{
		ip16_fxu1_avx2,
		ip64_fxu2_avx2,
		ip64_fxu3_avx2,
		ip16_fxu4_avx2,
		ip64_fxu5_avx2,
		ip64_fxu6_avx2,
		ip64_fxu7_avx2,
		ip16_fxu8_avx2,
	})
}

func TestSpaceExcodeAVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512DQ || !cpu.X86.HasAVX512BW {
		t.Skip("AVX-512F, AVX-512DQ, and AVX-512BW are not supported")
	}
	testSpaceExcode(t, []excodeIPFunc{
		ip16_fxu1_avx512,
		ip64_fxu2_avx512,
		ip64_fxu3_avx512,
		ip16_fxu4_avx512,
		ip64_fxu5_avx512,
		ip64_fxu6_avx512,
		ip64_fxu7_avx512,
		ip16_fxu8_avx512,
	})
}
