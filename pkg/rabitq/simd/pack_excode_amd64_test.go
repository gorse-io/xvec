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

func TestPackExcodeAVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported")
	}
	testPackExcode(t, []packExcodeFunc{
		packing_2bit_excode_avx2,
		packing_3bit_excode_avx2,
		packing_4bit_excode_avx2,
		packing_5bit_excode_avx2,
		packing_6bit_excode_avx2,
		packing_7bit_excode_avx2,
	})
}

func TestPackExcodeAVX512(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512DQ {
		t.Skip("AVX2, AVX-512F and AVX-512DQ are not supported")
	}
	testPackExcode(t, []packExcodeFunc{
		packing_2bit_excode_avx512,
		packing_3bit_excode_avx512,
		packing_4bit_excode_avx512,
		packing_5bit_excode_avx512,
		packing_6bit_excode_avx512,
		packing_7bit_excode_avx512,
	})
}
