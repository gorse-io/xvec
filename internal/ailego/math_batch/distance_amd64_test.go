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

package mathbatch

import (
	"testing"

	"github.com/klauspost/cpuid/v2"
	"golang.org/x/sys/cpu"
)

func TestAVXBatchKernels(t *testing.T) {
	if !cpu.X86.HasAVX {
		t.Skip("AVX is not supported")
	}
	testBatchKernels(t, innerProducts2AVX, innerProducts4AVX, squaredEuclideanDistances2AVX, squaredEuclideanDistances4AVX)
}

func TestAVX512BatchKernels(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX-512F is not supported")
	}
	testBatchKernels(t, innerProducts2AVX512, innerProducts4AVX512, squaredEuclideanDistances2AVX512, squaredEuclideanDistances4AVX512)
}

func TestFP16AVX512BatchKernels(t *testing.T) {
	if !cpuid.CPU.Supports(cpuid.AVX, cpuid.F16C, cpuid.AVX512F, cpuid.AVX512DQ) {
		t.Skip("AVX, F16C, AVX-512F and AVX-512DQ are not supported")
	}
	testFP16BatchKernels(t, fp16L2AVX512_4, fp16DotAVX512_4, fp16CosineAVX512_4, fp16MIPSAVX512_4)
}

func TestInnerProductsInt8AVX2_4(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported by this CPU")
	}
	testInnerProductsInt8(t, int8BatchWithKernel(innerProductsInt8AVX2_4))
}

func TestInnerProductsInt8AVX512_4(t *testing.T) {
	if !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512BW {
		t.Skip("AVX-512F and AVX-512BW are not supported by this CPU")
	}
	testInnerProductsInt8(t, int8BatchWithKernel(innerProductsInt8AVX512_4))
}

func TestInnerProductsInt4AVX512_4(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512BW {
		t.Skip("AVX2, AVX-512F and AVX-512BW are not supported by this CPU")
	}
	testInnerProductsInt4(t, int4BatchWithKernel(innerProductsInt4AVX512_4))
}

func TestInnerProductsInt4AVX2_4(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported by this CPU")
	}
	testInnerProductsInt4(t, int4BatchWithKernel(innerProductsInt4AVX2_4))
}
