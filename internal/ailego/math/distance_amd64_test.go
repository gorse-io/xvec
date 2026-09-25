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

package mathutil

import (
	"testing"

	"github.com/klauspost/cpuid/v2"
	"github.com/stretchr/testify/require"
	"golang.org/x/sys/cpu"
)

func TestSSEDistanceKernels(t *testing.T) {
	testArchitectureKernels(t, squaredEuclideanSSE, innerProductSSE, dotNormsSSE)
}

func TestAVXDistanceKernels(t *testing.T) {
	if !cpu.X86.HasAVX {
		t.Skip("AVX is not supported by this CPU")
	}
	testArchitectureKernels(t, squaredEuclideanAVX, innerProductAVX, dotNormsAVX)
}

func TestAVXFP16DistanceKernels(t *testing.T) {
	if !cpuid.CPU.Supports(cpuid.AVX, cpuid.F16C) {
		t.Skip("AVX/F16C is not supported by this CPU")
	}
	testArchitectureKernelsFP16(t, squaredEuclideanFP16AVX, innerProductFP16AVX, dotNormsFP16AVX)
}

func TestAVX512DistanceKernels(t *testing.T) {
	if !cpu.X86.HasAVX || !cpu.X86.HasFMA || !cpu.X86.HasAVX512F {
		t.Skip("AVX-512/FMA is not supported by this CPU")
	}
	testArchitectureKernels(t, squaredEuclideanAVX512, innerProductAVX512, dotNormsAVX512)
}

func TestAVX512FP16DistanceKernels(t *testing.T) {
	if !cpuid.CPU.Supports(cpuid.AVX, cpuid.F16C, cpuid.FMA3, cpuid.AVX512F, cpuid.AVX512DQ) {
		t.Skip("AVX-512/F16C/FMA is not supported by this CPU")
	}
	testArchitectureKernelsFP16(t, squaredEuclideanFP16AVX512, innerProductFP16AVX512, dotNormsFP16AVX512)
}

func TestAVX512FP16DistanceKernelsFallback(t *testing.T) {
	if !cpuid.CPU.Supports(cpuid.AVX, cpuid.F16C) {
		t.Skip("AVX/F16C is not supported by this CPU")
	}
	left := []uint16{0x3c00, 0x4000, 0x4200}
	right := []uint16{0x4400, 0x4500, 0x4600}
	require.Equal(t, squaredEuclideanFP16Scalar(left, right), squaredEuclideanFP16AVX512(left, right))
	require.Equal(t, innerProductFP16Scalar(left, right), innerProductFP16AVX512(left, right))
	wantDot, wantLeftNorm, wantRightNorm := dotNormsFP16Scalar(left, right)
	dot, leftNorm, rightNorm := dotNormsFP16AVX512(left, right)
	require.Equal(t, wantDot, dot)
	require.Equal(t, wantLeftNorm, leftNorm)
	require.Equal(t, wantRightNorm, rightNorm)
}

func TestInnerProductInt8AVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported by this CPU")
	}
	testInnerProductInt8(t, innerProductInt8AVX2)
}

func TestInnerProductInt8AVX512(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512BW {
		t.Skip("AVX2, AVX-512F and AVX-512BW are not supported by this CPU")
	}
	testInnerProductInt8(t, innerProductInt8AVX512)
}

func TestInt4DistanceKernelsAVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported by this CPU")
	}
	testInt4Kernels(t, innerProductInt4AVX2, squaredEuclideanInt4AVX2, dotNormsInt4AVX2)
}

func TestInt4DistanceKernelsAVX512(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512BW {
		t.Skip("AVX2, AVX-512F and AVX-512BW are not supported by this CPU")
	}
	testInt4Kernels(t, innerProductInt4AVX512, squaredEuclideanInt4AVX512, dotNormsInt4AVX512)
}
