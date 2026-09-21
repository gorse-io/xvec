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
