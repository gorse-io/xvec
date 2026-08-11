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

package floats

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func TestAVXDistanceKernels(t *testing.T) {
	if !cpu.X86.HasAVX {
		t.Skip("AVX is not supported by this CPU")
	}
	testArchitectureKernels(t, l2SquaredAVX, innerProductAVX, dotNormsAVX)
}

func TestAVX512DistanceKernels(t *testing.T) {
	if !cpu.X86.HasAVX || !cpu.X86.HasFMA || !cpu.X86.HasAVX512F {
		t.Skip("AVX-512/FMA is not supported by this CPU")
	}
	testArchitectureKernels(t, l2SquaredAVX512, innerProductAVX512, dotNormsAVX512)
}
