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

//go:generate go tool goat src/rotator_avx2.c --target amd64 -O3 -mavx2 -o ../simd
//go:generate go tool goat src/rotator_avx512.c --target amd64 -O3 -mavx512f -mavx512dq -o ../simd
//go:generate rm -f src/rotator_avx2.o src/rotator_avx2.s src/rotator_avx512.o src/rotator_avx512.s

import "golang.org/x/sys/cpu"

func init() {
	_, activeFlipSign, activeKacsWalk = selectAMD64RotatorKernels(
		cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ,
		cpu.X86.HasAVX2,
	)
}

func selectAMD64RotatorKernels(
	hasAVX512 bool,
	hasAVX2 bool,
) (rotatorImplementation, flipSignKernel, kacsWalkKernel) {
	switch {
	case hasAVX512:
		return rotatorAVX512, flip_sign_avx512, kacs_walk_avx512
	case hasAVX2:
		return rotatorAVX2, flip_sign_avx2, kacs_walk_avx2
	default:
		return rotatorScalar, flipSignScalarKernel, kacsWalkScalarKernel
	}
}
