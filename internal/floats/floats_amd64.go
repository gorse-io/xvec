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
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:generate go tool goat src/floats_avx.c -O3 -mavx
//go:generate go tool goat src/floats_avx512.c -O3 -mavx -mfma -mavx512f

func init() {
	switch {
	case cpu.X86.HasAVX && cpu.X86.HasFMA && cpu.X86.HasAVX512F:
		kernels.l2 = l2SquaredAVX512
		kernels.dot = innerProductAVX512
		kernels.products = dotNormsAVX512
	case cpu.X86.HasAVX:
		kernels.l2 = l2SquaredAVX
		kernels.dot = innerProductAVX
		kernels.products = dotNormsAVX
	}
}

func l2SquaredAVX(left, right []float32) float32 {
	if len(left) < 8 {
		return l2SquaredScalar(left, right)
	}
	var result float32
	xvec_avx_l2_squared(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func innerProductAVX(left, right []float32) float32 {
	if len(left) < 8 {
		return innerProductScalar(left, right)
	}
	var result float32
	xvec_avx_inner_product(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func dotNormsAVX(left, right []float32) (dot, leftNorm, rightNorm float32) {
	if len(left) < 8 {
		return dotNormsScalar(left, right)
	}
	xvec_avx_dot_norms(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&dot), unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}

func l2SquaredAVX512(left, right []float32) float32 {
	if len(left) < 16 {
		return l2SquaredAVX(left, right)
	}
	var result float32
	xvec_avx512_l2_squared(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func innerProductAVX512(left, right []float32) float32 {
	if len(left) < 16 {
		return innerProductAVX(left, right)
	}
	var result float32
	xvec_avx512_inner_product(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func dotNormsAVX512(left, right []float32) (dot, leftNorm, rightNorm float32) {
	if len(left) < 16 {
		return dotNormsAVX(left, right)
	}
	xvec_avx512_dot_norms(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&dot), unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}
