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

package mathutil

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:generate make rvv

func init() {
	if cpu.RISCV64.HasV {
		kernels.l2 = squaredEuclideanRVV
		kernels.dot = innerProductRVV
		kernels.products = dotNormsRVV
	}
}

func squaredEuclideanRVV(left, right []float32) float32 {
	var result float32
	xvec_rvv_l2_squared(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func innerProductRVV(left, right []float32) float32 {
	var result float32
	xvec_rvv_inner_product(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func dotNormsRVV(left, right []float32) (dot, leftNorm, rightNorm float32) {
	xvec_rvv_dot_norms(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&dot), unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}
