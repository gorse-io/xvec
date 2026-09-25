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
		kernelsInt4.l2 = squaredEuclideanInt4RVV
		kernelsInt4.dot = innerProductInt4RVV
		kernelsInt4.products = dotNormsInt4RVV
	}
}

func squaredEuclideanRVV(left, right []float32) float32 {
	return squared_euclidean_distance_fp32_rvv(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductRVV(left, right []float32) float32 {
	return inner_product_fp32_rvv(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsRVV(left, right []float32) (dot, leftNorm, rightNorm float32) {
	inner_product_and_squared_norm_fp32_rvv(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&dot), unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}

func innerProductInt4RVV(left, right []byte) int64 {
	if len(left) == 0 {
		return 0
	}
	return inner_product_int4_rvv(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func squaredEuclideanInt4RVV(left, right []byte) int64 {
	if len(left) == 0 {
		return 0
	}
	return squared_euclidean_int4_rvv(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsInt4RVV(left, right []byte) (dot, leftNorm, rightNorm int64) {
	if len(left) == 0 {
		return 0, 0, 0
	}
	var result [3]int64
	dot_norms_int4_rvv(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result[0]))
	return result[0], result[1], result[2]
}
