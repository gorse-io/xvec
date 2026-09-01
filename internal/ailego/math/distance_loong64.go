//go:build !noasm && loong64

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

//go:generate make lasx

func init() {
	if cpu.Loong64.HasLASX {
		kernels.l2 = squaredEuclideanLASX
		kernels.dot = innerProductLASX
		kernels.products = dotNormsLASX
	}
}

func squaredEuclideanLASX(left, right []float32) float32 {
	if len(left) < 8 {
		return squaredEuclideanScalar(left, right)
	}
	return squared_euclidean_distance_fp32_lasx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductLASX(left, right []float32) float32 {
	if len(left) < 8 {
		return innerProductScalar(left, right)
	}
	return inner_product_fp32_lasx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsLASX(left, right []float32) (dot, leftNorm, rightNorm float32) {
	if len(left) < 8 {
		return dotNormsScalar(left, right)
	}
	inner_product_and_squared_norm_fp32_lasx(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&dot), unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}
