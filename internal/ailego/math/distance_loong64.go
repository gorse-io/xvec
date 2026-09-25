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
		innerProductInt8Kernel = innerProductInt8LASX
		kernelsInt4.l2 = squaredEuclideanInt4LASX
		kernelsInt4.dot = innerProductInt4LASX
		kernelsInt4.products = dotNormsInt4LASX
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

func innerProductInt8LASX(left, right []byte) int64 {
	prefix := len(left) &^ 31
	if prefix == 0 {
		return innerProductInt8Scalar(left, right)
	}
	result := inner_product_int8_lasx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(prefix))
	return result + innerProductInt8Scalar(left[prefix:], right[prefix:])
}

func innerProductInt4LASX(left, right []byte) int64 {
	prefix := len(left) &^ 31
	if prefix == 0 {
		return innerProductInt4Scalar(left, right)
	}
	result := inner_product_int4_lasx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(prefix))
	return result + innerProductInt4Scalar(left[prefix:], right[prefix:])
}

func squaredEuclideanInt4LASX(left, right []byte) int64 {
	prefix := len(left) &^ 31
	if prefix == 0 {
		return squaredEuclideanInt4Scalar(left, right)
	}
	result := squared_euclidean_int4_lasx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(prefix))
	return result + squaredEuclideanInt4Scalar(left[prefix:], right[prefix:])
}

func dotNormsInt4LASX(left, right []byte) (dot, leftNorm, rightNorm int64) {
	prefix := len(left) &^ 31
	if prefix == 0 {
		return dotNormsInt4Scalar(left, right)
	}
	var result [3]int64
	dot_norms_int4_lasx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(prefix), unsafe.Pointer(&result[0]))
	dotTail, leftNormTail, rightNormTail := dotNormsInt4Scalar(left[prefix:], right[prefix:])
	return result[0] + dotTail, result[1] + leftNormTail, result[2] + rightNormTail
}
