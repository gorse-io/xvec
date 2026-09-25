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
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:generate make avx int4-avx512 int8-avx2

func init() {
	if cpu.X86.HasAVX {
		kernels.dot2 = innerProducts2AVX
		kernels.dot4 = innerProducts4AVX
		kernels.l2Squared2 = squaredEuclideanDistances2AVX
		kernels.l2Squared4 = squaredEuclideanDistances4AVX
	}
	if cpu.X86.HasAVX2 {
		innerProductsInt8Kernel4 = innerProductsInt8AVX2_4
	}
	if cpu.X86.HasAVX2 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW {
		innerProductsInt4Kernel4 = innerProductsInt4AVX512_4
	}
}

func innerProducts2AVX(query, first, second []float32) (firstProduct, secondProduct float32) {
	if len(query) < 8 {
		return innerProducts2Scalar(query, first, second)
	}
	xvec_avx_batch_inner_products2(
		unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]), int64(len(query)),
		unsafe.Pointer(&firstProduct), unsafe.Pointer(&secondProduct),
	)
	return
}

func innerProducts4AVX(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32) {
	if len(query) < 8 {
		return innerProducts4Scalar(query, first, second, third, fourth)
	}
	xvec_avx_batch_inner_products4(
		unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)),
		unsafe.Pointer(&firstProduct), unsafe.Pointer(&secondProduct),
		unsafe.Pointer(&thirdProduct), unsafe.Pointer(&fourthProduct),
	)
	return
}

func squaredEuclideanDistances2AVX(query, first, second []float32) (firstDistance, secondDistance float32) {
	if len(query) < 8 {
		return squaredEuclideanDistances2Scalar(query, first, second)
	}
	xvec_avx_batch_squared_euclidean_distances2(
		unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]), int64(len(query)),
		unsafe.Pointer(&firstDistance), unsafe.Pointer(&secondDistance),
	)
	return
}

func squaredEuclideanDistances4AVX(query, first, second, third, fourth []float32) (firstDistance, secondDistance, thirdDistance, fourthDistance float32) {
	if len(query) < 8 {
		return squaredEuclideanDistances4Scalar(query, first, second, third, fourth)
	}
	xvec_avx_batch_squared_euclidean_distances4(
		unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)),
		unsafe.Pointer(&firstDistance), unsafe.Pointer(&secondDistance),
		unsafe.Pointer(&thirdDistance), unsafe.Pointer(&fourthDistance),
	)
	return
}

func innerProductsInt8AVX2_4(query, first, second, third, fourth []byte, output []int64) {
	if len(query) < 32 {
		innerProductsInt8Scalar4(query, first, second, third, fourth, output)
		return
	}
	xvec_avx2_batch_inner_products_int8_4(
		unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&output[0]),
	)
}

func innerProductsInt4AVX512_4(query, first, second, third, fourth []byte, output []int64) {
	prefix := len(query) &^ 63
	if prefix == 0 {
		innerProductsInt4Scalar4(query, first, second, third, fourth, output)
		return
	}
	xvec_avx512_batch_inner_products_int4_4(
		unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(prefix), unsafe.Pointer(&output[0]),
	)
	if prefix != len(query) {
		var tail [4]int64
		innerProductsInt4Scalar4(query[prefix:], first[prefix:], second[prefix:], third[prefix:], fourth[prefix:], tail[:])
		for i := range tail {
			output[i] += tail[i]
		}
	}
}
