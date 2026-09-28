//go:build !noasm && arm64

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
	"math"
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:generate make neon

func init() {
	if cpu.ARM64.HasASIMD {
		kernels.dot2 = innerProducts2NEON
		kernels.dot4 = innerProducts4NEON
		kernels.l2Squared2 = squaredEuclideanDistances2NEON
		kernels.l2Squared4 = squaredEuclideanDistances4NEON
		innerProductsInt4Kernel4 = innerProductsInt4NEON_4
		innerProductsInt8Kernel4 = innerProductsInt8NEON_4
	}
	if cpu.ARM64.HasFPHP && cpu.ARM64.HasASIMDHP {
		fp16Kernels.l2 = fp16L2NEON4
		fp16Kernels.dot = fp16DotNEON4
		fp16Kernels.cosine = fp16CosineNEON4
		fp16Kernels.mips = fp16MIPSNEON4
	}
}

func fp16L2NEON4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16L2Scalar4(query, first, second, third, fourth, output)
		return
	}
	fp16_l2_neon4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&output[0]))
}

func fp16DotNEON4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16DotScalar4(query, first, second, third, fourth, output)
		return
	}
	fp16_dot_neon4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&output[0]))
}

func fp16ProductsNEON4(query, first, second, third, fourth []uint16) (products [9]float32) {
	fp16_products_neon4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&products[0]))
	return
}

func fp16CosineNEON4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16CosineScalar4(query, first, second, third, fourth, output)
		return
	}
	products := fp16ProductsNEON4(query, first, second, third, fourth)
	queryMagnitude := float32(math.Sqrt(float64(products[4])))
	for j := range 4 {
		output[j] = cosineDistanceFromProduct(products[j], queryMagnitude, float32(math.Sqrt(float64(products[5+j]))))
	}
}

func fp16MIPSNEON4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16MIPSScalar4(query, first, second, third, fourth, output)
		return
	}
	products := fp16ProductsNEON4(query, first, second, third, fourth)
	for j := range 4 {
		denominator := max(products[4], products[5+j])
		output[j] = 0
		if denominator != 0 {
			output[j] = 2 - 2*products[j]/denominator
		}
	}
}

func innerProducts2NEON(query, first, second []float32) (firstProduct, secondProduct float32) {
	if len(query) < 4 {
		return innerProducts2Scalar(query, first, second)
	}
	xvec_neon_batch_inner_products2(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]), int64(len(query)), unsafe.Pointer(&firstProduct), unsafe.Pointer(&secondProduct))
	return
}

func innerProducts4NEON(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32) {
	if len(query) < 4 {
		return innerProducts4Scalar(query, first, second, third, fourth)
	}
	xvec_neon_batch_inner_products4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]), unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&firstProduct), unsafe.Pointer(&secondProduct), unsafe.Pointer(&thirdProduct), unsafe.Pointer(&fourthProduct))
	return
}

func squaredEuclideanDistances2NEON(query, first, second []float32) (firstDistance, secondDistance float32) {
	if len(query) < 4 {
		return squaredEuclideanDistances2Scalar(query, first, second)
	}
	xvec_neon_batch_squared_euclidean_distances2(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]), int64(len(query)), unsafe.Pointer(&firstDistance), unsafe.Pointer(&secondDistance))
	return
}

func squaredEuclideanDistances4NEON(query, first, second, third, fourth []float32) (firstDistance, secondDistance, thirdDistance, fourthDistance float32) {
	if len(query) < 4 {
		return squaredEuclideanDistances4Scalar(query, first, second, third, fourth)
	}
	xvec_neon_batch_squared_euclidean_distances4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]), unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&firstDistance), unsafe.Pointer(&secondDistance), unsafe.Pointer(&thirdDistance), unsafe.Pointer(&fourthDistance))
	return
}

func innerProductsInt4NEON_4(query, first, second, third, fourth []byte, output []int64) {
	prefix := len(query) &^ 15
	if prefix == 0 {
		innerProductsInt4Scalar4(query, first, second, third, fourth, output)
		return
	}
	xvec_neon_batch_inner_products_int4_4(
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

func innerProductsInt8NEON_4(query, first, second, third, fourth []byte, output []int64) {
	prefix := len(query) &^ 15
	if prefix == 0 {
		innerProductsInt8Scalar4(query, first, second, third, fourth, output)
		return
	}
	vectors := [5]unsafe.Pointer{
		unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]),
	}
	xvec_neon_batch_inner_products_int8_4(unsafe.Pointer(&vectors[0]), int64(prefix), unsafe.Pointer(&output[0]))
	if prefix != len(query) {
		var tail [4]int64
		innerProductsInt8Scalar4(query[prefix:], first[prefix:], second[prefix:], third[prefix:], fourth[prefix:], tail[:])
		for i := range tail {
			output[i] += tail[i]
		}
	}
}
