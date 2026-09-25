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
	"math"
	"unsafe"

	"github.com/klauspost/cpuid/v2"
)

//go:generate make fp16-avx

func init() {
	if cpuid.CPU.Supports(cpuid.AVX, cpuid.F16C) {
		fp16Kernels.l2 = fp16L2AVX4
		fp16Kernels.dot = fp16DotAVX4
		fp16Kernels.cosine = fp16CosineAVX4
		fp16Kernels.mips = fp16MIPSAVX4
	}
}

func fp16L2AVX4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16L2Scalar4(query, first, second, third, fourth, output)
		return
	}
	fp16_l2_avx4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&output[0]))
}

func fp16DotAVX4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16DotScalar4(query, first, second, third, fourth, output)
		return
	}
	fp16_dot_avx4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&output[0]))
}

// The products kernel accumulates four dots, one shared query norm, and four
// candidate norms in a single pass. No persistent norm cache is required.
func fp16ProductsAVX4(query, first, second, third, fourth []uint16) (products [9]float32) {
	fp16_products_avx4(unsafe.Pointer(&query[0]), unsafe.Pointer(&first[0]), unsafe.Pointer(&second[0]),
		unsafe.Pointer(&third[0]), unsafe.Pointer(&fourth[0]), int64(len(query)), unsafe.Pointer(&products[0]))
	return
}

func fp16CosineAVX4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16CosineScalar4(query, first, second, third, fourth, output)
		return
	}
	products := fp16ProductsAVX4(query, first, second, third, fourth)
	queryMagnitude := float32(math.Sqrt(float64(products[4])))
	for j := range 4 {
		magnitude := float32(math.Sqrt(float64(products[5+j])))
		output[j] = cosineDistanceFromProduct(products[j], queryMagnitude, magnitude)
	}
}

func fp16MIPSAVX4(query, first, second, third, fourth []uint16, output []float32) {
	if len(query) < 8 {
		fp16MIPSScalar4(query, first, second, third, fourth, output)
		return
	}
	products := fp16ProductsAVX4(query, first, second, third, fourth)
	for j := range 4 {
		denominator := max(products[4], products[5+j])
		output[j] = 0
		if denominator != 0 {
			output[j] = 2 - 2*products[j]/denominator
		}
	}
}
