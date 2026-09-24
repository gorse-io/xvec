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

package mathutil

import (
	"unsafe"

	"github.com/klauspost/cpuid/v2"
	"golang.org/x/sys/cpu"
)

//go:generate make distance-avx
//go:generate make distance-avx512

func init() {
	kernels.l2 = squaredEuclideanSSE
	kernels.dot = innerProductSSE
	kernels.products = dotNormsSSE
	if cpu.X86.HasAVX {
		kernels.l2 = squaredEuclideanAVX
		kernels.dot = innerProductAVX
		kernels.products = dotNormsAVX
	}
	if cpu.X86.HasAVX && cpu.X86.HasFMA && cpu.X86.HasAVX512F {
		kernels.l2 = squaredEuclideanAVX512
		kernels.dot = innerProductAVX512
		kernels.products = dotNormsAVX512
	}
	if cpuid.CPU.Supports(cpuid.AVX, cpuid.F16C) {
		kernelsFP16.l2 = squaredEuclideanFP16AVX
		kernelsFP16.dot = innerProductFP16AVX
		kernelsFP16.products = dotNormsFP16AVX
	}
	if cpuid.CPU.Supports(cpuid.AVX, cpuid.F16C, cpuid.FMA3, cpuid.AVX512F, cpuid.AVX512DQ) {
		kernelsFP16.l2 = squaredEuclideanFP16AVX512
		kernelsFP16.dot = innerProductFP16AVX512
		kernelsFP16.products = dotNormsFP16AVX512
	}
}

func squaredEuclideanSSE(left, right []float32) float32 {
	return squared_euclidean_distance_fp32_sse(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductSSE(left, right []float32) float32 {
	return inner_product_fp32_sse(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsSSE(left, right []float32) (dot, leftNorm, rightNorm float32) {
	dot = inner_product_and_squared_norm_fp32_sse(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}

func squaredEuclideanAVX(left, right []float32) float32 {
	if len(left) < 8 {
		return squaredEuclideanSSE(left, right)
	}
	return squared_euclidean_distance_fp32_avx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductAVX(left, right []float32) float32 {
	if len(left) < 8 {
		return innerProductSSE(left, right)
	}
	return inner_product_fp32_avx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsAVX(left, right []float32) (dot, leftNorm, rightNorm float32) {
	if len(left) < 8 {
		return dotNormsSSE(left, right)
	}
	dot = inner_product_and_squared_norm_fp32_avx(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}

func squaredEuclideanAVX512(left, right []float32) float32 {
	if len(left) < 16 {
		return squaredEuclideanAVX(left, right)
	}
	return squared_euclidean_distance_fp32_avx512(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductAVX512(left, right []float32) float32 {
	if len(left) < 16 {
		return innerProductAVX(left, right)
	}
	return inner_product_fp32_avx512(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsAVX512(left, right []float32) (dot, leftNorm, rightNorm float32) {
	if len(left) < 16 {
		return dotNormsAVX(left, right)
	}
	dot = inner_product_and_squared_norm_fp32_avx512(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}

func squaredEuclideanFP16AVX(left, right []uint16) float32 {
	if len(left) < 8 {
		return squaredEuclideanFP16Scalar(left, right)
	}
	return squared_euclidean_distance_fp16_avx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductFP16AVX(left, right []uint16) float32 {
	if len(left) < 8 {
		return innerProductFP16Scalar(left, right)
	}
	return inner_product_fp16_avx(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsFP16AVX(left, right []uint16) (dot, leftNorm, rightNorm float32) {
	if len(left) < 8 {
		return dotNormsFP16Scalar(left, right)
	}
	dot = inner_product_and_squared_norm_fp16_avx(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}

func squaredEuclideanFP16AVX512(left, right []uint16) float32 {
	if len(left) < 16 {
		return squaredEuclideanFP16AVX(left, right)
	}
	return squared_euclidean_distance_fp16_avx512(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductFP16AVX512(left, right []uint16) float32 {
	if len(left) < 16 {
		return innerProductFP16AVX(left, right)
	}
	return inner_product_fp16_avx512(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func dotNormsFP16AVX512(left, right []uint16) (dot, leftNorm, rightNorm float32) {
	if len(left) < 16 {
		return dotNormsFP16AVX(left, right)
	}
	dot = inner_product_and_squared_norm_fp16_avx512(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}

//go:generate make distance-int4-avx2 distance-int8-avx2 distance-int8-avx512

func init() {
	switch {
	case cpu.X86.HasAVX2 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW:
		innerProductInt8Kernel = innerProductInt8AVX512
	case cpu.X86.HasAVX2:
		innerProductInt8Kernel = innerProductInt8AVX2
	}
	if cpu.X86.HasAVX2 {
		kernelsInt4.l2 = squaredEuclideanInt4AVX2
		kernelsInt4.dot = innerProductInt4AVX2
		kernelsInt4.products = dotNormsInt4AVX2
	}
}

func innerProductInt8AVX2(left, right []byte) int64 {
	if len(left) < 32 {
		return innerProductInt8Scalar(left, right)
	}
	return inner_product_int8_avx2(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func innerProductInt8AVX512(left, right []byte) int64 {
	if len(left) < 64 {
		return innerProductInt8Scalar(left, right)
	}
	return inner_product_int8_avx512(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

const (
	int4MaskBits = int64(0x0f0f0f0f0f0f0f0f)
	int4SignBits = int64(0x0808080808080808)
)

func innerProductInt4AVX2(left, right []byte) int64 {
	prefix := len(left) &^ 31
	if prefix == 0 {
		return innerProductInt4Scalar(left, right)
	}
	result := inner_product_int4_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(prefix), int4MaskBits, int4SignBits,
	)
	return result + innerProductInt4Scalar(left[prefix:], right[prefix:])
}

func squaredEuclideanInt4AVX2(left, right []byte) int64 {
	prefix := len(left) &^ 31
	if prefix == 0 {
		return squaredEuclideanInt4Scalar(left, right)
	}
	result := squared_euclidean_int4_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(prefix), int4MaskBits, int4SignBits,
	)
	return result + squaredEuclideanInt4Scalar(left[prefix:], right[prefix:])
}

func dotNormsInt4AVX2(left, right []byte) (dot, leftNorm, rightNorm int64) {
	prefix := len(left) &^ 31
	if prefix == 0 {
		return dotNormsInt4Scalar(left, right)
	}
	var result [3]int64
	dot_norms_int4_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(prefix), int4MaskBits, int4SignBits,
		unsafe.Pointer(&result[0]),
	)
	dotTail, leftNormTail, rightNormTail := dotNormsInt4Scalar(left[prefix:], right[prefix:])
	return result[0] + dotTail, result[1] + leftNormTail, result[2] + rightNormTail
}
