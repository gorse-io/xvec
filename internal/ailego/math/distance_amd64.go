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

const (
	int4NibbleMask int64 = 0x0f0f0f0f0f0f0f0f
	int4SignMask   int64 = 0x0808080808080808
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

//go:generate make distance-integer-avx2 distance-int8-avx2 distance-int8-avx512

func init() {
	switch {
	case cpu.X86.HasAVX2 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW:
		kernelsInt4.l2 = squaredEuclideanInt4AVX2
		kernelsInt4.dot = innerProductInt4AVX2
		kernelsInt4.products = dotNormsInt4AVX2
		kernelsInt8.l2 = squaredEuclideanInt8AVX2
		kernelsInt8.dot = innerProductInt8AVX512
		kernelsInt8.products = dotNormsInt8AVX2
	case cpu.X86.HasAVX2:
		kernelsInt4.l2 = squaredEuclideanInt4AVX2
		kernelsInt4.dot = innerProductInt4AVX2
		kernelsInt4.products = dotNormsInt4AVX2
		kernelsInt8.l2 = squaredEuclideanInt8AVX2
		kernelsInt8.dot = innerProductInt8AVX2
		kernelsInt8.products = dotNormsInt8AVX2
	}
}

func innerProductInt4AVX2(left, right []byte) int64 {
	if len(left) < 32 {
		return innerProductInt4Scalar(left, right)
	}
	aligned := len(left) &^ 31
	return inner_product_int4_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(aligned),
		int4NibbleMask, int4SignMask,
	) + innerProductInt4Scalar(left[aligned:], right[aligned:])
}

func squaredEuclideanInt4AVX2(left, right []byte) int64 {
	if len(left) < 32 {
		return squaredEuclideanInt4Scalar(left, right)
	}
	aligned := len(left) &^ 31
	return squared_euclidean_int4_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(aligned),
		int4NibbleMask, int4SignMask,
	) + squaredEuclideanInt4Scalar(left[aligned:], right[aligned:])
}

func dotNormsInt4AVX2(left, right []byte) (dot, leftNorm, rightNorm int64) {
	if len(left) < 32 {
		return dotNormsInt4Scalar(left, right)
	}
	aligned := len(left) &^ 31
	var products [3]int64
	dot_norms_int4_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(aligned),
		unsafe.Pointer(&products[0]), int4NibbleMask, int4SignMask,
	)
	dot, leftNorm, rightNorm = products[0], products[1], products[2]
	tailDot, tailLeftNorm, tailRightNorm := dotNormsInt4Scalar(left[aligned:], right[aligned:])
	dot += tailDot
	leftNorm += tailLeftNorm
	rightNorm += tailRightNorm
	return
}

func innerProductInt8AVX2(left, right []byte) int64 {
	if len(left) < 32 {
		return innerProductInt8Scalar(left, right)
	}
	return inner_product_int8_avx2(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}

func squaredEuclideanInt8AVX2(left, right []byte) int64 {
	if len(left) < 32 {
		return squaredEuclideanInt8Scalar(left, right)
	}
	aligned := len(left) &^ 31
	return squared_euclidean_int8_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(aligned),
	) + squaredEuclideanInt8Scalar(left[aligned:], right[aligned:])
}

func dotNormsInt8AVX2(left, right []byte) (dot, leftNorm, rightNorm int64) {
	if len(left) < 32 {
		return dotNormsInt8Scalar(left, right)
	}
	aligned := len(left) &^ 31
	dot_norms_int8_avx2(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(aligned),
		unsafe.Pointer(&dot), unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	tailDot, tailLeftNorm, tailRightNorm := dotNormsInt8Scalar(left[aligned:], right[aligned:])
	dot += tailDot
	leftNorm += tailLeftNorm
	rightNorm += tailRightNorm
	return
}

func innerProductInt8AVX512(left, right []byte) int64 {
	if len(left) < 64 {
		return innerProductInt8AVX2(left, right)
	}
	return inner_product_int8_avx512(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}
