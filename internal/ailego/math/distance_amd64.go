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
