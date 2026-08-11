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

package floats

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:generate go tool goat src/floats_neon.c -O3

func init() {
	if cpu.ARM64.HasASIMD {
		kernels.l2 = l2SquaredNEON
		kernels.dot = innerProductNEON
		kernels.products = dotNormsNEON
	}
}

func l2SquaredNEON(left, right []float32) float32 {
	if len(left) < 4 {
		return l2SquaredScalar(left, right)
	}
	var result float32
	xvec_neon_l2_squared(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func innerProductNEON(left, right []float32) float32 {
	if len(left) < 4 {
		return innerProductScalar(left, right)
	}
	var result float32
	xvec_neon_inner_product(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)), unsafe.Pointer(&result))
	return result
}

func dotNormsNEON(left, right []float32) (dot, leftNorm, rightNorm float32) {
	if len(left) < 4 {
		return dotNormsScalar(left, right)
	}
	xvec_neon_dot_norms(
		unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)),
		unsafe.Pointer(&dot), unsafe.Pointer(&leftNorm), unsafe.Pointer(&rightNorm),
	)
	return
}
