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

//go:generate make avx

func init() {
	if cpu.X86.HasAVX {
		kernels.dot2 = innerProducts2AVX
		kernels.dot4 = innerProducts4AVX
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
