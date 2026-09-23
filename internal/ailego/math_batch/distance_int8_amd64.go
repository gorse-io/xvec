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
	"golang.org/x/sys/cpu"
	"unsafe"
)

//go:generate make int8-avx2

func init() {
	if cpu.X86.HasAVX2 {
		innerProductsInt8Kernel4 = innerProductsInt8AVX2_4
	}
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
