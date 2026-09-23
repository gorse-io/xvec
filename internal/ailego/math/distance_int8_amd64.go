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

//go:generate make distance-int8-avx2

func init() {
	if cpu.X86.HasAVX2 {
		innerProductInt8Kernel = innerProductInt8AVX2
	}
}

func innerProductInt8AVX2(left, right []byte) int64 {
	if len(left) < 32 {
		return innerProductInt8Scalar(left, right)
	}
	return inner_product_int8_avx2(unsafe.Pointer(&left[0]), unsafe.Pointer(&right[0]), int64(len(left)))
}
