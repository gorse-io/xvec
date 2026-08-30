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

package floats

import "unsafe"

//go:noescape
func xvec_avx_batch_inner_products2(query, first, second unsafe.Pointer, size int64, firstOutput, secondOutput unsafe.Pointer)

//go:noescape
func xvec_avx_batch_inner_products4(query, first, second, third, fourth unsafe.Pointer, size int64, firstOutput, secondOutput, thirdOutput, fourthOutput unsafe.Pointer)
