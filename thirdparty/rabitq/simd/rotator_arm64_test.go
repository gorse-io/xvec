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

package simd

import (
	"testing"
	"unsafe"
)

func TestRotatorNEON(t *testing.T) {
	testRotatorKernels(t,
		func(flip []byte, data []float32) {
			flip_sign_neon(unsafe.Pointer(&flip[0]), unsafe.Pointer(&data[0]), int64(len(data)))
		},
		func(data []float32) {
			kacs_walk_neon(unsafe.Pointer(&data[0]), int64(len(data)))
		},
	)
}
