//go:build !noasm && riscv64

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
	"math"
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	if cpu.RISCV64.HasV {
		activeFHTKernels = fhtKernels{
			flipSigns:      fhtFlipSignsRVV,
			kacWalk:        fhtKacWalkRVV,
			inverseKacWalk: fhtInverseKacWalkRVV,
			inPlace:        fhtInPlaceRVV,
		}
	}
}

func fhtFlipSignsRVV(signs []byte, data []float32) {
	if len(data) == 0 {
		return
	}
	fht_flip_sign_rvv(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtKacWalkRVV(data []float32) {
	fht_kacs_walk_rvv(unsafe.Pointer(&data[0]), int64(len(data)))
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt2)
	}
}

func fhtInverseKacWalkRVV(data []float32) {
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt(.5))
	}
	fht_inv_kacs_walk_rvv(unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtInPlaceRVV(data []float32) {
	fht_inplace_rvv(unsafe.Pointer(&data[0]), int64(len(data)))
}
