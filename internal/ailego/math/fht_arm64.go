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

package mathutil

import (
	"math"
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:generate make neon

func init() {
	if cpu.ARM64.HasASIMD {
		activeFHTKernels = fhtKernels{
			flipSigns:      fhtFlipSignsNEON,
			kacWalk:        fhtKacWalkNEON,
			inverseKacWalk: fhtInverseKacWalkNEON,
			inPlace:        fhtInPlaceNEON,
		}
	}
}

func fhtFlipSignsNEON(signs []byte, data []float32) {
	if len(data) < 4 {
		fhtFlipSignsScalar(signs, data)
		return
	}
	xvec_neon_fht_flip_signs(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtKacWalkNEON(data []float32) {
	if len(data)/2 < 4 {
		fhtKacWalkScalar(data)
		return
	}
	xvec_neon_fht_kac_walk(unsafe.Pointer(&data[0]), int64(len(data)))
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt2)
	}
}

func fhtInverseKacWalkNEON(data []float32) {
	if len(data)/2 < 4 {
		fhtInverseKacWalkScalar(data)
		return
	}
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt(.5))
	}
	xvec_neon_fht_inverse_kac_walk(unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtInPlaceNEON(data []float32) {
	if len(data) < 8 {
		fhtInPlaceScalar(data)
		return
	}
	xvec_neon_fht_in_place(unsafe.Pointer(&data[0]), int64(len(data)))
}
