//go:build !noasm && loong64

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
	if cpu.Loong64.HasLASX {
		activeFHTKernels = fhtKernels{
			flipSigns:      fhtFlipSignsLASX,
			kacWalk:        fhtKacWalkLASX,
			inverseKacWalk: fhtInverseKacWalkLASX,
			inPlace:        fhtInPlaceLASX,
		}
	}
}

func fhtFlipSignsLASX(signs []byte, data []float32) {
	if len(data) < 8 {
		fhtFlipSignsScalar(signs, data)
		return
	}
	fht_flip_sign_lasx(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtKacWalkLASX(data []float32) {
	if len(data)/2 < 8 {
		fhtKacWalkScalar(data)
		return
	}
	fht_kacs_walk_lasx(unsafe.Pointer(&data[0]), int64(len(data)))
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt2)
	}
}

func fhtInverseKacWalkLASX(data []float32) {
	if len(data)/2 < 8 {
		fhtInverseKacWalkScalar(data)
		return
	}
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt(.5))
	}
	fht_inv_kacs_walk_lasx(unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtInPlaceLASX(data []float32) {
	if len(data) < 16 {
		fhtInPlaceScalar(data)
		return
	}
	fht_inplace_lasx(unsafe.Pointer(&data[0]), int64(len(data)))
}
