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
	"math"
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:generate make avx2 avx512

func init() {
	switch {
	case cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ:
		activeFHTKernels = fhtKernels{
			flipSigns:      fhtFlipSignsAVX512,
			kacWalk:        fhtKacWalkAVX512,
			inverseKacWalk: fhtInverseKacWalkAVX512,
			inPlace:        fhtInPlaceAVX512,
		}
	case cpu.X86.HasAVX2:
		activeFHTKernels = fhtKernels{
			flipSigns:      fhtFlipSignsAVX2,
			kacWalk:        fhtKacWalkAVX2,
			inverseKacWalk: fhtInverseKacWalkAVX2,
			inPlace:        fhtInPlaceAVX2,
		}
	}
}

func fhtFlipSignsAVX2(signs []byte, data []float32) {
	if len(data) < 8 {
		fhtFlipSignsScalar(signs, data)
		return
	}
	xvec_avx2_fht_flip_signs(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtKacWalkAVX2(data []float32) {
	if len(data)/2 < 8 {
		fhtKacWalkScalar(data)
		return
	}
	xvec_avx2_fht_kac_walk(unsafe.Pointer(&data[0]), int64(len(data)))
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt2)
	}
}

func fhtInverseKacWalkAVX2(data []float32) {
	if len(data)/2 < 8 {
		fhtInverseKacWalkScalar(data)
		return
	}
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt(.5))
	}
	xvec_avx2_fht_inverse_kac_walk(unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtInPlaceAVX2(data []float32) {
	if len(data) < 16 {
		fhtInPlaceScalar(data)
		return
	}
	xvec_avx2_fht_in_place(unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtFlipSignsAVX512(signs []byte, data []float32) {
	if len(data) < 16 {
		fhtFlipSignsScalar(signs, data)
		return
	}
	xvec_avx512_fht_flip_signs(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtKacWalkAVX512(data []float32) {
	if len(data)/2 < 16 {
		fhtKacWalkScalar(data)
		return
	}
	xvec_avx512_fht_kac_walk(unsafe.Pointer(&data[0]), int64(len(data)))
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt2)
	}
}

func fhtInverseKacWalkAVX512(data []float32) {
	if len(data)/2 < 16 {
		fhtInverseKacWalkScalar(data)
		return
	}
	if len(data)%2 != 0 {
		data[len(data)/2] *= float32(math.Sqrt(.5))
	}
	xvec_avx512_fht_inverse_kac_walk(unsafe.Pointer(&data[0]), int64(len(data)))
}

func fhtInPlaceAVX512(data []float32) {
	if len(data) < 32 {
		fhtInPlaceScalar(data)
		return
	}
	xvec_avx512_fht_in_place(unsafe.Pointer(&data[0]), int64(len(data)))
}
