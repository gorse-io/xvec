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

package core

import (
	"encoding/binary"
	"math"
	"unsafe"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/utility"
)

// fp16CodeDistance scores validated, finite, equally sized little-endian
// binary16 codes without decoding to temporary FP32 vectors. Storage and query
// constructors establish these invariants; public entry points validate first.
func fp16CodeDistance(metric Metric, left, right []byte) float32 {
	l, leftOK := fp16CodeWords(left)
	r, rightOK := fp16CodeWords(right)
	if leftOK && rightOK {
		switch metric {
		case MetricL2:
			return mathutil.L2SquaredFP16(l, r)
		case MetricIP:
			return mathutil.InnerProductFP16(l, r)
		case MetricCosine:
			return mathutil.CosineDistanceFP16(l, r)
		case MetricMIPSL2:
			return mathutil.MIPSL2SquaredFP16(l, r)
		}
	}
	return fp16CodeDistanceScalar(metric, left, right)
}

// The code format is always little-endian. Native views are read-only and
// require compatible byte order and alignment, including for scalar kernels
// on architectures without SIMD. Other layouts use the byte-wise fallback.
func fp16CodeWords(codes []byte) ([]uint16, bool) {
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 || uintptr(unsafe.Pointer(unsafe.SliceData(codes)))%unsafe.Alignof(uint16(0)) != 0 {
		return nil, false
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(codes))), len(codes)/2), true
}

func fp16CodeDistanceScalar(metric Metric, left, right []byte) float32 {
	var squared, inner, leftNorm, rightNorm float32
	for i := 0; i < len(left); i += 2 {
		l := utility.Float16BitsToFloat32(binary.LittleEndian.Uint16(left[i:]))
		r := utility.Float16BitsToFloat32(binary.LittleEndian.Uint16(right[i:]))
		switch metric {
		case MetricL2:
			delta := l - r
			squared += delta * delta
		case MetricIP:
			inner += l * r
		default:
			inner += l * r
			leftNorm += l * l
			rightNorm += r * r
		}
	}
	switch metric {
	case MetricL2:
		return squared
	case MetricIP:
		return inner
	case MetricCosine:
		if leftNorm == 0 && rightNorm == 0 {
			return 0
		}
		if leftNorm == 0 || rightNorm == 0 {
			return 1
		}
		leftMagnitude := float32(math.Sqrt(float64(leftNorm)))
		rightMagnitude := float32(math.Sqrt(float64(rightNorm)))
		return 1 - min(float32(1), max(float32(-1), inner/(leftMagnitude*rightMagnitude)))
	case MetricMIPSL2:
		denominator := max(leftNorm, rightNorm)
		if denominator == 0 {
			return 0
		}
		return 2 - 2*inner/denominator
	default:
		panic("invalid metric for validated FP16 codes")
	}
}
