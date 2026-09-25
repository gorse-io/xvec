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
	"github.com/gorse-io/xvec/internal/ailego/math_batch"
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

// fp16CodeDistances scores immutable, validated code buffers four at a time.
// Native views stay on the stack; unaligned or big-endian codes retain the
// single-pair portable path. Remaining candidates use the single-pair kernel.
func fp16CodeDistances(metric Metric, query []byte, candidates [][]byte, output []float32) {
	q, queryOK := fp16CodeWords(query)
	var batch func(query, first, second, third, fourth []uint16, output []float32)
	switch metric {
	case MetricL2:
		batch = mathbatch.SquaredEuclideanDistances4FP16
	case MetricIP:
		batch = mathbatch.InnerProducts4FP16
	case MetricCosine:
		batch = mathbatch.CosineDistances4FP16
	case MetricMIPSL2:
		batch = mathbatch.MIPSL2SquaredDistances4FP16
	}
	j := 0
	if queryOK {
		for ; j+4 <= len(candidates); j += 4 {
			first, ok1 := fp16CodeWords(candidates[j])
			second, ok2 := fp16CodeWords(candidates[j+1])
			third, ok3 := fp16CodeWords(candidates[j+2])
			fourth, ok4 := fp16CodeWords(candidates[j+3])
			if ok1 && ok2 && ok3 && ok4 {
				batch(q, first, second, third, fourth, output[j:])
			} else {
				for k := j; k < j+4; k++ {
					output[k] = fp16CodeDistance(metric, query, candidates[k])
				}
			}
		}
	}
	for ; j < len(candidates); j++ {
		output[j] = fp16CodeDistance(metric, query, candidates[j])
	}
}
