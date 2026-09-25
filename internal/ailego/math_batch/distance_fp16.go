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

import mathutil "github.com/gorse-io/xvec/internal/ailego/math"

type fp16Batch4Kernel func(query, first, second, third, fourth []uint16, output []float32)

var fp16Kernels = struct {
	l2, dot, cosine, mips fp16Batch4Kernel
}{fp16L2Scalar4, fp16DotScalar4, fp16CosineScalar4, fp16MIPSScalar4}

// SquaredEuclideanDistances4FP16 computes squared Euclidean distances from one query to four
// binary16 candidates, sharing query loads and conversions when supported.
// Inputs are unchecked: candidates must have the query dimension and output
// must contain at least four elements.
func SquaredEuclideanDistances4FP16(query, first, second, third, fourth []uint16, output []float32) {
	fp16Kernels.l2(query, first, second, third, fourth, output)
}

func fp16L2Scalar4(query, first, second, third, fourth []uint16, output []float32) {
	output[0] = mathutil.L2SquaredFP16(query, first)
	output[1] = mathutil.L2SquaredFP16(query, second)
	output[2] = mathutil.L2SquaredFP16(query, third)
	output[3] = mathutil.L2SquaredFP16(query, fourth)
}

// InnerProducts4FP16 computes inner products from one query to four
// binary16 candidates, sharing query loads and conversions when supported.
// Inputs are unchecked: candidates must have the query dimension and output
// must contain at least four elements.
func InnerProducts4FP16(query, first, second, third, fourth []uint16, output []float32) {
	fp16Kernels.dot(query, first, second, third, fourth, output)
}

func fp16DotScalar4(query, first, second, third, fourth []uint16, output []float32) {
	output[0] = mathutil.InnerProductFP16(query, first)
	output[1] = mathutil.InnerProductFP16(query, second)
	output[2] = mathutil.InnerProductFP16(query, third)
	output[3] = mathutil.InnerProductFP16(query, fourth)
}

// CosineDistances4FP16 computes cosine distances from one query to four
// binary16 candidates, sharing query loads and conversions when supported.
// Inputs are unchecked: candidates must have the query dimension and output
// must contain at least four elements.
func CosineDistances4FP16(query, first, second, third, fourth []uint16, output []float32) {
	fp16Kernels.cosine(query, first, second, third, fourth, output)
}

func fp16CosineScalar4(query, first, second, third, fourth []uint16, output []float32) {
	output[0] = mathutil.CosineDistanceFP16(query, first)
	output[1] = mathutil.CosineDistanceFP16(query, second)
	output[2] = mathutil.CosineDistanceFP16(query, third)
	output[3] = mathutil.CosineDistanceFP16(query, fourth)
}

// MIPSL2SquaredDistances4FP16 computes localized spherical MIPS distances from one query to four
// binary16 candidates, sharing query loads and conversions when supported.
// Inputs are unchecked: candidates must have the query dimension and output
// must contain at least four elements.
func MIPSL2SquaredDistances4FP16(query, first, second, third, fourth []uint16, output []float32) {
	fp16Kernels.mips(query, first, second, third, fourth, output)
}

func fp16MIPSScalar4(query, first, second, third, fourth []uint16, output []float32) {
	output[0] = mathutil.MIPSL2SquaredFP16(query, first)
	output[1] = mathutil.MIPSL2SquaredFP16(query, second)
	output[2] = mathutil.MIPSL2SquaredFP16(query, third)
	output[3] = mathutil.MIPSL2SquaredFP16(query, fourth)
}
