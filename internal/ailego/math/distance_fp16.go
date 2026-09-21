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

	"github.com/gorse-io/xvec/internal/ailego/utility"
)

type binaryKernelFP16 func(left, right []uint16) float32
type productsKernelFP16 func(left, right []uint16) (dot, leftNorm, rightNorm float32)

var kernelsFP16 = struct {
	l2       binaryKernelFP16
	dot      binaryKernelFP16
	products productsKernelFP16
}{
	l2:       squaredEuclideanFP16Scalar,
	dot:      innerProductFP16Scalar,
	products: dotNormsFP16Scalar,
}

// DenseDistanceFP16 computes an unchecked score for two dense binary16
// vectors. Callers must guarantee equal, non-zero dimensions and finite
// components. Arithmetic uses float32 accumulation.
type DenseDistanceFP16 func(left, right []uint16) float32

// L2SquaredFP16 computes unchecked squared Euclidean distance for binary16
// vectors using float32 accumulation.
func L2SquaredFP16(left, right []uint16) float32 {
	return kernelsFP16.l2(left, right)
}

// InnerProductFP16 computes unchecked dot-product similarity for binary16
// vectors. Higher scores are better.
func InnerProductFP16(left, right []uint16) float32 {
	return kernelsFP16.dot(left, right)
}

// CosineDistanceFP16 computes unchecked 1-cos(left,right) for binary16
// vectors. Lower scores are better.
func CosineDistanceFP16(left, right []uint16) float32 {
	inner, leftNorm, rightNorm := kernelsFP16.products(left, right)
	if leftNorm == 0 && rightNorm == 0 {
		return 0
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 1
	}
	leftMagnitude := float32(math.Sqrt(float64(leftNorm)))
	rightMagnitude := float32(math.Sqrt(float64(rightNorm)))
	return cosineDistanceFromProduct(inner, leftMagnitude, rightMagnitude)
}

// MIPSL2SquaredFP16 computes the unchecked baseline localized-spherical
// MIPS-to-L2 transformation for binary16 vectors. Lower scores are better.
func MIPSL2SquaredFP16(left, right []uint16) float32 {
	inner, leftNorm, rightNorm := kernelsFP16.products(left, right)
	denominator := max(leftNorm, rightNorm)
	if denominator == 0 {
		return 0
	}
	return 2 - 2*inner/denominator
}

// L2MagnitudeFP16 computes a binary16 vector magnitude.
func L2MagnitudeFP16(vector []uint16) float32 {
	norm := kernelsFP16.dot(vector, vector)
	if norm < 0 {
		norm = 0
	}
	return float32(math.Sqrt(float64(norm)))
}

// CosineDistanceWithMagnitudesFP16 computes binary16 cosine distance while
// reusing cached magnitudes.
func CosineDistanceWithMagnitudesFP16(
	left, right []uint16, leftMagnitude, rightMagnitude float32,
) float32 {
	return cosineDistanceFromProduct(
		kernelsFP16.dot(left, right), leftMagnitude, rightMagnitude,
	)
}

func squaredEuclideanFP16Scalar(left, right []uint16) (sum float32) {
	for index, leftBits := range left {
		leftValue := utility.Float16BitsToFloat32(leftBits)
		rightValue := utility.Float16BitsToFloat32(right[index])
		difference := leftValue - rightValue
		sum += difference * difference
	}
	return
}

func innerProductFP16Scalar(left, right []uint16) (sum float32) {
	for index, leftBits := range left {
		leftValue := utility.Float16BitsToFloat32(leftBits)
		rightValue := utility.Float16BitsToFloat32(right[index])
		sum += leftValue * rightValue
	}
	return
}

func dotNormsFP16Scalar(left, right []uint16) (dot, leftNorm, rightNorm float32) {
	for index, leftBits := range left {
		leftValue := utility.Float16BitsToFloat32(leftBits)
		rightValue := utility.Float16BitsToFloat32(right[index])
		dot += leftValue * rightValue
		leftNorm += leftValue * leftValue
		rightNorm += rightValue * rightValue
	}
	return
}
