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
	"errors"
	"math"

	"github.com/gorse-io/xvec/internal/ailego/utility"
)

var (
	ErrDimensionMismatch  = errors.New("ailego: vector dimension mismatch")
	ErrEmptyVector        = errors.New("ailego: vector is empty")
	ErrNonFiniteVector    = errors.New("ailego: vector contains a non-finite value")
	ErrInvalidSparseOrder = errors.New("ailego: sparse indices must be strictly increasing")
)

type binaryKernel func(left, right []float32) float32
type productsKernel func(left, right []float32) (dot, leftNorm, rightNorm float32)

var kernels = struct {
	l2       binaryKernel
	dot      binaryKernel
	products productsKernel
}{
	l2:       squaredEuclideanScalar,
	dot:      innerProductScalar,
	products: dotNormsScalar,
}

// DenseDistance computes an unchecked score for two dense vectors. Callers
// must guarantee equal, non-zero dimensions and finite components. The result
// may be non-finite if arithmetic overflows.
type DenseDistance func(left, right []float32) float32

// L2Squared computes unchecked squared Euclidean distance using float32
// accumulation.
func L2Squared(left, right []float32) float32 {
	return squaredEuclidean(left, right)
}

// InnerProduct computes unchecked dot-product similarity. Higher scores are
// better.
func InnerProduct(left, right []float32) float32 {
	return innerProduct(left, right)
}

// CosineDistance computes unchecked 1-cos(left,right). Lower scores are better.
// Two zero vectors have distance 0; exactly one zero vector has distance 1.
func CosineDistance(left, right []float32) float32 {
	inner, leftNorm, rightNorm := dotNorms(left, right)
	if leftNorm == 0 && rightNorm == 0 {
		return 0
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 1
	}
	leftMagnitude := float32(math.Sqrt(float64(leftNorm)))
	rightMagnitude := float32(math.Sqrt(float64(rightNorm)))
	cosine := inner / (leftMagnitude * rightMagnitude)
	cosine = min(1, max(-1, cosine))
	return 1 - cosine
}

// MIPSL2Squared computes the unchecked baseline localized-spherical MIPS-to-L2
// transformation: 2 - 2*ip/max(norm(left)^2,norm(right)^2). Lower scores are
// better. Two zero vectors have distance 0.
func MIPSL2Squared(left, right []float32) float32 {
	inner, leftNorm, rightNorm := dotNorms(left, right)
	denominator := max(leftNorm, rightNorm)
	if denominator == 0 {
		return 0
	}
	return 2 - 2*inner/denominator
}

// L2Magnitude computes a vector magnitude.
func L2Magnitude(vector []float32) float32 {
	norm := innerProduct(vector, vector)
	if norm < 0 {
		norm = 0
	}
	return float32(math.Sqrt(float64(norm)))
}

// CosineDistanceWithMagnitudes computes cosine distance while reusing cached
// magnitudes, reducing every candidate score to one dot product.
func CosineDistanceWithMagnitudes(left, right []float32, leftMagnitude, rightMagnitude float32) float32 {
	if leftMagnitude == 0 && rightMagnitude == 0 {
		return 0
	}
	if leftMagnitude == 0 || rightMagnitude == 0 {
		return 1
	}
	return cosineDistanceFromProduct(innerProduct(left, right), leftMagnitude, rightMagnitude)
}

func cosineDistanceFromProduct(product, leftMagnitude, rightMagnitude float32) float32 {
	if leftMagnitude == 0 && rightMagnitude == 0 {
		return 0
	}
	if leftMagnitude == 0 || rightMagnitude == 0 {
		return 1
	}
	cosine := product / (leftMagnitude * rightMagnitude)
	cosine = min(1, max(-1, cosine))
	return 1 - cosine
}

func squaredEuclidean(left, right []float32) float32 {
	return kernels.l2(left, right)
}

func innerProduct(left, right []float32) float32 {
	return kernels.dot(left, right)
}

func dotNorms(left, right []float32) (dot, leftNorm, rightNorm float32) {
	return kernels.products(left, right)
}

func squaredEuclideanScalar(left, right []float32) (sum float32) {
	for index, leftValue := range left {
		difference := leftValue - right[index]
		sum += difference * difference
	}
	return
}

func innerProductScalar(left, right []float32) (sum float32) {
	for index, leftValue := range left {
		sum += leftValue * right[index]
	}
	return
}

func dotNormsScalar(left, right []float32) (dot, leftNorm, rightNorm float32) {
	for index, leftValue := range left {
		rightValue := right[index]
		dot += leftValue * rightValue
		leftNorm += leftValue * leftValue
		rightNorm += rightValue * rightValue
	}
	return
}

// SparseInnerProduct computes the dot product of canonical sparse vectors.
func SparseInnerProduct(
	leftIndices []uint32,
	leftValues []float32,
	rightIndices []uint32,
	rightValues []float32,
) (float32, error) {
	if err := validateSparse(leftIndices, leftValues); err != nil {
		return 0, err
	}
	if err := validateSparse(rightIndices, rightValues); err != nil {
		return 0, err
	}
	var sum float64
	leftIndex, rightIndex := 0, 0
	for leftIndex < len(leftIndices) && rightIndex < len(rightIndices) {
		switch {
		case leftIndices[leftIndex] < rightIndices[rightIndex]:
			leftIndex++
		case leftIndices[leftIndex] > rightIndices[rightIndex]:
			rightIndex++
		default:
			sum += float64(leftValues[leftIndex]) * float64(rightValues[rightIndex])
			leftIndex++
			rightIndex++
		}
	}
	return finiteScore(sum)
}

// ValidateDense checks one dense vector at an API or storage boundary.
func ValidateDense(vector []float32, dimension int) error {
	if len(vector) != dimension {
		return ErrDimensionMismatch
	}
	if len(vector) == 0 {
		return ErrEmptyVector
	}
	for _, value := range vector {
		if !finite32(value) {
			return ErrNonFiniteVector
		}
	}
	return nil
}

func validateSparse(indices []uint32, values []float32) error {
	if len(indices) != len(values) {
		return ErrDimensionMismatch
	}
	for index, value := range values {
		if !finite32(value) {
			return ErrNonFiniteVector
		}
		if index > 0 && indices[index] <= indices[index-1] {
			return ErrInvalidSparseOrder
		}
	}
	return nil
}

func finite32(value float32) bool {
	return !float32IsNaN(value) && !float32IsInf(value)
}

func float32IsNaN(value float32) bool { return value != value }

func float32IsInf(value float32) bool {
	bits := math.Float32bits(value) & 0x7fffffff
	return bits == 0x7f800000
}

func finiteScore(value float64) (float32, error) {
	score := float32(value)
	if math.IsNaN(value) || math.IsInf(value, 0) || !finite32(score) {
		return 0, ErrNonFiniteVector
	}
	return score, nil
}

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

type binaryKernelInteger func(left, right []byte) int64
type productsKernelInteger func(left, right []byte) (dot, leftNorm, rightNorm int64)

var kernelsInt4 = struct {
	l2       binaryKernelInteger
	dot      binaryKernelInteger
	products productsKernelInteger
}{
	l2:       squaredEuclideanInt4Scalar,
	dot:      innerProductInt4Scalar,
	products: dotNormsInt4Scalar,
}

var kernelsInt8 = struct {
	l2       binaryKernelInteger
	dot      binaryKernelInteger
	products productsKernelInteger
}{
	l2:       squaredEuclideanInt8Scalar,
	dot:      innerProductInt8Scalar,
	products: dotNormsInt8Scalar,
}

// InnerProductInt4 computes the unchecked dot product of packed signed INT4
// codes. Each byte stores the low-nibble value first and the high-nibble value
// second. Callers must guarantee equal byte lengths.
func InnerProductInt4(left, right []byte) int64 {
	return kernelsInt4.dot(left, right)
}

// L2SquaredInt4 computes unchecked squared Euclidean distance between packed
// signed INT4 codes. Callers must guarantee equal byte lengths.
func L2SquaredInt4(left, right []byte) int64 {
	return kernelsInt4.l2(left, right)
}

// MIPSSphericalL2SquaredInt4 computes the unchecked spherical MIPS-to-L2
// transform for packed signed INT4 codes. Callers must guarantee equal byte
// lengths. An inverseMaxSquaredNorm of zero selects localized spherical
// injection.
func MIPSSphericalL2SquaredInt4(left, right []byte, inverseMaxSquaredNorm float32) float32 {
	dot, leftNorm, rightNorm := kernelsInt4.products(left, right)
	return sphericalMIPSL2Squared(dot, leftNorm, rightNorm, inverseMaxSquaredNorm)
}

// MIPSRepeatedQuadraticL2SquaredInt4 computes the unchecked repeated-quadratic
// MIPS-to-L2 transform for packed signed INT4 codes. Callers must guarantee
// equal byte lengths.
func MIPSRepeatedQuadraticL2SquaredInt4(
	left, right []byte, repetitions int, inverseMaxSquaredNorm float32,
) float32 {
	dot, leftNorm, rightNorm := kernelsInt4.products(left, right)
	return repeatedQuadraticMIPSL2Squared(dot, leftNorm, rightNorm, repetitions, inverseMaxSquaredNorm)
}

// InnerProductInt8 computes the unchecked dot product of signed INT8 codes
// stored in byte slices. Callers must guarantee equal lengths. Accumulation
// uses int64 so dimensions whose dot product exceeds int32 remain exact.
func InnerProductInt8(left, right []byte) int64 {
	return kernelsInt8.dot(left, right)
}

// L2SquaredInt8 computes unchecked squared Euclidean distance between signed
// INT8 codes stored in byte slices. Callers must guarantee equal lengths.
func L2SquaredInt8(left, right []byte) int64 {
	return kernelsInt8.l2(left, right)
}

// MIPSSphericalL2SquaredInt8 computes the unchecked spherical MIPS-to-L2
// transform for signed INT8 codes. Callers must guarantee equal byte lengths.
// An inverseMaxSquaredNorm of zero selects localized spherical injection.
func MIPSSphericalL2SquaredInt8(left, right []byte, inverseMaxSquaredNorm float32) float32 {
	dot, leftNorm, rightNorm := kernelsInt8.products(left, right)
	return sphericalMIPSL2Squared(dot, leftNorm, rightNorm, inverseMaxSquaredNorm)
}

// MIPSRepeatedQuadraticL2SquaredInt8 computes the unchecked repeated-quadratic
// MIPS-to-L2 transform for signed INT8 codes. Callers must guarantee equal byte
// lengths.
func MIPSRepeatedQuadraticL2SquaredInt8(
	left, right []byte, repetitions int, inverseMaxSquaredNorm float32,
) float32 {
	dot, leftNorm, rightNorm := kernelsInt8.products(left, right)
	return repeatedQuadraticMIPSL2Squared(dot, leftNorm, rightNorm, repetitions, inverseMaxSquaredNorm)
}

func innerProductInt4Scalar(left, right []byte) (sum int64) {
	dot, _, _ := dotNormsInt4Scalar(left, right)
	return dot
}

func squaredEuclideanInt4Scalar(left, right []byte) (sum int64) {
	for i, value := range left {
		leftLow, leftHigh := unpackInt4(value)
		rightLow, rightHigh := unpackInt4(right[i])
		lowDifference := leftLow - rightLow
		highDifference := leftHigh - rightHigh
		sum += lowDifference*lowDifference + highDifference*highDifference
	}
	return sum
}

func dotNormsInt4(left, right []byte) (dot, leftNorm, rightNorm int64) {
	return kernelsInt4.products(left, right)
}

func dotNormsInt4Scalar(left, right []byte) (dot, leftNorm, rightNorm int64) {
	for i, value := range left {
		leftLow, leftHigh := unpackInt4(value)
		rightLow, rightHigh := unpackInt4(right[i])
		dot += leftLow*rightLow + leftHigh*rightHigh
		leftNorm += leftLow*leftLow + leftHigh*leftHigh
		rightNorm += rightLow*rightLow + rightHigh*rightHigh
	}
	return
}

func unpackInt4(value byte) (low, high int64) {
	return int64(int8(value<<4) >> 4), int64(int8(value) >> 4)
}

func innerProductInt8Scalar(left, right []byte) (sum int64) {
	dot, _, _ := dotNormsInt8Scalar(left, right)
	return dot
}

func squaredEuclideanInt8Scalar(left, right []byte) (sum int64) {
	for i, value := range left {
		difference := int64(int8(value)) - int64(int8(right[i]))
		sum += difference * difference
	}
	return sum
}

func dotNormsInt8(left, right []byte) (dot, leftNorm, rightNorm int64) {
	return kernelsInt8.products(left, right)
}

func dotNormsInt8Scalar(left, right []byte) (dot, leftNorm, rightNorm int64) {
	for i, value := range left {
		leftValue := int64(int8(value))
		rightValue := int64(int8(right[i]))
		dot += leftValue * rightValue
		leftNorm += leftValue * leftValue
		rightNorm += rightValue * rightValue
	}
	return
}

func sphericalMIPSL2Squared(dot, leftNorm, rightNorm int64, inverseMaxSquaredNorm float32) float32 {
	if inverseMaxSquaredNorm == 0 {
		denominator := max(leftNorm, rightNorm)
		if denominator == 0 {
			return 0
		}
		return 2 - 2*float32(dot)/float32(denominator)
	}

	e2 := float64(inverseMaxSquaredNorm)
	value := (1 - e2*float64(leftNorm)) * (1 - e2*float64(rightNorm))
	score := 1 - e2*float64(dot)
	if value > 0 {
		score -= math.Sqrt(value)
	}
	return float32(2 * score)
}

func repeatedQuadraticMIPSL2Squared(
	dot, leftNorm, rightNorm int64, repetitions int, inverseMaxSquaredNorm float32,
) float32 {
	left := float32(leftNorm) * inverseMaxSquaredNorm
	right := float32(rightNorm) * inverseMaxSquaredNorm
	sum := float32(leftNorm+rightNorm-2*dot) * inverseMaxSquaredNorm
	for range repetitions {
		difference := left - right
		sum += difference * difference
		left *= left
		right *= right
	}
	return sum
}
