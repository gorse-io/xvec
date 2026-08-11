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

package ailego

import (
	"errors"
	"math"
)

var (
	ErrDimensionMismatch  = errors.New("ailego: vector dimension mismatch")
	ErrEmptyVector        = errors.New("ailego: vector is empty")
	ErrNonFiniteVector    = errors.New("ailego: vector contains a non-finite value")
	ErrInvalidSparseOrder = errors.New("ailego: sparse indices must be strictly increasing")
)

// L2Squared computes squared Euclidean distance using float64 accumulation and
// returns the float32 score used by indexes.
func L2Squared(left, right []float32) (float32, error) {
	if err := validateDensePair(left, right); err != nil {
		return 0, err
	}
	var sum float64
	for index, leftValue := range left {
		rightValue := right[index]
		if !finite32(leftValue) || !finite32(rightValue) {
			return 0, ErrNonFiniteVector
		}
		difference := float64(leftValue) - float64(rightValue)
		sum += difference * difference
	}
	return finiteScore(sum)
}

// InnerProduct computes the dot-product similarity. Higher scores are better.
func InnerProduct(left, right []float32) (float32, error) {
	if err := validateDensePair(left, right); err != nil {
		return 0, err
	}
	var sum float64
	for index, leftValue := range left {
		rightValue := right[index]
		if !finite32(leftValue) || !finite32(rightValue) {
			return 0, ErrNonFiniteVector
		}
		sum += float64(leftValue) * float64(rightValue)
	}
	return finiteScore(sum)
}

// CosineDistance computes 1-cos(left,right). Lower scores are better. Two zero
// vectors have distance 0; exactly one zero vector has distance 1.
func CosineDistance(left, right []float32) (float32, error) {
	if err := validateDensePair(left, right); err != nil {
		return 0, err
	}
	inner, leftNorm, rightNorm, err := denseProducts(left, right)
	if err != nil {
		return 0, err
	}
	if leftNorm == 0 && rightNorm == 0 {
		return 0, nil
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 1, nil
	}
	cosine := inner / math.Sqrt(leftNorm*rightNorm)
	cosine = min(1, max(-1, cosine))
	return finiteScore(1 - cosine)
}

// MIPSL2Squared computes the baseline localized-spherical MIPS-to-L2
// transformation: 2 - 2*ip/max(norm(left)^2,norm(right)^2). Lower scores are
// better. Two zero vectors have distance 0.
func MIPSL2Squared(left, right []float32) (float32, error) {
	if err := validateDensePair(left, right); err != nil {
		return 0, err
	}
	inner, leftNorm, rightNorm, err := denseProducts(left, right)
	if err != nil {
		return 0, err
	}
	denominator := max(leftNorm, rightNorm)
	if denominator == 0 {
		return 0, nil
	}
	return finiteScore(2 - 2*inner/denominator)
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

func validateDensePair(left, right []float32) error {
	if len(left) != len(right) {
		return ErrDimensionMismatch
	}
	if len(left) == 0 {
		return ErrEmptyVector
	}
	return nil
}

func denseProducts(left, right []float32) (inner, leftNorm, rightNorm float64, err error) {
	for index, leftValue := range left {
		rightValue := right[index]
		if !finite32(leftValue) || !finite32(rightValue) {
			return 0, 0, 0, ErrNonFiniteVector
		}
		leftFloat := float64(leftValue)
		rightFloat := float64(rightValue)
		inner += leftFloat * rightFloat
		leftNorm += leftFloat * leftFloat
		rightNorm += rightFloat * rightFloat
	}
	return inner, leftNorm, rightNorm, nil
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
