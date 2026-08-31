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

// The distance utility functions provide allocation-free float32 vector kernels.
// Callers must pass equal, non-empty slices; validation belongs at the API boundary.
package mathutil

type binaryKernel func(left, right []float32) float32
type batch2Kernel func(query, first, second []float32) (firstProduct, secondProduct float32)
type batch4Kernel func(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32)
type productsKernel func(left, right []float32) (dot, leftNorm, rightNorm float32)

var kernels = struct {
	l2       binaryKernel
	dot      binaryKernel
	dot2     batch2Kernel
	dot4     batch4Kernel
	products productsKernel
}{
	l2:       squaredEuclideanScalar,
	dot:      innerProductScalar,
	dot2:     innerProducts2Scalar,
	dot4:     innerProducts4Scalar,
	products: dotNormsScalar,
}

// squaredEuclidean returns the squared Euclidean distance between left and right.
func squaredEuclidean(left, right []float32) float32 {
	return kernels.l2(left, right)
}

// innerProduct returns the dot product of left and right.
func innerProduct(left, right []float32) float32 {
	return kernels.dot(left, right)
}

// innerProducts2 computes the dot product of one query with two candidates in
// one pass. SIMD implementations reuse each loaded query block for both
// candidates, which is the dominant HNSW one-to-many scoring pattern.
func innerProducts2(query, first, second []float32) (firstProduct, secondProduct float32) {
	return kernels.dot2(query, first, second)
}

// innerProducts4 computes the dot product of one query with four candidates
// in one pass, amortizing each query load across four independent products.
func innerProducts4(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32) {
	return kernels.dot4(query, first, second, third, fourth)
}

// dotNorms computes the dot product and both squared norms in one pass.
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

func innerProducts2Scalar(query, first, second []float32) (firstProduct, secondProduct float32) {
	for index, queryValue := range query {
		firstProduct += queryValue * first[index]
		secondProduct += queryValue * second[index]
	}
	return
}

func innerProducts4Scalar(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32) {
	for index, queryValue := range query {
		firstProduct += queryValue * first[index]
		secondProduct += queryValue * second[index]
		thirdProduct += queryValue * third[index]
		fourthProduct += queryValue * fourth[index]
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
