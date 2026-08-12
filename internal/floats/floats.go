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

// Package floats provides allocation-free float32 vector kernels. Callers must
// pass equal, non-empty slices; validation belongs at the API boundary.
package floats

type binaryKernel func(left, right []float32) float32
type productsKernel func(left, right []float32) (dot, leftNorm, rightNorm float32)

var kernels = struct {
	l2       binaryKernel
	dot      binaryKernel
	products productsKernel
}{
	l2:       l2SquaredScalar,
	dot:      innerProductScalar,
	products: dotNormsScalar,
}

// L2Squared returns the squared Euclidean distance between left and right.
func L2Squared(left, right []float32) float32 {
	return kernels.l2(left, right)
}

// InnerProduct returns the dot product of left and right.
func InnerProduct(left, right []float32) float32 {
	return kernels.dot(left, right)
}

// DotNorms computes the dot product and both squared norms in one pass.
func DotNorms(left, right []float32) (dot, leftNorm, rightNorm float32) {
	return kernels.products(left, right)
}

func l2SquaredScalar(left, right []float32) (sum float32) {
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
