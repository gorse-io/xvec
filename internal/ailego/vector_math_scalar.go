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

import "math"

// DenseDistance computes a score for two already validated dense vectors.
// Callers must guarantee equal, non-zero dimensions and finite components.
type DenseDistance func(left, right []float32) (float32, error)

type denseDistanceKernel func(left, right []float32) float32

// Keep the dispatch table separate from the checked API so architecture files
// can select SIMD kernels later without changing index code. The baseline uses
// portable scalar implementations on every platform.
var denseDistanceKernels = struct {
	l2     denseDistanceKernel
	ip     denseDistanceKernel
	cosine denseDistanceKernel
	mipsL2 denseDistanceKernel
}{
	l2:     l2SquaredScalar,
	ip:     innerProductScalar,
	cosine: cosineDistanceScalar,
	mipsL2: mipsL2SquaredScalar,
}

// L2SquaredPrevalidated computes squared Euclidean distance without validating
// its inputs. It is intended for index hot paths whose storage boundary has
// already validated every vector.
func L2SquaredPrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(denseDistanceKernels.l2(left, right)))
}

// InnerProductPrevalidated computes inner product without validating inputs.
func InnerProductPrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(denseDistanceKernels.ip(left, right)))
}

// CosineDistancePrevalidated computes cosine distance without validating inputs.
func CosineDistancePrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(denseDistanceKernels.cosine(left, right)))
}

// MIPSL2SquaredPrevalidated computes MIPS-to-L2 distance without validating inputs.
func MIPSL2SquaredPrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(denseDistanceKernels.mipsL2(left, right)))
}

func l2SquaredScalar(left, right []float32) float32 {
	var sum float64
	for index, leftValue := range left {
		difference := float64(leftValue) - float64(right[index])
		sum += difference * difference
	}
	return float32(sum)
}

func innerProductScalar(left, right []float32) float32 {
	var sum float64
	for index, leftValue := range left {
		sum += float64(leftValue) * float64(right[index])
	}
	return float32(sum)
}

func cosineDistanceScalar(left, right []float32) float32 {
	inner, leftNorm, rightNorm := denseProductsScalar(left, right)
	if leftNorm == 0 && rightNorm == 0 {
		return 0
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 1
	}
	cosine := inner / math.Sqrt(leftNorm*rightNorm)
	cosine = min(1, max(-1, cosine))
	return float32(1 - cosine)
}

func mipsL2SquaredScalar(left, right []float32) float32 {
	inner, leftNorm, rightNorm := denseProductsScalar(left, right)
	denominator := max(leftNorm, rightNorm)
	if denominator == 0 {
		return 0
	}
	return float32(2 - 2*inner/denominator)
}

func denseProductsScalar(left, right []float32) (inner, leftNorm, rightNorm float64) {
	for index, leftValue := range left {
		leftFloat := float64(leftValue)
		rightFloat := float64(right[index])
		inner += leftFloat * rightFloat
		leftNorm += leftFloat * leftFloat
		rightNorm += rightFloat * rightFloat
	}
	return inner, leftNorm, rightNorm
}
