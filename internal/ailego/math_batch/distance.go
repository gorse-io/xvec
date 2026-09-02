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

import "math"

type batch2Kernel func(query, first, second []float32) (firstProduct, secondProduct float32)
type batch4Kernel func(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32)

var kernels = struct {
	dot2       batch2Kernel
	dot4       batch4Kernel
	l2Squared2 batch2Kernel
	l2Squared4 batch4Kernel
}{
	dot2:       innerProducts2Scalar,
	dot4:       innerProducts4Scalar,
	l2Squared2: squaredEuclideanDistances2Scalar,
	l2Squared4: squaredEuclideanDistances4Scalar,
}

// InnerProducts2 computes inner products from one query to two candidates
// while sharing each query load.
func InnerProducts2(query, first, second []float32) (firstProduct, secondProduct float32) {
	return kernels.dot2(query, first, second)
}

// InnerProducts4 computes inner products from one query to four candidates
// while sharing each query load.
func InnerProducts4(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32) {
	return kernels.dot4(query, first, second, third, fourth)
}

// SquaredEuclideanDistances2 computes squared Euclidean distance from one
// query to two candidates while sharing each query load.
func SquaredEuclideanDistances2(query, first, second []float32) (firstDistance, secondDistance float32) {
	return kernels.l2Squared2(query, first, second)
}

// SquaredEuclideanDistances4 computes squared Euclidean distance from one
// query to four candidates while sharing each query load.
func SquaredEuclideanDistances4(query, first, second, third, fourth []float32) (firstDistance, secondDistance, thirdDistance, fourthDistance float32) {
	return kernels.l2Squared4(query, first, second, third, fourth)
}

// EuclideanDistances2 computes Euclidean distance from one query to two
// candidates while sharing each query load.
func EuclideanDistances2(query, first, second []float32) (firstDistance, secondDistance float32) {
	firstDistance, secondDistance = SquaredEuclideanDistances2(query, first, second)
	return float32(math.Sqrt(float64(firstDistance))), float32(math.Sqrt(float64(secondDistance)))
}

// EuclideanDistances4 computes Euclidean distance from one query to four
// candidates while sharing each query load.
func EuclideanDistances4(query, first, second, third, fourth []float32) (firstDistance, secondDistance, thirdDistance, fourthDistance float32) {
	firstDistance, secondDistance, thirdDistance, fourthDistance = SquaredEuclideanDistances4(query, first, second, third, fourth)
	return float32(math.Sqrt(float64(firstDistance))),
		float32(math.Sqrt(float64(secondDistance))),
		float32(math.Sqrt(float64(thirdDistance))),
		float32(math.Sqrt(float64(fourthDistance)))
}

// CosineDistances2WithMagnitudes computes cosine distance from one query to two
// candidates while sharing the query load in the SIMD dot-product kernel.
func CosineDistances2WithMagnitudes(
	query, first, second []float32,
	queryMagnitude, firstMagnitude, secondMagnitude float32,
) (firstDistance, secondDistance float32) {
	firstProduct, secondProduct := InnerProducts2(query, first, second)
	return cosineDistanceFromProduct(firstProduct, queryMagnitude, firstMagnitude),
		cosineDistanceFromProduct(secondProduct, queryMagnitude, secondMagnitude)
}

// CosineDistances4WithMagnitudes computes cosine distance from one query to
// four candidates while sharing query loads in the SIMD dot-product kernel.
func CosineDistances4WithMagnitudes(
	query, first, second, third, fourth []float32,
	queryMagnitude, firstMagnitude, secondMagnitude, thirdMagnitude, fourthMagnitude float32,
) (firstDistance, secondDistance, thirdDistance, fourthDistance float32) {
	firstProduct, secondProduct, thirdProduct, fourthProduct := InnerProducts4(query, first, second, third, fourth)
	return cosineDistanceFromProduct(firstProduct, queryMagnitude, firstMagnitude),
		cosineDistanceFromProduct(secondProduct, queryMagnitude, secondMagnitude),
		cosineDistanceFromProduct(thirdProduct, queryMagnitude, thirdMagnitude),
		cosineDistanceFromProduct(fourthProduct, queryMagnitude, fourthMagnitude)
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

func innerProducts2(query, first, second []float32) (firstProduct, secondProduct float32) {
	return InnerProducts2(query, first, second)
}

func innerProducts4(query, first, second, third, fourth []float32) (firstProduct, secondProduct, thirdProduct, fourthProduct float32) {
	return InnerProducts4(query, first, second, third, fourth)
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

func squaredEuclideanDistances2Scalar(query, first, second []float32) (firstDistance, secondDistance float32) {
	for index, queryValue := range query {
		firstDifference := queryValue - first[index]
		secondDifference := queryValue - second[index]
		firstDistance += firstDifference * firstDifference
		secondDistance += secondDifference * secondDifference
	}
	return
}

func squaredEuclideanDistances4Scalar(query, first, second, third, fourth []float32) (firstDistance, secondDistance, thirdDistance, fourthDistance float32) {
	for index, queryValue := range query {
		firstDifference := queryValue - first[index]
		secondDifference := queryValue - second[index]
		thirdDifference := queryValue - third[index]
		fourthDifference := queryValue - fourth[index]
		firstDistance += firstDifference * firstDifference
		secondDistance += secondDifference * secondDifference
		thirdDistance += thirdDifference * thirdDifference
		fourthDistance += fourthDifference * fourthDifference
	}
	return
}
