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
)

// DenseDistance computes a score for two already validated dense vectors.
// Callers must guarantee equal, non-zero dimensions and finite components.
// The returned score is unchecked and may be non-finite if arithmetic overflows.
type DenseDistance func(left, right []float32) float32

// L2SquaredPrevalidated computes squared Euclidean distance without validating
// its inputs. It is intended for index hot paths whose storage boundary has
// already validated every vector.
func L2SquaredPrevalidated(left, right []float32) float32 {
	return squaredEuclidean(left, right)
}

// InnerProductPrevalidated computes inner product without validating inputs.
func InnerProductPrevalidated(left, right []float32) float32 {
	return innerProduct(left, right)
}

// CosineDistancePrevalidated computes cosine distance without validating inputs.
func CosineDistancePrevalidated(left, right []float32) float32 {
	return cosineDistance(left, right)
}

// L2MagnitudePrevalidated computes a vector magnitude without validating its
// components. It is intended for indexes that cache norms at ingestion time.
func L2MagnitudePrevalidated(vector []float32) float32 {
	norm := innerProduct(vector, vector)
	if norm < 0 {
		norm = 0
	}
	return float32(math.Sqrt(float64(norm)))
}

// CosineDistanceWithMagnitudesPrevalidated computes cosine distance while
// reusing magnitudes cached by an index. This reduces every candidate score to
// one dot product, matching the normalized-vector hot path used by zvec.
func CosineDistanceWithMagnitudesPrevalidated(left, right []float32, leftMagnitude, rightMagnitude float32) float32 {
	if leftMagnitude == 0 && rightMagnitude == 0 {
		return 0
	}
	if leftMagnitude == 0 || rightMagnitude == 0 {
		return 1
	}
	cosine := innerProduct(left, right) / (leftMagnitude * rightMagnitude)
	cosine = min(1, max(-1, cosine))
	return 1 - cosine
}

// CosineDistances2WithMagnitudesPrevalidated computes cosine distance from one
// query to two candidates while sharing the query load in the SIMD dot-product
// kernel. All vectors and magnitudes must already be validated.
func CosineDistances2WithMagnitudesPrevalidated(
	query, first, second []float32,
	queryMagnitude, firstMagnitude, secondMagnitude float32,
) (firstDistance, secondDistance float32) {
	firstProduct, secondProduct := innerProducts2(query, first, second)
	return cosineDistanceFromProduct(firstProduct, queryMagnitude, firstMagnitude),
		cosineDistanceFromProduct(secondProduct, queryMagnitude, secondMagnitude)
}

// CosineDistances4WithMagnitudesPrevalidated computes cosine distance from one
// query to four candidates while sharing query loads in the SIMD dot-product
// kernel. All vectors and magnitudes must already be validated.
func CosineDistances4WithMagnitudesPrevalidated(
	query, first, second, third, fourth []float32,
	queryMagnitude, firstMagnitude, secondMagnitude, thirdMagnitude, fourthMagnitude float32,
) (firstDistance, secondDistance, thirdDistance, fourthDistance float32) {
	firstProduct, secondProduct, thirdProduct, fourthProduct := innerProducts4(query, first, second, third, fourth)
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

// MIPSL2SquaredPrevalidated computes MIPS-to-L2 distance without validating inputs.
func MIPSL2SquaredPrevalidated(left, right []float32) float32 {
	return mipsL2Squared(left, right)
}
