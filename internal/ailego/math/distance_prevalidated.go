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

import "github.com/gorse-io/xvec/internal/floats"

// DenseDistance computes a score for two already validated dense vectors.
// Callers must guarantee equal, non-zero dimensions and finite components.
type DenseDistance func(left, right []float32) (float32, error)

// L2SquaredPrevalidated computes squared Euclidean distance without validating
// its inputs. It is intended for index hot paths whose storage boundary has
// already validated every vector.
func L2SquaredPrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(floats.L2Squared(left, right)))
}

// InnerProductPrevalidated computes inner product without validating inputs.
func InnerProductPrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(floats.InnerProduct(left, right)))
}

// CosineDistancePrevalidated computes cosine distance without validating inputs.
func CosineDistancePrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(cosineDistance(left, right)))
}

// MIPSL2SquaredPrevalidated computes MIPS-to-L2 distance without validating inputs.
func MIPSL2SquaredPrevalidated(left, right []float32) (float32, error) {
	return finiteScore(float64(mipsL2Squared(left, right)))
}
