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

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCosineDistancesWithMagnitudes(t *testing.T) {
	t.Parallel()

	query := []float32{1, 2, 3}
	candidates := [][]float32{
		{4, 5, 6},
		{-1, 0, 1},
		{1, 2, 3},
		{0, 0, 0},
	}
	queryMagnitude := magnitude(query)
	candidateMagnitudes := []float32{
		magnitude(candidates[0]),
		magnitude(candidates[1]),
		magnitude(candidates[2]),
		magnitude(candidates[3]),
	}

	first, second := CosineDistances2WithMagnitudes(
		query, candidates[0], candidates[1],
		queryMagnitude, candidateMagnitudes[0], candidateMagnitudes[1],
	)
	require.InDelta(t, cosineDistance(query, candidates[0]), first, 1e-6)
	require.InDelta(t, cosineDistance(query, candidates[1]), second, 1e-6)

	first, second, third, fourth := CosineDistances4WithMagnitudes(
		query, candidates[0], candidates[1], candidates[2], candidates[3],
		queryMagnitude, candidateMagnitudes[0], candidateMagnitudes[1], candidateMagnitudes[2], candidateMagnitudes[3],
	)
	for index, actual := range []float32{first, second, third, fourth} {
		require.InDelta(t, cosineDistance(query, candidates[index]), actual, 1e-6)
	}
}

func magnitude(vector []float32) float32 {
	return float32(math.Sqrt(float64(innerProductOracle(vector, vector))))
}

func cosineDistance(left, right []float32) float32 {
	leftMagnitude := magnitude(left)
	rightMagnitude := magnitude(right)
	if leftMagnitude == 0 && rightMagnitude == 0 {
		return 0
	}
	if leftMagnitude == 0 || rightMagnitude == 0 {
		return 1
	}
	return 1 - innerProductOracle(left, right)/(leftMagnitude*rightMagnitude)
}

func innerProductOracle(left, right []float32) (sum float32) {
	for index, value := range left {
		sum += value * right[index]
	}
	return
}
