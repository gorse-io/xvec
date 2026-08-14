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

import stdmath "math"

// NormalizeL2 scales vector to unit L2 norm in place. A zero vector is left
// unchanged.
func NormalizeL2(vector []float32) {
	var normSquared float64
	for _, value := range vector {
		normSquared += float64(value) * float64(value)
	}
	if normSquared == 0 {
		return
	}
	inverseNorm := 1 / stdmath.Sqrt(normSquared)
	for index := range vector {
		vector[index] = float32(float64(vector[index]) * inverseNorm)
	}
}
