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

package core

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDenseDistanceBatches(t *testing.T) {
	query := []float32{1, 2, 3}
	candidates := [][]float32{{4, 5, 6}, {-1, 0, 1}, {1, 2, 3}, {0, 0, 0}}
	for _, metric := range []Metric{MetricL2, MetricIP} {
		distance, err := metric.Distance()
		require.NoError(t, err)

		first, second := denseDistances2(metric, query, candidates[0], candidates[1])
		require.Equal(t, distance(query, candidates[0]), first)
		require.Equal(t, distance(query, candidates[1]), second)

		first, second, third, fourth := denseDistances4(
			metric, query, candidates[0], candidates[1], candidates[2], candidates[3],
		)
		for index, actual := range []float32{first, second, third, fourth} {
			require.Equal(t, distance(query, candidates[index]), actual)
		}
	}
}
