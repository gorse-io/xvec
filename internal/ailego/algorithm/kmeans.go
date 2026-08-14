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

package algorithm

import (
	"context"
	"math"
	"slices"
)

// InitializeReservoir selects centroids uniformly without replacement using
// reservoir sampling.
func InitializeReservoir(
	ctx context.Context,
	vectors [][]float32,
	clusters int,
	intn func(int) int,
) ([][]float32, error) {
	indices := make([]int, clusters)
	for index := range indices {
		indices[index] = index
	}
	for index := clusters; index < len(vectors); index++ {
		if index&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		selected := intn(index + 1)
		if selected < clusters {
			indices[selected] = index
		}
	}
	centroids := make([][]float32, clusters)
	for index, input := range indices {
		if index&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		centroids[index] = slices.Clone(vectors[input])
	}
	return centroids, nil
}

// InitializePlusPlus selects centroids using squared-distance weighted
// k-means++ sampling.
func InitializePlusPlus(
	ctx context.Context,
	vectors [][]float32,
	clusters int,
	intn func(int) int,
	randomFloat64 func() float64,
	squaredDistance func([]float32, []float32) (float64, error),
) ([][]float32, error) {
	first := intn(len(vectors))
	selected := make([]bool, len(vectors))
	selected[first] = true
	centroids := [][]float32{slices.Clone(vectors[first])}
	distances := make([]float64, len(vectors))
	for len(centroids) < clusters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var total float64
		for index, vector := range vectors {
			if index&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if selected[index] {
				distances[index] = 0
				continue
			}
			best := math.Inf(1)
			for centroidIndex, centroid := range centroids {
				if centroidIndex&63 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				distance, err := squaredDistance(vector, centroid)
				if err != nil {
					return nil, err
				}
				best = min(best, distance)
			}
			distances[index] = best
			total += best
		}

		chosen := -1
		if total > 0 && !math.IsInf(total, 0) {
			target := randomFloat64() * total
			var cumulative float64
			for index, distance := range distances {
				cumulative += distance
				if !selected[index] && target < cumulative {
					chosen = index
					break
				}
			}
		}
		if chosen < 0 {
			for index := range vectors {
				if !selected[index] {
					chosen = index
					break
				}
			}
		}
		selected[chosen] = true
		centroids = append(centroids, slices.Clone(vectors[chosen]))
	}
	return centroids, nil
}
