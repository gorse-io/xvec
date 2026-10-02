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
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/math_batch"
)

// computeBuildDistances gathers immutable FP32 rows and uses the existing
// four-candidate kernels with cached cosine magnitudes. Native FP16 builders
// share query conversions across four candidates; quantized builders supply their own
// batch callback via fp16BuildScorers.
func (i *HNSWIndex) computeBuildDistances(query int, positions []int, scratch *hnswVisited) error {
	count := len(positions)
	scratch.batchScores = slices.Grow(scratch.batchScores[:0], count)[:count]
	if i.fp16 {
		var batch func(query, first, second, third, fourth []uint16, output []float32)
		switch i.options.Metric {
		case MetricL2:
			batch = mathbatch.SquaredEuclideanDistances4FP16
		case MetricIP:
			batch = mathbatch.InnerProducts4FP16
		case MetricMIPSL2:
			batch = mathbatch.MIPSL2SquaredDistances4FP16
		}
		// Cosine retains cached magnitudes and its single-pair reduction order.
		j := 0
		if batch != nil {
			q := i.vectorFP16At(query)
			for ; j+4 <= count; j += 4 {
				batch(q, i.vectorFP16At(positions[j]), i.vectorFP16At(positions[j+1]),
					i.vectorFP16At(positions[j+2]), i.vectorFP16At(positions[j+3]), scratch.batchScores[j:])
			}
		}
		for ; j < count; j++ {
			position := positions[j]
			score, err := i.computeDistanceAt(query, position)
			if err != nil {
				return err
			}
			scratch.batchScores[j] = score
		}
		return nil
	}
	scratch.batchVectors = slices.Grow(scratch.batchVectors[:0], count)[:count]
	cached := i.options.Metric == MetricCosine && len(i.vectorMagnitudes) == len(i.keys)
	scratch.batchMagnitudes = scratch.batchMagnitudes[:0]
	if cached {
		scratch.batchMagnitudes = slices.Grow(scratch.batchMagnitudes, count)[:count]
	}
	for j, position := range positions {
		scratch.batchVectors[j] = i.vectorAt(position)
		if cached {
			scratch.batchMagnitudes[j] = i.vectorMagnitudes[position]
		}
	}
	prefetchDenseHNSWRows(scratch.batchVectors, 8, 1)
	return denseDistances(i.options.Metric, i.vectorAt(query), scratch.batchVectors, i.magnitudeAt(query), scratch.batchMagnitudes, scratch.batchScores)
}
