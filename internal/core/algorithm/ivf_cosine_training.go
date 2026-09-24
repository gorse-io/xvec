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
	"context"
	"fmt"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
)

// trainCosineIVF follows zvec's cosine IVF pipeline: train L2 centroids on
// [unit vector, original magnitude], then assign originals by dot product with
// the vector part of each centroid. The magnitude coordinate participates in
// training only. Centroids must not be renormalized after training: doing so
// changes the lists selected by zvec's 1-dot cosine routing.
func trainCosineIVF(ctx context.Context, vectors [][]float32, options KMeansOptions) (*KMeansModel, []int, error) {
	dimension := len(vectors[0])
	stride := dimension + 1
	if len(vectors) > maxPlatformInt()/stride {
		return nil, nil, ErrIVFCapacity
	}
	storage := make([]float32, len(vectors)*stride)
	training := make([][]float32, len(vectors))
	for position, vector := range vectors {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, nil, err
			}
		}
		normalized := storage[position*stride : (position+1)*stride]
		magnitude := mathutil.L2Magnitude(vector)
		if !finiteFloat32(magnitude) {
			return nil, nil, fmt.Errorf("core: non-finite cosine IVF magnitude at %d", position)
		}
		copy(normalized, vector)
		mathutil.NormalizeL2(normalized[:dimension])
		normalized[dimension] = magnitude
		training[position] = normalized
	}
	options.Metric = MetricL2
	options.Initializer = KMeansInitKMC2
	options.EmptyPolicy = KMeansEmptyKeep
	model, _, err := trainKMeansWithAssignments(ctx, training, options)
	if err != nil {
		return nil, nil, err
	}
	for position := range training {
		training[position] = training[position][:dimension]
	}
	for position := range model.centroids {
		model.centroids[position] = model.centroids[position][:dimension]
	}
	labels, scores, err := assignKMeans(ctx, MetricIP, training, model.centroids, options.Workers)
	if err != nil {
		return nil, nil, err
	}
	model.metric = MetricCosine
	model.dimension = dimension
	clear(model.counts)
	model.cost = 0
	for position, label := range labels {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, nil, err
			}
		}
		model.counts[label]++
		model.cost += float64(1 - min(float32(1), max(float32(-1), scores[position])))
	}
	return model, labels, nil
}
