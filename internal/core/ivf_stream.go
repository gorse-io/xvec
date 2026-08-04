// Copyright 2026-present the zvec-go project
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
	"errors"
	"fmt"
	"math"
	"slices"
)

var ErrIVFCapacity = errors.New("core: IVF index capacity exceeded")

// Add incrementally inserts one unique key and finite original vector. While
// the index contains fewer vectors than configured lists, each new vector
// extends the centroid set and starts its own list. Once NList is reached, the
// trained centroids remain fixed and additions enter their metric-best list.
func (i *IVFIndex) Add(ctx context.Context, key uint64, vector []float32) error {
	if i == nil {
		return errors.New("core: nil IVF index")
	}
	if ctx == nil {
		return errors.New("core: nil IVF add context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTrainingVector(vector, i.dimension); err != nil {
		return fmt.Errorf("core: validate incremental IVF vector: %w", err)
	}

	i.mu.Lock()
	defer i.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, exists := i.positions[key]; exists {
		return fmt.Errorf("%w: %d", ErrDuplicateKey, key)
	}
	if len(i.keys) == maxPlatformInt() || len(i.vectors) > maxPlatformInt()-i.dimension {
		return ErrIVFCapacity
	}

	list := 0
	var score float32
	var err error
	growCentroids := len(i.lists) < i.options.NList
	if growCentroids {
		list = len(i.lists)
		score, err = i.options.Metric.Compute(vector, vector)
	} else {
		if i.model == nil {
			return fmt.Errorf("%w: missing trained centroids", ErrInvalidIVFFile)
		}
		list, score, err = i.model.Nearest(vector)
	}
	if err != nil {
		return fmt.Errorf("core: assign incremental IVF vector: %w", err)
	}
	delta := float64(score)
	if i.options.Metric == MetricIP {
		delta = -delta
	}
	cost := delta
	if i.model != nil {
		cost += i.model.cost
	}
	if math.IsNaN(cost) || math.IsInf(cost, 0) {
		return fmt.Errorf("core: assign incremental IVF vector: non-finite objective")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	position := len(i.keys)
	i.positions[key] = position
	i.keys = append(i.keys, key)
	i.vectors = append(i.vectors, vector...)
	i.listForPosition = append(i.listForPosition, list)
	if i.model == nil {
		i.model = &KMeansModel{
			metric:     i.options.Metric,
			dimension:  i.dimension,
			centroids:  [][]float32{slices.Clone(vector)},
			counts:     []int{1},
			cost:       cost,
			iterations: 0,
			converged:  false,
		}
		i.lists = append(i.lists, ivfList{positions: []int{position}})
		return nil
	}
	if growCentroids {
		i.model.centroids = append(i.model.centroids, slices.Clone(vector))
		i.model.counts = append(i.model.counts, 1)
		i.lists = append(i.lists, ivfList{positions: []int{position}})
	} else {
		i.model.counts[list]++
		i.lists[list].positions = append(i.lists[list].positions, position)
	}
	i.model.cost = cost
	return nil
}

var (
	_ DenseStreamer = (*IVFIndex)(nil)
	_ DenseIndex    = (*IVFIndex)(nil)
)
