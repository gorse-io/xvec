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
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/math_batch"
)

const (
	initialDistanceBatchCapacity   = 256
	maxPooledDistanceBatchCapacity = 4096
)

var denseDistanceBatchPool = sync.Pool{
	New: func() any { return new(denseDistanceBatch) },
}

type denseDistanceBatch struct {
	positions  []int
	ids        []uint32
	ties       []uint64
	overflow   []uint32
	vectors    [][]float32
	magnitudes []float32
	scores     []float32
	blockHeap  BlockHeap
}

func acquireDenseDistanceBatch(capacity int) *denseDistanceBatch {
	batch := denseDistanceBatchPool.Get().(*denseDistanceBatch)
	capacity = min(capacity, initialDistanceBatchCapacity)
	if cap(batch.positions) < capacity {
		batch.positions = make([]int, 0, capacity)
	}
	if cap(batch.ids) < capacity {
		batch.ids = make([]uint32, 0, capacity)
	}
	if cap(batch.ties) < capacity {
		batch.ties = make([]uint64, 0, capacity)
	}
	if cap(batch.vectors) < capacity {
		batch.vectors = make([][]float32, 0, capacity)
	}
	if cap(batch.magnitudes) < capacity {
		batch.magnitudes = make([]float32, 0, capacity)
	}
	if cap(batch.scores) < capacity {
		batch.scores = make([]float32, 0, capacity)
	}
	return batch
}

func releaseDenseDistanceBatch(batch *denseDistanceBatch) {
	clear(batch.vectors[:cap(batch.vectors)])
	batch.positions = batch.positions[:0]
	batch.ids = batch.ids[:0]
	batch.ties = batch.ties[:0]
	batch.overflow = batch.overflow[:0]
	batch.vectors = batch.vectors[:0]
	batch.magnitudes = batch.magnitudes[:0]
	batch.scores = batch.scores[:0]
	batch.blockHeap.release(maxPooledDistanceBatchCapacity)
	if cap(batch.positions) > maxPooledDistanceBatchCapacity || cap(batch.ids) > maxPooledDistanceBatchCapacity || cap(batch.ties) > maxPooledDistanceBatchCapacity || cap(batch.overflow) > maxPooledDistanceBatchCapacity || cap(batch.vectors) > maxPooledDistanceBatchCapacity || cap(batch.magnitudes) > maxPooledDistanceBatchCapacity || cap(batch.scores) > maxPooledDistanceBatchCapacity {
		batch.positions = nil
		batch.ids = nil
		batch.ties = nil
		batch.overflow = nil
		batch.vectors = nil
		batch.magnitudes = nil
		batch.scores = nil
	}
	denseDistanceBatchPool.Put(batch)
}

func denseDistances2(metric Metric, query, first, second []float32) (float32, float32) {
	if metric == MetricIP {
		return mathbatch.InnerProducts2(query, first, second)
	}
	return mathbatch.SquaredEuclideanDistances2(query, first, second)
}

func denseDistances4(metric Metric, query, first, second, third, fourth []float32) (float32, float32, float32, float32) {
	if metric == MetricIP {
		return mathbatch.InnerProducts4(query, first, second, third, fourth)
	}
	return mathbatch.SquaredEuclideanDistances4(query, first, second, third, fourth)
}

func denseDistances(
	metric Metric,
	query []float32,
	candidates [][]float32,
	queryMagnitude float32,
	candidateMagnitudes []float32,
	output []float32,
) error {
	switch metric {
	case MetricL2:
		mathbatch.SquaredEuclideanDistances(query, candidates, output)
	case MetricIP:
		mathbatch.InnerProducts(query, candidates, output)
	case MetricCosine:
		if len(candidateMagnitudes) == len(candidates) {
			mathbatch.CosineDistancesWithMagnitudes(query, candidates, queryMagnitude, candidateMagnitudes, output)
			break
		}
		distance, err := metric.Distance()
		if err != nil {
			return err
		}
		for index := range candidates {
			output[index] = distance(query, candidates[index])
		}
	default:
		distance, err := metric.Distance()
		if err != nil {
			return err
		}
		for index := range candidates {
			output[index] = distance(query, candidates[index])
		}
	}
	return nil
}
