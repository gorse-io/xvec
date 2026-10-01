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
	"errors"
	"slices"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/gorse-io/xvec/internal/ailego/math"
)

// SearchDenseReader scans encoded or borrowed originals without materializing
// an index. Only one decoded vector and O(top-k) result storage are allocated.
func SearchDenseReader(ctx context.Context, metric Metric, query []float32, keys []uint64, reader DenseVectorReader, options SearchOptions) ([]Result, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}
	heap := container.NewHeapWithCapacity(min(options.TopK, len(keys)), func(left, right Result) bool { return resultBetter(metric, right, left) })
	err := visitDenseReader(ctx, metric, query, keys, reader, options.Radius, options.Filter, func(result Result) {
		if heap.Len() < options.TopK {
			heap.Push(result)
			return
		}
		worst, _ := heap.Peek()
		if resultBetter(metric, result, worst) {
			heap.Replace(result)
		}
	})
	if err != nil {
		return nil, err
	}
	results := heap.Values()
	slices.SortFunc(results, func(left, right Result) int {
		if resultBetter(metric, left, right) {
			return -1
		}
		if resultBetter(metric, right, left) {
			return 1
		}
		return 0
	})
	return results, nil
}

// SearchDenseReaderGroups applies complete grouping to an original-vector scan.
func SearchDenseReaderGroups(ctx context.Context, metric Metric, query []float32, keys []uint64, reader DenseVectorReader, options GroupByOptions) ([]GroupResult, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}
	accumulator := newGroupAccumulator(metric, options.TopKPerGroup)
	err := visitDenseReader(ctx, metric, query, keys, reader, options.Radius, options.Filter, func(result Result) {
		if value, found := options.Resolve(result.Key); found {
			accumulator.add(value, result)
		}
	})
	if err != nil {
		return nil, err
	}
	return accumulator.finish(options.GroupCount), nil
}

func visitDenseReader(ctx context.Context, metric Metric, query []float32, keys []uint64, reader DenseVectorReader, radius float32, filter CandidateFilter, visit func(Result)) error {
	if ctx == nil || reader == nil {
		return errors.New("core: nil dense reader search context or reader")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := mathutil.ValidateDense(query, len(query)); err != nil {
		return err
	}
	distance, err := metric.Distance()
	if err != nil {
		return err
	}
	vector := make([]float32, len(query))
	for position, key := range keys {
		if err := ctx.Err(); err != nil {
			return err
		}
		if filter != nil && !filter(key) {
			continue
		}
		if err := reader.ReadVector(position, vector); err != nil {
			return err
		}
		score := distance(vector, query)
		if scoreWithinRadius(metric, score, radius) {
			visit(Result{Key: key, Score: score})
		}
	}
	return nil
}
