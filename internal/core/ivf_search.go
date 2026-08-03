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
)

const DefaultIVFNProbe = 10

var ErrInvalidIVFNProbe = errors.New("core: IVF NProbe must be positive")

// IVFSearchOptions combines common exact-result controls with the number of
// centroid lists to probe.
type IVFSearchOptions struct {
	SearchOptions
	NProbe int
}

// Validate checks top-k, radius, and probe-count invariants.
func (o IVFSearchOptions) Validate() error {
	if err := o.SearchOptions.Validate(); err != nil {
		return err
	}
	if o.NProbe <= 0 {
		return ErrInvalidIVFNProbe
	}
	return nil
}

// Search uses the baseline default NProbe. A zero top-k returns an empty
// result for consistency with the common DenseSearcher contract.
func (i *IVFIndex) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	return i.searchIVF(ctx, query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: k},
		NProbe:        DefaultIVFNProbe,
	}, false)
}

// SearchWithOptions applies a filter and exact-result radius while using the
// baseline default NProbe.
func (i *IVFIndex) SearchWithOptions(ctx context.Context, query []float32, options SearchOptions) ([]Result, error) {
	return i.searchIVF(ctx, query, IVFSearchOptions{
		SearchOptions: options,
		NProbe:        DefaultIVFNProbe,
	}, true)
}

// SearchIVF probes the metric-best centroids and exact-scores originals in
// only those lists.
func (i *IVFIndex) SearchIVF(ctx context.Context, query []float32, options IVFSearchOptions) ([]Result, error) {
	return i.searchIVF(ctx, query, options, true)
}

func (i *IVFIndex) searchIVF(ctx context.Context, query []float32, options IVFSearchOptions, requirePositiveTopK bool) ([]Result, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if options.NProbe <= 0 {
		return nil, ErrInvalidIVFNProbe
	}
	if requirePositiveTopK {
		if err := options.SearchOptions.Validate(); err != nil {
			return nil, err
		}
	} else {
		if options.TopK < 0 {
			return nil, errors.New("core: negative IVF top-k")
		}
		if options.Radius < 0 {
			return nil, ErrInvalidRadius
		}
	}
	if _, err := i.options.Metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate IVF query: %w", err)
	}
	if options.TopK == 0 || len(i.keys) == 0 {
		return []Result{}, nil
	}

	lists, err := i.ProbedLists(ctx, query, options.NProbe)
	if err != nil {
		return nil, err
	}
	count := 0
	for _, list := range lists {
		count += len(i.lists[list].positions)
	}
	positions := make([]int, 0, count)
	for _, list := range lists {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		positions = append(positions, i.lists[list].positions...)
	}
	return topKCandidatesWithOptions(ctx, i.options.Metric, query, options.SearchOptions, len(positions), func(index int) Candidate {
		position := positions[index]
		start := position * i.dimension
		return Candidate{Key: i.keys[position], Vector: i.vectors[start : start+i.dimension]}
	}, requirePositiveTopK)
}

// ProbedLists returns up to nprobe centroid indexes in metric-best order.
func (i *IVFIndex) ProbedLists(ctx context.Context, query []float32, nprobe int) ([]int, error) {
	if i == nil {
		return nil, errors.New("core: nil IVF index")
	}
	if ctx == nil {
		return nil, errors.New("core: nil IVF probe context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if nprobe <= 0 {
		return nil, ErrInvalidIVFNProbe
	}
	if len(query) != i.dimension {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), i.dimension)
	}
	if _, err := i.options.Metric.Compute(query, query); err != nil {
		return nil, fmt.Errorf("core: validate IVF probe query: %w", err)
	}
	if len(i.lists) == 0 {
		return []int{}, nil
	}
	centroids := i.model.centroids
	candidates := make([]Candidate, len(centroids))
	for index := range centroids {
		candidates[index] = Candidate{Key: uint64(index), Vector: centroids[index]}
	}
	count := min(nprobe, len(candidates))
	results, err := TopK(ctx, i.options.Metric, query, candidates, count)
	if err != nil {
		return nil, fmt.Errorf("core: select IVF centroids: %w", err)
	}
	lists := make([]int, len(results))
	for index, result := range results {
		lists[index] = int(result.Key)
	}
	return lists, nil
}

var _ DenseSearcher = (*IVFIndex)(nil)
var _ DenseQuerySearcher = (*IVFIndex)(nil)
