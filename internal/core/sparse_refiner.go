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

	"github.com/gorse-io/zvec/internal/ailego"
)

// SparseRefiner re-scores approximate sparse candidates in an exact
// representation. Sparse collection indexes use inner product exclusively.
type SparseRefiner interface {
	Metric() Metric
	RefineSparse(ctx context.Context, query SparseVector, candidates []Result, options SearchOptions) ([]Result, error)
}

// OriginalSparseVectorRefiner computes final inner-product scores from a
// provider that retains the unquantized sparse vectors.
type OriginalSparseVectorRefiner struct {
	provider SparseProvider
}

// NewOriginalSparseVectorRefiner constructs an exact sparse candidate
// refiner.
func NewOriginalSparseVectorRefiner(provider SparseProvider) (*OriginalSparseVectorRefiner, error) {
	if provider == nil {
		return nil, errors.New("core: nil original sparse-vector provider")
	}
	return &OriginalSparseVectorRefiner{provider: provider}, nil
}

func (r *OriginalSparseVectorRefiner) Metric() Metric {
	if r == nil {
		return 0
	}
	return MetricIP
}

// RefineSparse ignores approximate scores, resolves each unique candidate key
// to its original sparse vector, and returns deterministic exact top-k results.
func (r *OriginalSparseVectorRefiner) RefineSparse(
	ctx context.Context,
	query SparseVector,
	candidates []Result,
	options SearchOptions,
) ([]Result, error) {
	if r == nil || r.provider == nil {
		return nil, errors.New("core: nil original sparse-vector refiner")
	}
	if ctx == nil {
		return nil, errors.New("core: nil sparse refine context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if _, err := ailego.SparseInnerProduct(query.Indices, query.Values, nil, nil); err != nil {
		return nil, fmt.Errorf("core: validate sparse refine query: %w", err)
	}

	seen := make(map[uint64]struct{}, len(candidates))
	exact := make([]Result, 0, len(candidates))
	for _, candidate := range candidates {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, duplicate := seen[candidate.Key]; duplicate {
			continue
		}
		seen[candidate.Key] = struct{}{}
		if options.Filter != nil && !options.Filter(candidate.Key) {
			continue
		}
		vector, found := r.provider.SparseVector(candidate.Key)
		if !found {
			return nil, fmt.Errorf("%w: key %d", ErrMissingRefineVector, candidate.Key)
		}
		score, err := ailego.SparseInnerProduct(query.Indices, query.Values, vector.Indices, vector.Values)
		if err != nil {
			return nil, fmt.Errorf("core: score original sparse vector %d: %w", candidate.Key, err)
		}
		if scoreWithinRadius(MetricIP, score, options.Radius) {
			exact = append(exact, Result{Key: candidate.Key, Score: score})
		}
	}
	return MergeSearchResults(MetricIP, options.TopK, exact), nil
}

// RefinedSparseSearch expands the base candidate count, disables approximate
// radius pruning, and applies exact filter/radius/top-k in the sparse refiner.
func RefinedSparseSearch(
	ctx context.Context,
	base SparseQuerySearcher,
	refiner SparseRefiner,
	query SparseVector,
	options SearchOptions,
	scaleFactor float32,
) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil refined sparse search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if base == nil || refiner == nil {
		return nil, errors.New("core: nil sparse base searcher or refiner")
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if base.Metric() != refiner.Metric() {
		return nil, fmt.Errorf("core: sparse base metric %d does not match refiner metric %d", base.Metric(), refiner.Metric())
	}
	candidateCount, err := RefinementCandidateCount(options.TopK, scaleFactor)
	if err != nil {
		return nil, err
	}
	baseOptions := SearchOptions{TopK: candidateCount, Filter: options.Filter}
	candidates, err := base.SearchSparseWithOptions(ctx, query, baseOptions)
	if err != nil {
		return nil, fmt.Errorf("core: sparse base candidate search: %w", err)
	}
	return refiner.RefineSparse(ctx, query, candidates, options)
}

var _ SparseRefiner = (*OriginalSparseVectorRefiner)(nil)
