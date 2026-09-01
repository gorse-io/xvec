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
	"fmt"
	"math"

	"github.com/gorse-io/xvec/internal/ailego/math"
)

var (
	ErrInvalidRefinerScale = errors.New("core: refiner scale factor must be finite and positive")
	ErrMissingRefineVector = errors.New("core: original vector is missing for refinement")
)

// DenseRefiner re-scores approximate candidates in an exact representation.
type DenseRefiner interface {
	Metric() Metric
	Refine(ctx context.Context, query []float32, candidates []Result, options SearchOptions) ([]Result, error)
}

// OriginalVectorRefiner computes final scores from a provider that retains
// original FP32 vectors.
type OriginalVectorRefiner struct {
	provider DenseProvider
	metric   Metric
}

// NewOriginalVectorRefiner constructs an exact candidate refiner.
func NewOriginalVectorRefiner(provider DenseProvider, metric Metric) (*OriginalVectorRefiner, error) {
	if provider == nil {
		return nil, errors.New("core: nil original-vector provider")
	}
	if provider.Dimension() <= 0 {
		return nil, ErrInvalidDimension
	}
	if !metric.Valid() {
		return nil, errors.New("core: invalid metric")
	}
	return &OriginalVectorRefiner{provider: provider, metric: metric}, nil
}

// Metric returns the exact scoring metric.
func (r *OriginalVectorRefiner) Metric() Metric {
	if r == nil {
		return 0
	}
	return r.metric
}

// Refine ignores approximate scores, resolves each unique candidate key to
// its original vector, and returns exact deterministic top-k results.
func (r *OriginalVectorRefiner) Refine(ctx context.Context, query []float32, candidates []Result, options SearchOptions) ([]Result, error) {
	if r == nil || r.provider == nil {
		return nil, errors.New("core: nil original-vector refiner")
	}
	if ctx == nil {
		return nil, errors.New("core: nil refine context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if len(query) != r.provider.Dimension() {
		return nil, fmt.Errorf("%w: query has %d, want %d", ErrInvalidDimension, len(query), r.provider.Dimension())
	}
	if err := mathutil.ValidateDense(query, r.provider.Dimension()); err != nil {
		return nil, fmt.Errorf("core: validate refine query: %w", err)
	}

	seen := make(map[uint64]struct{}, len(candidates))
	exact := make([]Candidate, 0, len(candidates))
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
		vector, found := r.provider.Vector(candidate.Key)
		if !found {
			return nil, fmt.Errorf("%w: key %d", ErrMissingRefineVector, candidate.Key)
		}
		exact = append(exact, Candidate{Key: candidate.Key, Vector: vector})
	}
	return topKCandidatesWithOptions(ctx, r.metric, query, options, len(exact), func(index int) Candidate {
		return exact[index]
	}, true)
}

// RefinementCandidateCount applies the baseline floor(top-k*scale-factor)
// rule, with a minimum of one candidate and overflow validation.
func RefinementCandidateCount(topK int, scaleFactor float32) (int, error) {
	if topK <= 0 {
		return 0, ErrInvalidTopK
	}
	if scaleFactor <= 0 || math.IsNaN(float64(scaleFactor)) || math.IsInf(float64(scaleFactor), 0) {
		return 0, ErrInvalidRefinerScale
	}
	product := float64(topK) * float64(scaleFactor)
	maxInt := int(^uint(0) >> 1)
	if product >= float64(maxInt) {
		return 0, ErrInvalidRefinerScale
	}
	count := int(math.Floor(product))
	if count < 1 {
		count = 1
	}
	return count, nil
}

// RefinedSearch expands the base candidate count, disables approximate-score
// radius pruning, and applies the exact filter/radius/top-k in the refiner.
func RefinedSearch(
	ctx context.Context,
	base DenseQuerySearcher,
	refiner DenseRefiner,
	query []float32,
	options SearchOptions,
	scaleFactor float32,
) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("core: nil refined search context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if base == nil || refiner == nil {
		return nil, errors.New("core: nil base searcher or refiner")
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	if base.Metric() != refiner.Metric() {
		return nil, fmt.Errorf("core: base metric %d does not match refiner metric %d", base.Metric(), refiner.Metric())
	}
	candidateCount, err := RefinementCandidateCount(options.TopK, scaleFactor)
	if err != nil {
		return nil, err
	}
	baseOptions := SearchOptions{TopK: candidateCount, Filter: options.Filter}
	candidates, err := base.SearchWithOptions(ctx, query, baseOptions)
	if err != nil {
		return nil, fmt.Errorf("core: base candidate search: %w", err)
	}
	return refiner.Refine(ctx, query, candidates, options)
}

var _ DenseRefiner = (*OriginalVectorRefiner)(nil)
