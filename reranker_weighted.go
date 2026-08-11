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

package xvec

import (
	"context"
	"math"

	"github.com/gorse-io/xvec/internal/core"
)

// WeightedReranker normalizes each branch score according to its field metric,
// multiplies it by the corresponding weight, and sums by primary key. Weights
// may be negative but must be finite, and their count must match the batches.
type WeightedReranker struct {
	Weights []float64
}

// NewWeightedReranker returns a reranker with an owned weight snapshot.
func NewWeightedReranker(weights ...float64) WeightedReranker {
	return WeightedReranker{Weights: append([]float64(nil), weights...)}
}

// Validate checks that every configured weight is finite.
func (r WeightedReranker) Validate() error {
	for index, weight := range r.Weights {
		if math.IsNaN(weight) || math.IsInf(weight, 0) {
			return invalidArgument("validate weighted reranker", "Weights[%d] must be finite", index)
		}
	}
	return nil
}

// Rerank applies baseline metric normalization and deterministic weighted
// score fusion. The first occurrence supplies the returned document payload.
func (r WeightedReranker) Rerank(ctx context.Context, batches []RerankBatch, topK int) ([]Document, error) {
	const op = "weighted rerank"
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	if len(r.Weights) != len(batches) {
		return nil, invalidArgument(op, "weight count %d does not match batch count %d", len(r.Weights), len(batches))
	}
	weights := append([]float64(nil), r.Weights...)
	return scoreBasedRerank(ctx, op, batches, topK, func(document Document, _ int, batch int) (float64, error) {
		normalized, err := normalizeWeightedScore(document.Score, batches[batch].Field)
		if err != nil {
			return 0, err
		}
		return normalized * weights[batch], nil
	})
}

func normalizeWeightedScore(score float32, field FieldSchema) (float64, error) {
	value := float64(score)
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, invalidArgument("weighted rerank", "field %q has a non-finite document score", field.Name)
	}
	if field.IndexType() == IndexTypeFTS {
		return 2 * math.Atan(value) / math.Pi, nil
	}
	if !field.DataType.IsVector() {
		return 0, invalidArgument("weighted rerank", "field %q is neither vector nor FTS", field.Name)
	}
	index, err := resolveCollectionVectorIndex(field, "weighted rerank", "")
	if err != nil {
		return 0, err
	}
	switch index.metric {
	case core.MetricL2:
		return 1 - 2*math.Atan(value)/math.Pi, nil
	case core.MetricIP:
		return 0.5 + math.Atan(value)/math.Pi, nil
	case core.MetricCosine:
		return 1 - value/2, nil
	default:
		return 0, invalidArgument("weighted rerank", "field %q uses unsupported metric %d", field.Name, index.metric)
	}
}

var _ Reranker = WeightedReranker{}
