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

package zvec

import (
	"context"
	"math"
	"sort"
)

// DefaultRRFRankConstant is the pinned reciprocal-rank-fusion constant.
const DefaultRRFRankConstant = 60

// RRFReranker combines ranks without inspecting source scores. A document at
// zero-based rank r contributes 1/(RankConstant+r+1) in each batch where it
// occurs. Documents are fused by primary key, matching the pinned baseline.
type RRFReranker struct {
	RankConstant int
}

// NewRRFReranker returns the pinned rank-constant default.
func NewRRFReranker() RRFReranker {
	return RRFReranker{RankConstant: DefaultRRFRankConstant}
}

// Validate rejects a negative rank constant. Zero is valid and intentionally
// differs from the default; use NewRRFReranker for baseline defaults.
func (r RRFReranker) Validate() error {
	if r.RankConstant < 0 {
		return invalidArgument("validate RRF reranker", "RankConstant must be non-negative")
	}
	return nil
}

// Rerank applies reciprocal rank fusion, returning at most topK distinct
// documents by descending fused score. Equal scores are ordered by primary
// key and then DocID so results remain deterministic across processes.
func (r RRFReranker) Rerank(ctx context.Context, batches []RerankBatch, topK int) ([]Document, error) {
	const op = "RRF rerank"
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return scoreBasedRerank(ctx, op, batches, topK, func(_ Document, rank, _ int) (float64, error) {
		return 1 / (float64(r.RankConstant) + float64(rank) + 1), nil
	})
}

type rerankScoreFunc func(document Document, rank, batch int) (float64, error)

func scoreBasedRerank(ctx context.Context, op string, batches []RerankBatch, topK int, score rerankScoreFunc) ([]Document, error) {
	if topK <= 0 || len(batches) == 0 {
		return []Document{}, nil
	}
	type fusedCandidate struct {
		document Document
		score    float64
	}
	candidates := make(map[string]*fusedCandidate)
	work := 0
	for batchIndex, batch := range batches {
		for rank, document := range batch.Documents {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			work++
			contribution, err := score(document, rank, batchIndex)
			if err != nil {
				return nil, err
			}
			if math.IsNaN(contribution) || math.IsInf(contribution, 0) {
				return nil, invalidArgument(op, "document %q produced a non-finite contribution", document.PrimaryKey)
			}
			candidate := candidates[document.PrimaryKey]
			if candidate == nil {
				candidate = &fusedCandidate{document: document}
				candidates[document.PrimaryKey] = candidate
			}
			candidate.score += contribution
			if math.IsNaN(candidate.score) || math.IsInf(candidate.score, 0) {
				return nil, invalidArgument(op, "document %q produced a non-finite fused score", document.PrimaryKey)
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ordered := make([]*fusedCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		ordered = append(ordered, candidate)
	}
	sort.Slice(ordered, func(left, right int) bool {
		if ordered[left].score != ordered[right].score {
			return ordered[left].score > ordered[right].score
		}
		if ordered[left].document.PrimaryKey != ordered[right].document.PrimaryKey {
			return ordered[left].document.PrimaryKey < ordered[right].document.PrimaryKey
		}
		return ordered[left].document.DocID < ordered[right].document.DocID
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if topK < len(ordered) {
		ordered = ordered[:topK]
	}
	result := make([]Document, len(ordered))
	for index, candidate := range ordered {
		candidate.document.Score = float32(candidate.score)
		if math.IsInf(float64(candidate.document.Score), 0) {
			return nil, invalidArgument(op, "document %q fused score exceeds float32", candidate.document.PrimaryKey)
		}
		result[index] = candidate.document
	}
	return result, nil
}

var _ Reranker = RRFReranker{}
