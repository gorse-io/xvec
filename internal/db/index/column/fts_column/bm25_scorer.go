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

package ftscolumn

import (
	"errors"
	"fmt"
	"math"
)

// ErrInvalidBM25 identifies invalid parameters or corpus statistics.
var ErrInvalidBM25 = errors.New("core: invalid BM25 configuration")

// BM25Params configures term-frequency saturation and document-length
// normalization. K1 must be non-negative and B must be in [0, 1].
type BM25Params struct {
	K1 float32
	B  float32
}

// DefaultBM25Params returns the pinned baseline defaults.
func DefaultBM25Params() BM25Params {
	return BM25Params{K1: 1.2, B: 0.75}
}

// BM25Scorer is an immutable, concurrency-safe scorer over one exact corpus
// statistics snapshot.
type BM25Scorer struct {
	params BM25Params
	stats  FTSCorpusStats
}

// NewBM25Scorer validates and snapshots params and deletion-aware corpus
// statistics. Call AggregateFTSCorpusStats to build stats from segment views.
func NewBM25Scorer(params BM25Params, stats FTSCorpusStats) (*BM25Scorer, error) {
	if math.IsNaN(float64(params.K1)) || math.IsInf(float64(params.K1), 0) || params.K1 < 0 {
		return nil, fmt.Errorf("%w: K1 must be finite and non-negative", ErrInvalidBM25)
	}
	if math.IsNaN(float64(params.B)) || math.IsInf(float64(params.B), 0) || params.B < 0 || params.B > 1 {
		return nil, fmt.Errorf("%w: B must be finite and in [0, 1]", ErrInvalidBM25)
	}
	ownedStats := FTSCorpusStats{
		TotalDocuments:      stats.TotalDocuments,
		TotalTokens:         stats.TotalTokens,
		documentFrequencies: make(map[string]uint64, len(stats.documentFrequencies)),
	}
	if stats.TotalDocuments == 0 && stats.TotalTokens != 0 {
		return nil, fmt.Errorf("%w: a document-empty corpus cannot contain tokens", ErrInvalidBM25)
	}
	if stats.TotalTokens == 0 && len(stats.documentFrequencies) != 0 {
		return nil, fmt.Errorf("%w: a token-empty corpus cannot contain terms", ErrInvalidBM25)
	}
	for term, frequency := range stats.documentFrequencies {
		if frequency == 0 || frequency > stats.TotalDocuments || frequency > stats.TotalTokens {
			return nil, fmt.Errorf("%w: term %q frequency %d exceeds corpus totals", ErrInvalidBM25, term, frequency)
		}
		ownedStats.documentFrequencies[term] = frequency
	}
	return &BM25Scorer{params: params, stats: ownedStats}, nil
}

// Params returns the immutable parameter snapshot.
func (s *BM25Scorer) Params() BM25Params {
	if s == nil {
		return BM25Params{}
	}
	return s.params
}

// Stats returns an owned copy of the scorer's corpus snapshot.
func (s *BM25Scorer) Stats() FTSCorpusStats {
	if s == nil {
		return FTSCorpusStats{}
	}
	return FTSCorpusStats{
		TotalDocuments:      s.stats.TotalDocuments,
		TotalTokens:         s.stats.TotalTokens,
		documentFrequencies: s.stats.DocumentFrequencies(),
	}
}

// DocumentFrequency returns the live corpus document frequency of term.
func (s *BM25Scorer) DocumentFrequency(term string) uint64 {
	if s == nil {
		return 0
	}
	return s.stats.DocumentFrequency(term)
}

// IDF calculates the baseline Robertson-Sparck Jones inverse document
// frequency with 0.5 smoothing.
func (s *BM25Scorer) IDF(documentFrequency uint64) float32 {
	if s == nil || s.stats.TotalDocuments == 0 {
		return 0
	}
	totalDocuments := float32(s.stats.TotalDocuments)
	documentFrequency32 := float32(documentFrequency)
	ratio := (totalDocuments-documentFrequency32+0.5)/(documentFrequency32+0.5) + 1
	return float32(math.Log(float64(ratio)))
}

// TermIDF calculates IDF from the scorer's snapshotted term statistics.
func (s *BM25Scorer) TermIDF(term string) float32 {
	return s.IDF(s.DocumentFrequency(term))
}

// MaxScoreBound returns the tf-to-infinity upper bound for a term.
func (s *BM25Scorer) MaxScoreBound(documentFrequency uint64) float32 {
	idf := s.IDF(documentFrequency)
	if s == nil || idf <= 0 {
		return 0
	}
	return idf * (s.params.K1 + 1)
}

// Score calculates one term's BM25 contribution.
func (s *BM25Scorer) Score(documentFrequency uint64, termFrequency, documentLength uint32) float32 {
	return s.ScoreWithIDF(s.IDF(documentFrequency), termFrequency, documentLength)
}

// ScoreWithIDF calculates one term's BM25 contribution using a precomputed
// IDF value.
func (s *BM25Scorer) ScoreWithIDF(idf float32, termFrequency, documentLength uint32) float32 {
	return s.ScoreWithIDFAndBoost(idf, termFrequency, documentLength, 1)
}

// ScoreWithIDFAndBoost applies a linear query-term or phrase boost.
func (s *BM25Scorer) ScoreWithIDFAndBoost(idf float32, termFrequency, documentLength uint32, boost float32) float32 {
	if s == nil || s.stats.TotalDocuments == 0 || s.stats.TotalTokens == 0 || idf <= 0 || termFrequency == 0 || boost == 0 {
		return 0
	}
	return boost * idf * s.termNormalization(termFrequency, documentLength)
}

func (s *BM25Scorer) termNormalization(termFrequency, documentLength uint32) float32 {
	if s == nil || s.stats.TotalDocuments == 0 || s.stats.TotalTokens == 0 || termFrequency == 0 {
		return 0
	}
	termFrequency32 := float32(termFrequency)
	documentLength32 := float32(documentLength)
	averageDocumentLength := float32(s.stats.TotalTokens) / float32(s.stats.TotalDocuments)
	denominator := termFrequency32 + s.params.K1*(1-s.params.B+s.params.B*documentLength32/averageDocumentLength)
	if denominator == 0 {
		return 0
	}
	return termFrequency32 * (s.params.K1 + 1) / denominator
}
