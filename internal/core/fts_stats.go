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
	"sort"

	"github.com/gorse-io/zvec/internal/ailego"
)

// ErrInvalidFTSStats identifies an invalid segment view or overflow while
// aggregating corpus statistics.
var ErrInvalidFTSStats = errors.New("core: invalid FTS statistics")

// FTSSegmentView contributes one immutable dictionary and an optional bitmap
// of deleted segment-local document IDs to corpus statistics.
type FTSSegmentView struct {
	Dictionary       *FTSTermDictionary
	DeletedDocuments *ailego.Bitmap
}

// FTSCorpusStats is a deletion-aware summary across multiple FTS segments.
// Term document frequencies are exposed through copy-returning accessors.
type FTSCorpusStats struct {
	TotalDocuments      uint64
	TotalTokens         uint64
	documentFrequencies map[string]uint64
}

// AverageDocumentLength returns total tokens divided by total live documents,
// or one for an empty corpus.
func (s FTSCorpusStats) AverageDocumentLength() float64 {
	if s.TotalDocuments == 0 {
		return 1
	}
	return float64(s.TotalTokens) / float64(s.TotalDocuments)
}

// DocumentFrequency returns the number of live documents containing term.
func (s FTSCorpusStats) DocumentFrequency(term string) uint64 {
	return s.documentFrequencies[term]
}

// Terms returns every live term in byte-lexical order.
func (s FTSCorpusStats) Terms() []string {
	terms := make([]string, 0, len(s.documentFrequencies))
	for term := range s.documentFrequencies {
		terms = append(terms, term)
	}
	sort.Strings(terms)
	return terms
}

// DocumentFrequencies returns an independent copy keyed by term.
func (s FTSCorpusStats) DocumentFrequencies() map[string]uint64 {
	result := make(map[string]uint64, len(s.documentFrequencies))
	for term, frequency := range s.documentFrequencies {
		result[term] = frequency
	}
	return result
}

// AggregateFTSCorpusStats calculates exact live document, token, and per-term
// document-frequency totals without changing or merging source segments.
func AggregateFTSCorpusStats(ctx context.Context, segments []FTSSegmentView) (FTSCorpusStats, error) {
	if ctx == nil {
		return FTSCorpusStats{}, fmt.Errorf("%w: nil context", ErrInvalidFTSStats)
	}
	if err := ctx.Err(); err != nil {
		return FTSCorpusStats{}, err
	}
	result := FTSCorpusStats{documentFrequencies: make(map[string]uint64)}
	work := 0
	for segmentIndex, segment := range segments {
		if segment.Dictionary == nil {
			return FTSCorpusStats{}, fmt.Errorf("%w: segment %d has nil dictionary", ErrInvalidFTSStats, segmentIndex)
		}
		documentCount := len(segment.Dictionary.documentLengths)
		deletedWords := []uint64(nil)
		if segment.DeletedDocuments != nil {
			deletedWords = segment.DeletedDocuments.Snapshot()
			if invalidFTSDeletionBits(deletedWords, uint64(documentCount)) {
				return FTSCorpusStats{}, fmt.Errorf("%w: segment %d deletion is outside its document domain", ErrInvalidFTSStats, segmentIndex)
			}
		}
		for documentID, documentLength := range segment.Dictionary.documentLengths {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return FTSCorpusStats{}, err
				}
			}
			work++
			if ftsDeleted(deletedWords, uint32(documentID)) {
				continue
			}
			if result.TotalDocuments == math.MaxUint64 || math.MaxUint64-result.TotalTokens < uint64(documentLength) {
				return FTSCorpusStats{}, fmt.Errorf("%w: corpus totals overflow", ErrInvalidFTSStats)
			}
			result.TotalDocuments++
			result.TotalTokens += uint64(documentLength)
		}
		terms, err := segment.Dictionary.terms(ctx)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return FTSCorpusStats{}, ctxErr
			}
			return FTSCorpusStats{}, fmt.Errorf("%w: segment %d term index is inconsistent", ErrInvalidFTSStats, segmentIndex)
		}
		if len(terms) != len(segment.Dictionary.postings) {
			return FTSCorpusStats{}, fmt.Errorf("%w: segment %d term index is inconsistent", ErrInvalidFTSStats, segmentIndex)
		}
		for termIndex, term := range terms {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return FTSCorpusStats{}, err
				}
			}
			work++
			var liveFrequency uint64
			iterator := segment.Dictionary.postings[termIndex].Iterator()
			for iterator.Next() {
				if work&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return FTSCorpusStats{}, err
					}
				}
				work++
				if !ftsDeleted(deletedWords, iterator.DocumentID()) {
					liveFrequency++
				}
			}
			if liveFrequency == 0 {
				continue
			}
			if math.MaxUint64-result.documentFrequencies[term] < liveFrequency {
				return FTSCorpusStats{}, fmt.Errorf("%w: term %q frequency overflows", ErrInvalidFTSStats, term)
			}
			result.documentFrequencies[term] += liveFrequency
		}
	}
	return result, nil
}

func ftsDeleted(words []uint64, documentID uint32) bool {
	word := uint64(documentID) >> 6
	return word < uint64(len(words)) && words[word]&(uint64(1)<<(documentID&63)) != 0
}

func invalidFTSDeletionBits(words []uint64, documentCount uint64) bool {
	fullWords := documentCount >> 6
	remainingBits := documentCount & 63
	for index, word := range words {
		wordIndex := uint64(index)
		switch {
		case wordIndex < fullWords:
			continue
		case wordIndex == fullWords && remainingBits != 0:
			validMask := (uint64(1) << remainingBits) - 1
			if word&^validMask != 0 {
				return true
			}
		default:
			if word != 0 {
				return true
			}
		}
	}
	return false
}
