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
	"math/bits"
	"strings"
)

// ErrInvalidFTSMerge identifies an invalid source segment, deletion snapshot,
// or count overflow during FTS dictionary compaction.
var ErrInvalidFTSMerge = errors.New("core: invalid FTS merge")

type ftsMergeSegment struct {
	dictionary   *FTSTermDictionary
	deletedWords []uint64
	deletePrefix []uint64
	outputBase   uint64
	liveCount    uint64
}

// MergeFTSTermDictionaries compacts source dictionaries in slice order.
// Deleted segment-local documents are removed and surviving document IDs are
// remapped densely into the concatenated output domain. Terms, tf, document
// lengths, and positions are preserved. Inputs and deletion bitmaps are
// snapshotted or immutable and are never mutated.
func MergeFTSTermDictionaries(ctx context.Context, sources []FTSSegmentView) (*FTSTermDictionary, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSMerge)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	segments := make([]ftsMergeSegment, len(sources))
	var totalDocuments, totalTokens uint64
	work := 0
	for segmentIndex, source := range sources {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		work++
		if source.Dictionary == nil {
			return nil, fmt.Errorf("%w: segment %d has nil dictionary", ErrInvalidFTSMerge, segmentIndex)
		}
		dictionary := source.Dictionary
		if uint64(len(dictionary.documentLengths)) != dictionary.stats.TotalDocuments ||
			len(dictionary.terms) != len(dictionary.postings) || len(dictionary.terms) != len(dictionary.maximumTF) {
			return nil, fmt.Errorf("%w: segment %d dictionary is inconsistent", ErrInvalidFTSMerge, segmentIndex)
		}
		var deletedWords []uint64
		if source.DeletedDocuments != nil {
			var valid bool
			deletedWords, valid = source.DeletedDocuments.SnapshotWithin(uint64(len(dictionary.documentLengths)))
			if !valid {
				return nil, fmt.Errorf("%w: segment %d deletion is outside its document domain", ErrInvalidFTSMerge, segmentIndex)
			}
		}
		deletePrefix := make([]uint64, len(deletedWords)+1)
		for index, word := range deletedWords {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			work++
			deletePrefix[index+1] = deletePrefix[index] + uint64(bits.OnesCount64(word))
		}
		segment := ftsMergeSegment{
			dictionary: dictionary, deletedWords: deletedWords,
			deletePrefix: deletePrefix, outputBase: totalDocuments,
		}
		var sourceTokens uint64
		for documentID, documentLength := range dictionary.documentLengths {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			work++
			if math.MaxUint64-sourceTokens < uint64(documentLength) {
				return nil, fmt.Errorf("%w: segment %d token statistics overflow", ErrInvalidFTSMerge, segmentIndex)
			}
			sourceTokens += uint64(documentLength)
			if ftsDeleted(deletedWords, uint32(documentID)) {
				continue
			}
			if totalDocuments == math.MaxUint64 || math.MaxUint64-totalTokens < uint64(documentLength) {
				return nil, fmt.Errorf("%w: output statistics overflow", ErrInvalidFTSMerge)
			}
			totalDocuments++
			totalTokens += uint64(documentLength)
			segment.liveCount++
		}
		if sourceTokens != dictionary.stats.TotalTokens {
			return nil, fmt.Errorf("%w: segment %d token statistics are inconsistent", ErrInvalidFTSMerge, segmentIndex)
		}
		segments[segmentIndex] = segment
	}
	if totalDocuments > math.MaxUint32 || totalDocuments > uint64(ftsMaximumInt()) {
		return nil, fmt.Errorf("%w: %d output documents exceed native format", ErrInvalidFTSMerge, totalDocuments)
	}

	documentLengths := make([]uint32, 0, int(totalDocuments))
	for _, segment := range segments {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		work++
		for documentID, documentLength := range segment.dictionary.documentLengths {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			work++
			if !ftsDeleted(segment.deletedWords, uint32(documentID)) {
				documentLengths = append(documentLengths, documentLength)
			}
		}
	}

	output := &FTSTermDictionary{
		terms:           make([]string, 0),
		postings:        make([]*FTSPostingList, 0),
		maximumTF:       make([]uint32, 0),
		documentLengths: documentLengths,
		stats: FTSSegmentStats{
			TotalDocuments: totalDocuments,
			TotalTokens:    totalTokens,
		},
	}
	cursors := make([]int, len(segments))
	for {
		minimumTerm := ""
		found := false
		for segmentIndex, segment := range segments {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			work++
			cursor := cursors[segmentIndex]
			if cursor >= len(segment.dictionary.terms) {
				continue
			}
			term := segment.dictionary.terms[cursor]
			if !found || term < minimumTerm {
				minimumTerm, found = term, true
			}
		}
		if !found {
			break
		}

		postings := make([]FTSPosting, 0)
		var maximumTermFrequency uint32
		for segmentIndex, segment := range segments {
			cursor := cursors[segmentIndex]
			if cursor >= len(segment.dictionary.terms) || segment.dictionary.terms[cursor] != minimumTerm {
				continue
			}
			list := segment.dictionary.postings[cursor]
			if list == nil {
				return nil, fmt.Errorf("%w: segment %d term %q has nil postings", ErrInvalidFTSMerge, segmentIndex, minimumTerm)
			}
			iterator := list.Iterator()
			for iterator.Next() {
				if work&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				work++
				oldDocumentID := iterator.DocumentID()
				if uint64(oldDocumentID) >= uint64(len(segment.dictionary.documentLengths)) {
					return nil, fmt.Errorf("%w: segment %d term %q document %d is outside its domain", ErrInvalidFTSMerge, segmentIndex, minimumTerm, oldDocumentID)
				}
				if ftsDeleted(segment.deletedWords, oldDocumentID) {
					continue
				}
				if iterator.DocumentLength() != segment.dictionary.documentLengths[oldDocumentID] {
					return nil, fmt.Errorf("%w: segment %d term %q document length is inconsistent", ErrInvalidFTSMerge, segmentIndex, minimumTerm)
				}
				newDocumentID, ok := segment.remap(oldDocumentID)
				if !ok {
					return nil, fmt.Errorf("%w: segment %d term %q document remap overflow", ErrInvalidFTSMerge, segmentIndex, minimumTerm)
				}
				positions, err := ftsPostingIteratorPositions(ctx, iterator)
				if err != nil {
					return nil, fmt.Errorf("%w: segment %d term %q positions: %v", ErrInvalidFTSMerge, segmentIndex, minimumTerm, err)
				}
				termFrequency := iterator.TermFrequency()
				maximumTermFrequency = max(maximumTermFrequency, termFrequency)
				postings = append(postings, FTSPosting{
					DocumentID: newDocumentID, TermFrequency: termFrequency,
					DocumentLength: iterator.DocumentLength(), Positions: positions,
				})
			}
			cursors[segmentIndex]++
		}
		if len(postings) == 0 {
			continue
		}
		postingList, err := BuildFTSPostingList(ctx, postings)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, fmt.Errorf("%w: term %q: %v", ErrInvalidFTSMerge, minimumTerm, err)
		}
		output.terms = append(output.terms, strings.Clone(minimumTerm))
		output.postings = append(output.postings, postingList)
		output.maximumTF = append(output.maximumTF, maximumTermFrequency)
	}
	return output, nil
}

func (s ftsMergeSegment) remap(documentID uint32) (uint32, bool) {
	wordIndex := uint64(documentID) >> 6
	deletedBefore := uint64(0)
	if wordIndex < uint64(len(s.deletePrefix)) {
		deletedBefore = s.deletePrefix[wordIndex]
	} else if len(s.deletePrefix) > 0 {
		deletedBefore = s.deletePrefix[len(s.deletePrefix)-1]
	}
	if wordIndex < uint64(len(s.deletedWords)) {
		bitIndex := documentID & 63
		if bitIndex > 0 {
			deletedBefore += uint64(bits.OnesCount64(s.deletedWords[wordIndex] & ((uint64(1) << bitIndex) - 1)))
		}
	}
	newDocumentID := s.outputBase + uint64(documentID) - deletedBefore
	if newDocumentID > math.MaxUint32 || newDocumentID >= s.outputBase+s.liveCount {
		return 0, false
	}
	return uint32(newDocumentID), true
}

func ftsPostingIteratorPositions(ctx context.Context, iterator *FTSPostingIterator) ([]uint32, error) {
	if iterator == nil || iterator.list == nil || !iterator.Valid() {
		return nil, nil
	}
	start := uint64(iterator.list.positionsOffset) + uint64(iterator.positionOffsets[iterator.indexInBlock])
	end := start + uint64(iterator.positionLengths[iterator.indexInBlock])
	return decodeFTSPositionDeltas(ctx, iterator.list.data[start:end], iterator.TermFrequency())
}
