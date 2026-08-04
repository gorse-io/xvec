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

// ErrInvalidFTSQueryExecution identifies an invalid dictionary, AST, deletion
// snapshot, or iterator call.
var ErrInvalidFTSQueryExecution = errors.New("core: invalid FTS query execution")

// FTSQueryExecutionOptions configures one immutable segment query.
type FTSQueryExecutionOptions struct {
	// DeletedDocuments contains segment-local tombstones. It is snapshotted by
	// NewFTSQueryIterator and may be changed by the caller afterward.
	DeletedDocuments *ailego.Bitmap
}

// FTSQueryIterator lazily emits exact matching segment-local document IDs in
// ascending order. Iterators built with NewFTSScoredQueryIterator also expose
// the current document's BM25 score.
type FTSQueryIterator struct {
	root         ftsDocumentIterator
	deletedWords []uint64
	scorer       *BM25Scorer
	documentID   uint32
	score        float32
	valid        bool
	err          error
}

// NewFTSQueryIterator snapshots and simplifies node, then builds a lazy term,
// phrase, and boolean iterator tree over dictionary.
func NewFTSQueryIterator(ctx context.Context, dictionary *FTSTermDictionary, node FTSQueryNode, options FTSQueryExecutionOptions) (*FTSQueryIterator, error) {
	return newFTSQueryIterator(ctx, dictionary, node, nil, options)
}

// NewFTSScoredQueryIterator builds an exact iterator whose Score method uses
// scorer's immutable corpus snapshot. The scorer may describe multiple
// segments so every segment uses the same live IDF and average length.
func NewFTSScoredQueryIterator(ctx context.Context, dictionary *FTSTermDictionary, node FTSQueryNode, scorer *BM25Scorer, options FTSQueryExecutionOptions) (*FTSQueryIterator, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSQueryExecution)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if scorer == nil {
		return nil, fmt.Errorf("%w: BM25 scorer is nil", ErrInvalidFTSQueryExecution)
	}
	return newFTSQueryIterator(ctx, dictionary, node, scorer, options)
}

func newFTSQueryIterator(ctx context.Context, dictionary *FTSTermDictionary, node FTSQueryNode, scorer *BM25Scorer, options FTSQueryExecutionOptions) (*FTSQueryIterator, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSQueryExecution)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if dictionary == nil {
		return nil, fmt.Errorf("%w: dictionary is nil", ErrInvalidFTSQueryExecution)
	}
	simplified, err := SimplifyFTSQuery(ctx, node)
	if err != nil {
		return nil, err
	}
	var deletedWords []uint64
	if options.DeletedDocuments != nil {
		deletedWords = options.DeletedDocuments.Snapshot()
		if invalidFTSDeletionBits(deletedWords, uint64(len(dictionary.documentLengths))) {
			return nil, fmt.Errorf("%w: deletion is outside the document domain", ErrInvalidFTSQueryExecution)
		}
	}
	iterator := &FTSQueryIterator{deletedWords: deletedWords, scorer: scorer}
	if simplified.Modifier().MustNot {
		return iterator, nil
	}
	iterator.root, err = buildFTSDocumentIterator(ctx, dictionary, simplified, scorer)
	if err != nil {
		return nil, err
	}
	return iterator, nil
}

// Next advances to the next live match.
func (i *FTSQueryIterator) Next(ctx context.Context) bool {
	if !i.prepare(ctx) || i.root == nil {
		return false
	}
	for {
		documentID, ok, err := i.root.next(ctx)
		if err != nil {
			i.fail(err)
			return false
		}
		if !ok {
			i.valid, i.score = false, 0
			return false
		}
		if !ftsDeleted(i.deletedWords, documentID) {
			score, err := i.root.score(ctx)
			if err != nil {
				i.fail(err)
				return false
			}
			i.documentID, i.valid = documentID, true
			i.score = score
			return true
		}
	}
}

// Advance moves to the first live match with document ID >= target. If the
// current match already satisfies target, it remains current.
func (i *FTSQueryIterator) Advance(ctx context.Context, target uint32) bool {
	if !i.prepare(ctx) || i.root == nil {
		return false
	}
	if i.valid && i.documentID >= target {
		return true
	}
	for {
		documentID, ok, err := i.root.advance(ctx, target)
		if err != nil {
			i.fail(err)
			return false
		}
		if !ok {
			i.valid, i.score = false, 0
			return false
		}
		if !ftsDeleted(i.deletedWords, documentID) {
			score, err := i.root.score(ctx)
			if err != nil {
				i.fail(err)
				return false
			}
			i.documentID, i.valid = documentID, true
			i.score = score
			return true
		}
		if documentID == math.MaxUint32 {
			i.valid, i.score = false, 0
			return false
		}
		target = documentID + 1
	}
}

// Valid reports whether the iterator addresses a live match.
func (i *FTSQueryIterator) Valid() bool { return i != nil && i.valid && i.err == nil }

// DocumentID returns the current document ID, or zero when invalid.
func (i *FTSQueryIterator) DocumentID() uint32 {
	if !i.Valid() {
		return 0
	}
	return i.documentID
}

// Score returns the current BM25 score, or zero for an invalid or unscored
// iterator.
func (i *FTSQueryIterator) Score() float32 {
	if !i.Valid() || i.scorer == nil {
		return 0
	}
	return i.score
}

// Cost returns the posting-based match-work estimate before deletions.
func (i *FTSQueryIterator) Cost() uint64 {
	if i == nil || i.root == nil {
		return 0
	}
	return i.root.cost()
}

// Err returns the first iteration error.
func (i *FTSQueryIterator) Err() error {
	if i == nil {
		return fmt.Errorf("%w: nil iterator", ErrInvalidFTSQueryExecution)
	}
	return i.err
}

func (i *FTSQueryIterator) prepare(ctx context.Context) bool {
	if i == nil {
		return false
	}
	if i.err != nil {
		return false
	}
	if ctx == nil {
		i.fail(fmt.Errorf("%w: nil iteration context", ErrInvalidFTSQueryExecution))
		return false
	}
	if err := ctx.Err(); err != nil {
		i.fail(err)
		return false
	}
	return true
}

func (i *FTSQueryIterator) fail(err error) {
	if i.err == nil {
		i.err = err
	}
	i.valid = false
	i.score = 0
}

type ftsDocumentIterator interface {
	next(context.Context) (uint32, bool, error)
	advance(context.Context, uint32) (uint32, bool, error)
	current() (uint32, bool)
	cost() uint64
	score(context.Context) (float32, error)
}

type ftsTermDocumentIterator struct {
	posting *FTSPostingIterator
	scorer  *BM25Scorer
	idf     float32
	boost   float32
}

func (i *ftsTermDocumentIterator) next(ctx context.Context) (uint32, bool, error) {
	if err := ctx.Err(); err != nil {
		return 0, false, err
	}
	if !i.posting.Next() {
		return 0, false, nil
	}
	return i.posting.DocumentID(), true, nil
}

func (i *ftsTermDocumentIterator) advance(ctx context.Context, target uint32) (uint32, bool, error) {
	if err := ctx.Err(); err != nil {
		return 0, false, err
	}
	if !i.posting.Advance(target) {
		return 0, false, nil
	}
	return i.posting.DocumentID(), true, nil
}

func (i *ftsTermDocumentIterator) current() (uint32, bool) {
	if i == nil || i.posting == nil || !i.posting.Valid() {
		return 0, false
	}
	return i.posting.DocumentID(), true
}

func (i *ftsTermDocumentIterator) cost() uint64 {
	if i == nil || i.posting == nil || i.posting.list == nil {
		return 0
	}
	return uint64(i.posting.list.DocumentFrequency())
}

func (i *ftsTermDocumentIterator) score(ctx context.Context) (float32, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if i == nil || i.scorer == nil || i.posting == nil || !i.posting.Valid() {
		return 0, nil
	}
	return i.scorer.ScoreWithIDFAndBoost(i.idf, i.posting.TermFrequency(), i.posting.DocumentLength(), i.boost), nil
}

func (i *ftsTermDocumentIterator) positions(ctx context.Context) ([]uint32, error) {
	if i == nil || i.posting == nil || !i.posting.Valid() {
		return nil, nil
	}
	start := uint64(i.posting.list.positionsOffset) + uint64(i.posting.positionOffsets[i.posting.indexInBlock])
	end := start + uint64(i.posting.positionLengths[i.posting.indexInBlock])
	return decodeFTSPositionDeltas(ctx, i.posting.list.data[start:end], i.posting.TermFrequency())
}

type ftsAndDocumentIterator struct {
	must            []ftsDocumentIterator
	mustNot         []ftsDocumentIterator
	should          []ftsDocumentIterator
	currentDocument uint32
	valid           bool
}

func newFTSAndDocumentIterator(must, mustNot, should []ftsDocumentIterator) *ftsAndDocumentIterator {
	sort.SliceStable(must, func(left, right int) bool { return must[left].cost() < must[right].cost() })
	return &ftsAndDocumentIterator{must: must, mustNot: mustNot, should: should}
}

func (i *ftsAndDocumentIterator) next(ctx context.Context) (uint32, bool, error) {
	if len(i.must) == 0 {
		return 0, false, nil
	}
	candidate, ok, err := i.must[0].next(ctx)
	if err != nil || !ok {
		i.valid = false
		return 0, false, err
	}
	return i.align(ctx, candidate)
}

func (i *ftsAndDocumentIterator) advance(ctx context.Context, target uint32) (uint32, bool, error) {
	if i.valid && i.currentDocument >= target {
		return i.currentDocument, true, nil
	}
	if len(i.must) == 0 {
		return 0, false, nil
	}
	candidate, ok, err := i.must[0].advance(ctx, target)
	if err != nil || !ok {
		i.valid = false
		return 0, false, err
	}
	return i.align(ctx, candidate)
}

func (i *ftsAndDocumentIterator) align(ctx context.Context, candidate uint32) (uint32, bool, error) {
	work := 0
	for {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				i.valid = false
				return 0, false, err
			}
		}
		work++
		allMatch := true
		for index := 1; index < len(i.must); index++ {
			other, ok, err := i.must[index].advance(ctx, candidate)
			if err != nil || !ok {
				i.valid = false
				return 0, false, err
			}
			if other != candidate {
				candidate, ok, err = i.must[0].advance(ctx, other)
				if err != nil || !ok {
					i.valid = false
					return 0, false, err
				}
				allMatch = false
				break
			}
		}
		if !allMatch {
			continue
		}
		excluded := false
		for _, negative := range i.mustNot {
			documentID, ok, err := negative.advance(ctx, candidate)
			if err != nil {
				i.valid = false
				return 0, false, err
			}
			if ok && documentID == candidate {
				excluded = true
				break
			}
		}
		if !excluded {
			i.currentDocument, i.valid = candidate, true
			return candidate, true, nil
		}
		nextCandidate, nextOK, nextErr := i.must[0].next(ctx)
		candidate = nextCandidate
		if nextErr != nil || !nextOK {
			i.valid = false
			return 0, false, nextErr
		}
	}
}

func (i *ftsAndDocumentIterator) current() (uint32, bool) {
	if i == nil || !i.valid {
		return 0, false
	}
	return i.currentDocument, true
}

func (i *ftsAndDocumentIterator) cost() uint64 {
	if i == nil || len(i.must) == 0 {
		return 0
	}
	return i.must[0].cost()
}

func (i *ftsAndDocumentIterator) score(ctx context.Context) (float32, error) {
	if i == nil || !i.valid {
		return 0, nil
	}
	var total float32
	for _, iterator := range i.must {
		score, err := iterator.score(ctx)
		if err != nil {
			return 0, err
		}
		total += score
	}
	for _, iterator := range i.should {
		documentID, ok, err := iterator.advance(ctx, i.currentDocument)
		if err != nil {
			return 0, err
		}
		if !ok || documentID != i.currentDocument {
			continue
		}
		score, err := iterator.score(ctx)
		if err != nil {
			return 0, err
		}
		total += score
	}
	return total, nil
}

type ftsOrDocumentIterator struct {
	children        []ftsDocumentIterator
	started         bool
	currentDocument uint32
	valid           bool
}

func (i *ftsOrDocumentIterator) next(ctx context.Context) (uint32, bool, error) {
	if !i.started {
		i.started = true
		for _, child := range i.children {
			if _, _, err := child.next(ctx); err != nil {
				return 0, false, err
			}
		}
	} else if i.valid {
		for _, child := range i.children {
			if documentID, ok := child.current(); ok && documentID == i.currentDocument {
				if _, _, err := child.next(ctx); err != nil {
					return 0, false, err
				}
			}
		}
	}
	return i.minimum(ctx)
}

func (i *ftsOrDocumentIterator) advance(ctx context.Context, target uint32) (uint32, bool, error) {
	if i.valid && i.currentDocument >= target {
		return i.currentDocument, true, nil
	}
	i.started = true
	for _, child := range i.children {
		documentID, ok := child.current()
		if !ok || documentID < target {
			if _, _, err := child.advance(ctx, target); err != nil {
				return 0, false, err
			}
		}
	}
	return i.minimum(ctx)
}

func (i *ftsOrDocumentIterator) minimum(ctx context.Context) (uint32, bool, error) {
	if err := ctx.Err(); err != nil {
		i.valid = false
		return 0, false, err
	}
	minimum, found := uint32(0), false
	for _, child := range i.children {
		if documentID, ok := child.current(); ok && (!found || documentID < minimum) {
			minimum, found = documentID, true
		}
	}
	i.currentDocument, i.valid = minimum, found
	return minimum, found, nil
}

func (i *ftsOrDocumentIterator) current() (uint32, bool) {
	if i == nil || !i.valid {
		return 0, false
	}
	return i.currentDocument, true
}

func (i *ftsOrDocumentIterator) cost() uint64 {
	var total uint64
	for _, child := range i.children {
		cost := child.cost()
		if math.MaxUint64-total < cost {
			return math.MaxUint64
		}
		total += cost
	}
	return total
}

func (i *ftsOrDocumentIterator) score(ctx context.Context) (float32, error) {
	if i == nil || !i.valid {
		return 0, nil
	}
	var total float32
	for _, child := range i.children {
		documentID, ok := child.current()
		if !ok || documentID != i.currentDocument {
			continue
		}
		score, err := child.score(ctx)
		if err != nil {
			return 0, err
		}
		total += score
	}
	return total, nil
}

type ftsPhraseTermIterator struct {
	term     string
	iterator *ftsTermDocumentIterator
}

type ftsPhraseDocumentIterator struct {
	conjunction     *ftsAndDocumentIterator
	terms           []ftsPhraseTermIterator
	currentDocument uint32
	valid           bool
}

func (i *ftsPhraseDocumentIterator) next(ctx context.Context) (uint32, bool, error) {
	documentID, ok, err := i.conjunction.next(ctx)
	return i.findMatch(ctx, documentID, ok, err)
}

func (i *ftsPhraseDocumentIterator) advance(ctx context.Context, target uint32) (uint32, bool, error) {
	if i.valid && i.currentDocument >= target {
		return i.currentDocument, true, nil
	}
	documentID, ok, err := i.conjunction.advance(ctx, target)
	return i.findMatch(ctx, documentID, ok, err)
}

func (i *ftsPhraseDocumentIterator) findMatch(ctx context.Context, documentID uint32, ok bool, err error) (uint32, bool, error) {
	for ok && err == nil {
		matches, matchErr := i.verifyPositions(ctx)
		if matchErr != nil {
			i.valid = false
			return 0, false, matchErr
		}
		if matches {
			i.currentDocument, i.valid = documentID, true
			return documentID, true, nil
		}
		documentID, ok, err = i.conjunction.next(ctx)
	}
	i.valid = false
	return 0, false, err
}

func (i *ftsPhraseDocumentIterator) verifyPositions(ctx context.Context) (bool, error) {
	if len(i.terms) == 0 {
		return false, nil
	}
	positionsByTerm := make(map[string][]uint32, len(i.terms))
	positionLists := make([][]uint32, len(i.terms))
	anchor := 0
	for index, term := range i.terms {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return false, err
			}
		}
		positions, found := positionsByTerm[term.term]
		if !found {
			var err error
			positions, err = term.iterator.positions(ctx)
			if err != nil {
				return false, err
			}
			positionsByTerm[term.term] = positions
		}
		if len(positions) == 0 {
			return false, nil
		}
		positionLists[index] = positions
		if len(positions) < len(positionLists[anchor]) || index == 0 {
			anchor = index
		}
	}
	work := 0
	for positionIndex, anchorPosition := range positionLists[anchor] {
		if positionIndex&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return false, err
			}
		}
		if uint64(anchorPosition) < uint64(anchor) {
			continue
		}
		start := uint64(anchorPosition) - uint64(anchor)
		matched := true
		for termIndex, positions := range positionLists {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return false, err
				}
			}
			work++
			if termIndex == anchor {
				continue
			}
			expected := start + uint64(termIndex)
			if expected > math.MaxUint32 {
				matched = false
				break
			}
			position := uint32(expected)
			found := sort.Search(len(positions), func(index int) bool { return positions[index] >= position })
			if found == len(positions) || positions[found] != position {
				matched = false
				break
			}
		}
		if matched {
			return true, nil
		}
	}
	return false, nil
}

func (i *ftsPhraseDocumentIterator) current() (uint32, bool) {
	if i == nil || !i.valid {
		return 0, false
	}
	return i.currentDocument, true
}

func (i *ftsPhraseDocumentIterator) cost() uint64 {
	if i == nil || i.conjunction == nil {
		return 0
	}
	return i.conjunction.cost()
}

func (i *ftsPhraseDocumentIterator) score(ctx context.Context) (float32, error) {
	if i == nil || !i.valid || i.conjunction == nil {
		return 0, nil
	}
	return i.conjunction.score(ctx)
}

func buildFTSDocumentIterator(ctx context.Context, dictionary *FTSTermDictionary, node FTSQueryNode, scorer *BM25Scorer) (ftsDocumentIterator, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch typed := node.(type) {
	case *FTSTermQueryNode:
		_, posting, found := dictionary.Lookup(typed.Term)
		if !found {
			return nil, nil
		}
		return &ftsTermDocumentIterator{
			posting: posting.Iterator(), scorer: scorer,
			idf: ftsTermIDF(scorer, typed.Term), boost: typed.Flags.Boost,
		}, nil
	case *FTSPhraseQueryNode:
		if len(typed.Terms) == 0 {
			return nil, nil
		}
		terms := make([]ftsPhraseTermIterator, len(typed.Terms))
		must := make([]ftsDocumentIterator, len(typed.Terms))
		for index, term := range typed.Terms {
			if index&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			_, posting, found := dictionary.Lookup(term)
			if !found {
				return nil, nil
			}
			iterator := &ftsTermDocumentIterator{
				posting: posting.Iterator(), scorer: scorer,
				idf: ftsTermIDF(scorer, term), boost: typed.Flags.Boost,
			}
			terms[index] = ftsPhraseTermIterator{term: term, iterator: iterator}
			must[index] = iterator
		}
		return &ftsPhraseDocumentIterator{conjunction: newFTSAndDocumentIterator(must, nil, nil), terms: terms}, nil
	case *FTSAndQueryNode:
		must := make([]ftsDocumentIterator, 0, len(typed.Children))
		mustNot := make([]ftsDocumentIterator, 0)
		should := make([]ftsDocumentIterator, 0)
		for _, child := range typed.Children {
			iterator, err := buildFTSDocumentIterator(ctx, dictionary, child, scorer)
			if err != nil {
				return nil, err
			}
			modifier := child.Modifier()
			if iterator == nil {
				if !modifier.MustNot && !modifier.Should {
					return nil, nil
				}
				continue
			}
			if modifier.MustNot {
				mustNot = append(mustNot, iterator)
			} else if modifier.Should {
				if scorer != nil {
					should = append(should, iterator)
				}
			} else {
				must = append(must, iterator)
			}
		}
		if len(must) == 0 {
			return nil, nil
		}
		if len(must) == 1 && len(mustNot) == 0 && len(should) == 0 {
			return must[0], nil
		}
		return newFTSAndDocumentIterator(must, mustNot, should), nil
	case *FTSOrQueryNode:
		children := make([]ftsDocumentIterator, 0, len(typed.Children))
		for _, child := range typed.Children {
			modifier := child.Modifier()
			if modifier.Must || modifier.MustNot {
				return nil, fmt.Errorf("%w: non-canonical OR occurrence modifier", ErrInvalidFTSQueryExecution)
			}
			iterator, err := buildFTSDocumentIterator(ctx, dictionary, child, scorer)
			if err != nil {
				return nil, err
			}
			if iterator != nil {
				children = append(children, iterator)
			}
		}
		switch len(children) {
		case 0:
			return nil, nil
		case 1:
			return children[0], nil
		default:
			return &ftsOrDocumentIterator{children: children}, nil
		}
	case *FTSEmptyQueryNode:
		return nil, nil
	default:
		return nil, fmt.Errorf("%w: unknown node type %T", ErrInvalidFTSQueryExecution, node)
	}
}

func ftsTermIDF(scorer *BM25Scorer, term string) float32 {
	if scorer == nil {
		return 0
	}
	return scorer.TermIDF(term)
}
