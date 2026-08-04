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
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	ftsDictionaryVersion    = uint16(1)
	ftsDictionaryHeaderSize = 64
)

var (
	ftsDictionaryMagic = [4]byte{'Z', 'V', 'F', 'D'}

	// ErrInvalidFTSDocument identifies a non-sequential document or malformed
	// token stream supplied to FTSFieldBuilder.
	ErrInvalidFTSDocument = errors.New("core: invalid FTS document")
	// ErrInvalidFTSDictionary identifies an invalid in-memory dictionary input.
	ErrInvalidFTSDictionary = errors.New("core: invalid FTS dictionary")
	// ErrCorruptFTSDictionary identifies malformed or checksummed dictionary bytes.
	ErrCorruptFTSDictionary = errors.New("core: corrupt FTS dictionary")
)

// FTSSegmentStats is the exact document/token summary used to calculate
// average document length. Empty documents count toward TotalDocuments.
type FTSSegmentStats struct {
	TotalDocuments uint64
	TotalTokens    uint64
}

// AverageDocumentLength returns total tokens divided by total documents, or
// one for an empty segment, matching the baseline scorer convention.
func (s FTSSegmentStats) AverageDocumentLength() float64 {
	if s.TotalDocuments == 0 {
		return 1
	}
	return float64(s.TotalTokens) / float64(s.TotalDocuments)
}

// FTSTermInfo is immutable dictionary metadata for one byte-ordered term.
type FTSTermInfo struct {
	Term                 string
	DocumentFrequency    uint32
	MaximumTermFrequency uint32
}

// FTSTermDictionary is an immutable term dictionary with front-coded native
// persistence and independently checksummed compressed posting lists.
type FTSTermDictionary struct {
	terms           []string
	postings        []*FTSPostingList
	maximumTF       []uint32
	documentLengths []uint32
	stats           FTSSegmentStats
	data            []byte
}

// FTSFieldBuilder builds one segment-local dictionary. It is intentionally
// single-writer; dictionaries returned by Build are safe for concurrent use.
type FTSFieldBuilder struct {
	documentLengths []uint32
	totalTokens     uint64
	postings        map[string][]FTSPosting
}

// NewFTSFieldBuilder creates an empty segment-local FTS builder.
func NewFTSFieldBuilder() *FTSFieldBuilder {
	return &FTSFieldBuilder{postings: make(map[string][]FTSPosting)}
}

// AddDocument indexes one already-analyzed token stream. Document IDs must be
// dense and start at zero so they share the segment's forward-row domain.
func (b *FTSFieldBuilder) AddDocument(ctx context.Context, documentID uint32, tokens []Token) error {
	if ctx == nil {
		return fmt.Errorf("%w: nil context", ErrInvalidFTSDocument)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if b == nil {
		return fmt.Errorf("%w: nil builder", ErrInvalidFTSDocument)
	}
	if uint64(len(b.documentLengths)) >= math.MaxUint32 || documentID != uint32(len(b.documentLengths)) {
		return fmt.Errorf("%w: document ID %d, want %d", ErrInvalidFTSDocument, documentID, len(b.documentLengths))
	}
	if uint64(len(tokens)) > math.MaxUint32 || math.MaxUint64-b.totalTokens < uint64(len(tokens)) {
		return fmt.Errorf("%w: document token count overflows statistics", ErrInvalidFTSDocument)
	}
	for index := 1; index < len(tokens); index++ {
		if tokens[index-1].Position > tokens[index].Position {
			return fmt.Errorf("%w: token positions decrease at %d", ErrInvalidFTSDocument, index)
		}
	}
	positionsByTerm := make(map[string][]uint32)
	for index, token := range tokens {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		positionsByTerm[token.Text] = append(positionsByTerm[token.Text], token.Position)
	}
	documentLength := uint32(len(tokens))
	for term, positions := range positionsByTerm {
		ownedTerm := term
		if _, found := b.postings[term]; !found {
			ownedTerm = strings.Clone(term)
		}
		ownedPositions := append([]uint32(nil), positions...)
		b.postings[ownedTerm] = append(b.postings[ownedTerm], FTSPosting{
			DocumentID:     documentID,
			TermFrequency:  uint32(len(positions)),
			DocumentLength: documentLength,
			Positions:      ownedPositions,
		})
	}
	b.documentLengths = append(b.documentLengths, documentLength)
	b.totalTokens += uint64(documentLength)
	return nil
}

// Build seals a point-in-time immutable dictionary without consuming the
// builder; later AddDocument calls cannot mutate the returned snapshot.
func (b *FTSFieldBuilder) Build(ctx context.Context) (*FTSTermDictionary, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSDictionary)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b == nil {
		return nil, fmt.Errorf("%w: nil builder", ErrInvalidFTSDictionary)
	}
	terms := make([]string, 0, len(b.postings))
	for term := range b.postings {
		terms = append(terms, term)
	}
	sort.Strings(terms)
	dictionary := &FTSTermDictionary{
		terms:           make([]string, len(terms)),
		postings:        make([]*FTSPostingList, len(terms)),
		maximumTF:       make([]uint32, len(terms)),
		documentLengths: append([]uint32(nil), b.documentLengths...),
		stats: FTSSegmentStats{
			TotalDocuments: uint64(len(b.documentLengths)),
			TotalTokens:    b.totalTokens,
		},
	}
	for index, term := range terms {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		postingList, err := BuildFTSPostingList(ctx, b.postings[term])
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, fmt.Errorf("%w: term %q: %v", ErrInvalidFTSDictionary, term, err)
		}
		var maximumTF uint32
		for _, posting := range b.postings[term] {
			maximumTF = max(maximumTF, posting.TermFrequency)
		}
		dictionary.terms[index] = strings.Clone(term)
		dictionary.postings[index] = postingList
		dictionary.maximumTF[index] = maximumTF
	}
	return dictionary, nil
}

// Stats returns the immutable segment summary.
func (d *FTSTermDictionary) Stats() FTSSegmentStats {
	if d == nil {
		return FTSSegmentStats{}
	}
	return d.stats
}

// DocumentLength returns a segment-local document's token count.
func (d *FTSTermDictionary) DocumentLength(documentID uint32) (uint32, bool) {
	if d == nil || uint64(documentID) >= uint64(len(d.documentLengths)) {
		return 0, false
	}
	return d.documentLengths[documentID], true
}

// TermCount returns the number of unique terms.
func (d *FTSTermDictionary) TermCount() int {
	if d == nil {
		return 0
	}
	return len(d.terms)
}

// Terms returns an owned byte-lexicographically sorted term slice.
func (d *FTSTermDictionary) Terms() []string {
	if d == nil {
		return []string{}
	}
	return append([]string(nil), d.terms...)
}

// Lookup returns immutable term metadata and its posting list.
func (d *FTSTermDictionary) Lookup(term string) (FTSTermInfo, *FTSPostingList, bool) {
	if d == nil {
		return FTSTermInfo{}, nil, false
	}
	index := sort.SearchStrings(d.terms, term)
	if index == len(d.terms) || d.terms[index] != term {
		return FTSTermInfo{}, nil, false
	}
	return d.termInfo(index), d.postings[index], true
}

// Prefix returns up to limit terms beginning with prefix in byte-lexical
// order. A zero limit means no limit.
func (d *FTSTermDictionary) Prefix(prefix string, limit int) []FTSTermInfo {
	if d == nil || limit < 0 {
		return []FTSTermInfo{}
	}
	start := sort.Search(len(d.terms), func(index int) bool { return d.terms[index] >= prefix })
	result := make([]FTSTermInfo, 0)
	for index := start; index < len(d.terms) && strings.HasPrefix(d.terms[index], prefix); index++ {
		if limit > 0 && len(result) >= limit {
			break
		}
		result = append(result, d.termInfo(index))
	}
	return result
}

func (d *FTSTermDictionary) termInfo(index int) FTSTermInfo {
	return FTSTermInfo{
		Term:                 d.terms[index],
		DocumentFrequency:    d.postings[index].DocumentFrequency(),
		MaximumTermFrequency: d.maximumTF[index],
	}
}

// Encode serializes the dictionary into the versioned native Go format.
func (d *FTSTermDictionary) Encode(ctx context.Context) ([]byte, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSDictionary)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if d == nil || len(d.terms) != len(d.postings) || len(d.terms) != len(d.maximumTF) || uint64(len(d.documentLengths)) != d.stats.TotalDocuments {
		return nil, fmt.Errorf("%w: inconsistent dictionary", ErrInvalidFTSDictionary)
	}
	if uint64(len(d.terms)) > math.MaxUint32 || uint64(len(d.documentLengths)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: counts exceed uint32", ErrInvalidFTSDictionary)
	}
	var termTable []byte
	postingBytes := make([][]byte, len(d.postings))
	previousTerm := ""
	for index, term := range d.terms {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if index > 0 && previousTerm >= term {
			return nil, fmt.Errorf("%w: terms are not strictly sorted", ErrInvalidFTSDictionary)
		}
		if d.postings[index] == nil || d.postings[index].DocumentFrequency() == 0 || d.maximumTF[index] == 0 {
			return nil, fmt.Errorf("%w: term %q has empty metadata", ErrInvalidFTSDictionary, term)
		}
		postingBytes[index] = d.postings[index].data
		prefixLength := commonFTSPrefixLength(previousTerm, term)
		termTable = binary.AppendUvarint(termTable, uint64(prefixLength))
		termTable = binary.AppendUvarint(termTable, uint64(len(term)-prefixLength))
		termTable = append(termTable, term[prefixLength:]...)
		termTable = binary.AppendUvarint(termTable, uint64(len(postingBytes[index])))
		termTable = binary.AppendUvarint(termTable, uint64(d.maximumTF[index]))
		previousTerm = term
	}

	documentLengthBytes := uint64(len(d.documentLengths)) * 4
	termsOffset := uint64(ftsDictionaryHeaderSize) + documentLengthBytes
	postingsOffset := termsOffset + uint64(len(termTable))
	totalSize := postingsOffset
	for _, posting := range postingBytes {
		totalSize += uint64(len(posting))
	}
	if totalSize > math.MaxUint32 || totalSize > ftsMaximumInt() {
		return nil, fmt.Errorf("%w: encoded dictionary exceeds uint32", ErrInvalidFTSDictionary)
	}
	output := make([]byte, ftsDictionaryHeaderSize, int(totalSize))
	for _, documentLength := range d.documentLengths {
		output = binary.LittleEndian.AppendUint32(output, documentLength)
	}
	output = append(output, termTable...)
	for _, posting := range postingBytes {
		output = append(output, posting...)
	}
	copy(output[0:4], ftsDictionaryMagic[:])
	binary.LittleEndian.PutUint16(output[4:6], ftsDictionaryVersion)
	binary.LittleEndian.PutUint16(output[6:8], ftsDictionaryHeaderSize)
	binary.LittleEndian.PutUint32(output[8:12], uint32(len(d.terms)))
	binary.LittleEndian.PutUint32(output[12:16], uint32(len(d.documentLengths)))
	binary.LittleEndian.PutUint64(output[16:24], d.stats.TotalTokens)
	binary.LittleEndian.PutUint32(output[24:28], ftsDictionaryHeaderSize)
	binary.LittleEndian.PutUint32(output[28:32], uint32(termsOffset))
	binary.LittleEndian.PutUint32(output[32:36], uint32(postingsOffset))
	binary.LittleEndian.PutUint32(output[36:40], uint32(totalSize))
	binary.LittleEndian.PutUint32(output[40:44], ailego.CRC32C(output[ftsDictionaryHeaderSize:]))
	binary.LittleEndian.PutUint32(output[60:64], ailego.CRC32C(output[:60]))
	return output, nil
}

// OpenFTSTermDictionary validates and opens a native Go dictionary from an
// owned copy of data.
func OpenFTSTermDictionary(ctx context.Context, data []byte) (*FTSTermDictionary, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrCorruptFTSDictionary)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) < ftsDictionaryHeaderSize || uint64(len(data)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: invalid file length", ErrCorruptFTSDictionary)
	}
	if string(data[0:4]) != string(ftsDictionaryMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrCorruptFTSDictionary)
	}
	if version := binary.LittleEndian.Uint16(data[4:6]); version != ftsDictionaryVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrCorruptFTSDictionary, version)
	}
	if headerSize := binary.LittleEndian.Uint16(data[6:8]); headerSize != ftsDictionaryHeaderSize {
		return nil, fmt.Errorf("%w: unsupported header size %d", ErrCorruptFTSDictionary, headerSize)
	}
	termCount := binary.LittleEndian.Uint32(data[8:12])
	documentCount := binary.LittleEndian.Uint32(data[12:16])
	totalTokens := binary.LittleEndian.Uint64(data[16:24])
	documentLengthsOffset := binary.LittleEndian.Uint32(data[24:28])
	termsOffset := binary.LittleEndian.Uint32(data[28:32])
	postingsOffset := binary.LittleEndian.Uint32(data[32:36])
	totalSize := binary.LittleEndian.Uint32(data[36:40])
	if documentLengthsOffset != ftsDictionaryHeaderSize || totalSize != uint32(len(data)) {
		return nil, fmt.Errorf("%w: inconsistent header lengths", ErrCorruptFTSDictionary)
	}
	expectedTermsOffset := uint64(ftsDictionaryHeaderSize) + uint64(documentCount)*4
	if uint64(termsOffset) != expectedTermsOffset || termsOffset > postingsOffset || postingsOffset > totalSize {
		return nil, fmt.Errorf("%w: invalid section offsets", ErrCorruptFTSDictionary)
	}
	// Four single-byte uvarints are the minimum possible descriptor (empty
	// suffix plus posting length and max-tf), so this bounds allocations before
	// parsing attacker-controlled counts.
	if uint64(termCount)*4 > uint64(postingsOffset-termsOffset) {
		return nil, fmt.Errorf("%w: term count exceeds descriptor section", ErrCorruptFTSDictionary)
	}
	for _, value := range data[44:60] {
		if value != 0 {
			return nil, fmt.Errorf("%w: reserved header bytes are nonzero", ErrCorruptFTSDictionary)
		}
	}
	if got, want := ailego.CRC32C(data[:60]), binary.LittleEndian.Uint32(data[60:64]); got != want {
		return nil, fmt.Errorf("%w: header CRC32C mismatch", ErrCorruptFTSDictionary)
	}
	if got, want := ailego.CRC32C(data[ftsDictionaryHeaderSize:]), binary.LittleEndian.Uint32(data[40:44]); got != want {
		return nil, fmt.Errorf("%w: payload CRC32C mismatch", ErrCorruptFTSDictionary)
	}
	data = append([]byte(nil), data...)
	dictionary := &FTSTermDictionary{
		terms:           make([]string, 0, termCount),
		postings:        make([]*FTSPostingList, 0, termCount),
		maximumTF:       make([]uint32, 0, termCount),
		documentLengths: make([]uint32, documentCount),
		stats: FTSSegmentStats{
			TotalDocuments: uint64(documentCount),
			TotalTokens:    totalTokens,
		},
		data: data,
	}
	var computedTokens uint64
	for index := range dictionary.documentLengths {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		offset := ftsDictionaryHeaderSize + index*4
		dictionary.documentLengths[index] = binary.LittleEndian.Uint32(data[offset : offset+4])
		if math.MaxUint64-computedTokens < uint64(dictionary.documentLengths[index]) {
			return nil, fmt.Errorf("%w: total tokens overflow", ErrCorruptFTSDictionary)
		}
		computedTokens += uint64(dictionary.documentLengths[index])
	}
	if computedTokens != totalTokens {
		return nil, fmt.Errorf("%w: total token count mismatch", ErrCorruptFTSDictionary)
	}

	type termRecord struct {
		postingLength uint32
		maximumTF     uint32
	}
	records := make([]termRecord, 0, termCount)
	cursor := int(termsOffset)
	previousTerm := ""
	for index := uint32(0); index < termCount; index++ {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		prefixLength, err := readFTSUvarint(data, &cursor, int(postingsOffset))
		if err != nil || prefixLength > uint64(len(previousTerm)) {
			return nil, fmt.Errorf("%w: invalid term prefix at %d", ErrCorruptFTSDictionary, index)
		}
		suffixLength, err := readFTSUvarint(data, &cursor, int(postingsOffset))
		if err != nil || suffixLength > uint64(int(postingsOffset)-cursor) {
			return nil, fmt.Errorf("%w: invalid term suffix at %d", ErrCorruptFTSDictionary, index)
		}
		term := previousTerm[:prefixLength] + string(data[cursor:cursor+int(suffixLength)])
		cursor += int(suffixLength)
		postingLength, err := readFTSUvarint(data, &cursor, int(postingsOffset))
		if err != nil || postingLength == 0 || postingLength > math.MaxUint32 {
			return nil, fmt.Errorf("%w: invalid posting length at %d", ErrCorruptFTSDictionary, index)
		}
		maximumTF, err := readFTSUvarint(data, &cursor, int(postingsOffset))
		if err != nil || maximumTF == 0 || maximumTF > math.MaxUint32 {
			return nil, fmt.Errorf("%w: invalid maximum term frequency at %d", ErrCorruptFTSDictionary, index)
		}
		if index > 0 && previousTerm >= term {
			return nil, fmt.Errorf("%w: terms are not strictly sorted", ErrCorruptFTSDictionary)
		}
		dictionary.terms = append(dictionary.terms, term)
		records = append(records, termRecord{uint32(postingLength), uint32(maximumTF)})
		previousTerm = term
	}
	if cursor != int(postingsOffset) {
		return nil, fmt.Errorf("%w: term table has trailing bytes", ErrCorruptFTSDictionary)
	}

	postingCursor := uint64(postingsOffset)
	for index, record := range records {
		postingEnd := postingCursor + uint64(record.postingLength)
		if postingEnd > uint64(len(data)) {
			return nil, fmt.Errorf("%w: posting %d is truncated", ErrCorruptFTSDictionary, index)
		}
		postingList, err := openFTSPostingList(ctx, data[postingCursor:postingEnd], false)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, fmt.Errorf("%w: term %q: %v", ErrCorruptFTSDictionary, dictionary.terms[index], err)
		}
		if postingList.DocumentFrequency() == 0 {
			return nil, fmt.Errorf("%w: term %q has an empty posting", ErrCorruptFTSDictionary, dictionary.terms[index])
		}
		var maximumTF uint32
		iterator := postingList.Iterator()
		for iterator.Next() {
			if iterator.globalIndex&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			documentID := iterator.DocumentID()
			if uint64(documentID) >= uint64(len(dictionary.documentLengths)) || iterator.DocumentLength() != dictionary.documentLengths[documentID] {
				return nil, fmt.Errorf("%w: term %q has inconsistent document metadata", ErrCorruptFTSDictionary, dictionary.terms[index])
			}
			maximumTF = max(maximumTF, iterator.TermFrequency())
		}
		if maximumTF != record.maximumTF {
			return nil, fmt.Errorf("%w: term %q maximum frequency mismatch", ErrCorruptFTSDictionary, dictionary.terms[index])
		}
		dictionary.postings = append(dictionary.postings, postingList)
		dictionary.maximumTF = append(dictionary.maximumTF, maximumTF)
		postingCursor = postingEnd
	}
	if postingCursor != uint64(len(data)) {
		return nil, fmt.Errorf("%w: posting section has trailing bytes", ErrCorruptFTSDictionary)
	}
	return dictionary, nil
}

func commonFTSPrefixLength(left, right string) int {
	limit := min(len(left), len(right))
	index := 0
	for index < limit && left[index] == right[index] {
		index++
	}
	return index
}

func readFTSUvarint(data []byte, cursor *int, limit int) (uint64, error) {
	if *cursor < 0 || *cursor >= limit {
		return 0, errors.New("truncated uvarint")
	}
	value, length := binary.Uvarint(data[*cursor:limit])
	if length <= 0 {
		return 0, errors.New("invalid uvarint")
	}
	*cursor += length
	return value, nil
}
