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
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"

	"github.com/gorse-io/xvec/internal/ailego/hash"
)

const (
	ftsPostingVersion         = uint16(1)
	ftsPostingDocumentsBlock  = uint16(128)
	ftsPostingHeaderSize      = 48
	ftsPostingDirectorySize   = 12
	ftsPostingBlockHeaderSize = 12
)

var (
	ftsPostingMagic = [4]byte{'Z', 'V', 'F', 'P'}

	// ErrInvalidFTSPosting identifies invalid posting input supplied by a caller.
	ErrInvalidFTSPosting = errors.New("core: invalid FTS posting list")
	// ErrCorruptFTSPosting identifies malformed or checksummed posting bytes.
	ErrCorruptFTSPosting = errors.New("core: corrupt FTS posting list")
)

// FTSPosting is one term occurrence summary in one document. Positions are
// ordered token positions and their count must equal TermFrequency.
type FTSPosting struct {
	DocumentID     uint32   `json:"document_id"`
	TermFrequency  uint32   `json:"term_frequency"`
	DocumentLength uint32   `json:"document_length"`
	Positions      []uint32 `json:"positions"`
}

type ftsPostingBlock struct {
	maxDocumentID  uint32
	blockOffset    uint32
	positionOffset uint32
}

// FTSPostingList is an immutable, checksummed sequence of block-bitpacked
// document IDs, term frequencies, document lengths, and delta-varint positions.
type FTSPostingList struct {
	data                   []byte
	count                  uint32
	positionsOffset        uint32
	blocks                 []ftsPostingBlock
	blockMaxNormalizations []float32
	blockMaxParams         BM25Params
	blockMaxTotalDocuments uint64
	blockMaxTotalTokens    uint64
}

// BuildFTSPostingList validates and encodes postings in strictly increasing
// document-ID order. The returned list owns its encoded bytes.
func BuildFTSPostingList(ctx context.Context, postings []FTSPosting) (*FTSPostingList, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSPosting)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(len(postings)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: %d entries exceed uint32", ErrInvalidFTSPosting, len(postings))
	}

	type encodedBlock struct {
		maxDocumentID uint32
		data          []byte
		positions     []byte
	}
	blockCount := (len(postings) + int(ftsPostingDocumentsBlock) - 1) / int(ftsPostingDocumentsBlock)
	blocks := make([]encodedBlock, 0, blockCount)
	work := 0
	for blockStart := 0; blockStart < len(postings); blockStart += int(ftsPostingDocumentsBlock) {
		blockEnd := min(blockStart+int(ftsPostingDocumentsBlock), len(postings))
		blockPostings := postings[blockStart:blockEnd]
		documentDeltas := make([]uint32, len(blockPostings))
		termFrequencies := make([]uint32, len(blockPostings))
		documentLengths := make([]uint32, len(blockPostings))
		positionLengths := make([]uint32, len(blockPostings))
		var blockPositions []byte
		for index, posting := range blockPostings {
			globalIndex := blockStart + index
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			work++
			if globalIndex > 0 && postings[globalIndex-1].DocumentID >= posting.DocumentID {
				return nil, fmt.Errorf("%w: document IDs are not strictly increasing at %d", ErrInvalidFTSPosting, globalIndex)
			}
			if posting.TermFrequency == 0 || uint64(posting.TermFrequency) != uint64(len(posting.Positions)) {
				return nil, fmt.Errorf("%w: posting %d has frequency %d and %d positions", ErrInvalidFTSPosting, globalIndex, posting.TermFrequency, len(posting.Positions))
			}
			if posting.DocumentLength < posting.TermFrequency {
				return nil, fmt.Errorf("%w: posting %d frequency exceeds document length", ErrInvalidFTSPosting, globalIndex)
			}
			for positionIndex := 1; positionIndex < len(posting.Positions); positionIndex++ {
				if work&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				work++
				if posting.Positions[positionIndex-1] > posting.Positions[positionIndex] {
					return nil, fmt.Errorf("%w: posting %d positions decrease at %d", ErrInvalidFTSPosting, globalIndex, positionIndex)
				}
			}
			if index > 0 {
				documentDeltas[index] = posting.DocumentID - blockPostings[index-1].DocumentID
			}
			termFrequencies[index] = posting.TermFrequency
			documentLengths[index] = posting.DocumentLength
			positionStart := len(blockPositions)
			var err error
			blockPositions, err = appendFTSPositionDeltas(ctx, blockPositions, posting.Positions)
			if err != nil {
				return nil, err
			}
			positionLength := len(blockPositions) - positionStart
			if uint64(positionLength) > math.MaxUint32 {
				return nil, fmt.Errorf("%w: posting %d position payload exceeds uint32", ErrInvalidFTSPosting, globalIndex)
			}
			positionLengths[index] = uint32(positionLength)
		}

		widthDocumentID := ftsBitsNeeded(maxUint32Slice(documentDeltas))
		widthTermFrequency := ftsBitsNeeded(maxUint32Slice(termFrequencies))
		widthDocumentLength := ftsBitsNeeded(maxUint32Slice(documentLengths))
		widthPositionLength := ftsBitsNeeded(maxUint32Slice(positionLengths))
		blockData := make([]byte, ftsPostingBlockHeaderSize)
		binary.LittleEndian.PutUint32(blockData[0:4], blockPostings[0].DocumentID)
		binary.LittleEndian.PutUint16(blockData[4:6], uint16(len(blockPostings)))
		blockData[6] = widthDocumentID
		blockData[7] = widthTermFrequency
		blockData[8] = widthDocumentLength
		blockData[9] = widthPositionLength
		blockData = append(blockData, packFTSUint32(documentDeltas, widthDocumentID)...)
		blockData = append(blockData, packFTSUint32(termFrequencies, widthTermFrequency)...)
		blockData = append(blockData, packFTSUint32(documentLengths, widthDocumentLength)...)
		blockData = append(blockData, packFTSUint32(positionLengths, widthPositionLength)...)
		blocks = append(blocks, encodedBlock{
			maxDocumentID: blockPostings[len(blockPostings)-1].DocumentID,
			data:          blockData,
			positions:     blockPositions,
		})
	}

	directoryBytes := uint64(len(blocks)) * ftsPostingDirectorySize
	if directoryBytes > math.MaxUint32-ftsPostingHeaderSize || uint64(ftsPostingHeaderSize)+directoryBytes > ftsMaximumInt() {
		return nil, fmt.Errorf("%w: directory exceeds uint32", ErrInvalidFTSPosting)
	}
	output := make([]byte, ftsPostingHeaderSize+int(directoryBytes))
	blockOffsets := make([]uint32, len(blocks))
	for index, block := range blocks {
		newSize := uint64(len(output)) + uint64(len(block.data))
		if newSize > math.MaxUint32 || newSize > ftsMaximumInt() {
			return nil, fmt.Errorf("%w: encoded blocks exceed uint32", ErrInvalidFTSPosting)
		}
		blockOffsets[index] = uint32(len(output))
		output = append(output, block.data...)
	}
	positionsOffset := uint32(len(output))
	positionOffsets := make([]uint32, len(blocks))
	var positionsLength uint64
	for index, block := range blocks {
		positionOffsets[index] = uint32(positionsLength)
		positionsLength += uint64(len(block.positions))
		newSize := uint64(len(output)) + uint64(len(block.positions))
		if newSize > math.MaxUint32 || newSize > ftsMaximumInt() {
			return nil, fmt.Errorf("%w: encoded positions exceed uint32", ErrInvalidFTSPosting)
		}
		output = append(output, block.positions...)
	}

	for index, block := range blocks {
		offset := ftsPostingHeaderSize + index*ftsPostingDirectorySize
		binary.LittleEndian.PutUint32(output[offset:offset+4], block.maxDocumentID)
		binary.LittleEndian.PutUint32(output[offset+4:offset+8], blockOffsets[index])
		binary.LittleEndian.PutUint32(output[offset+8:offset+12], positionOffsets[index])
	}
	copy(output[0:4], ftsPostingMagic[:])
	binary.LittleEndian.PutUint16(output[4:6], ftsPostingVersion)
	binary.LittleEndian.PutUint16(output[6:8], ftsPostingDocumentsBlock)
	binary.LittleEndian.PutUint32(output[8:12], uint32(len(postings)))
	binary.LittleEndian.PutUint32(output[12:16], uint32(len(blocks)))
	binary.LittleEndian.PutUint32(output[16:20], ftsPostingHeaderSize)
	binary.LittleEndian.PutUint32(output[20:24], uint32(ftsPostingHeaderSize)+uint32(directoryBytes))
	binary.LittleEndian.PutUint32(output[24:28], positionsOffset)
	binary.LittleEndian.PutUint32(output[28:32], uint32(len(output)))
	binary.LittleEndian.PutUint32(output[32:36], hashutil.CRC32C(output[ftsPostingHeaderSize:]))
	binary.LittleEndian.PutUint32(output[44:48], hashutil.CRC32C(output[:44]))
	return openFTSPostingList(ctx, output, false)
}

// OpenFTSPostingList validates encoded native-Go posting bytes and retains an
// owned copy so later caller mutation cannot affect iteration.
func OpenFTSPostingList(ctx context.Context, data []byte) (*FTSPostingList, error) {
	return openFTSPostingList(ctx, data, true)
}

func openFTSPostingList(ctx context.Context, data []byte, clone bool) (*FTSPostingList, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrCorruptFTSPosting)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) < ftsPostingHeaderSize || uint64(len(data)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: header is truncated", ErrCorruptFTSPosting)
	}
	if string(data[0:4]) != string(ftsPostingMagic[:]) {
		return nil, fmt.Errorf("%w: bad magic", ErrCorruptFTSPosting)
	}
	if version := binary.LittleEndian.Uint16(data[4:6]); version != ftsPostingVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrCorruptFTSPosting, version)
	}
	if blockSize := binary.LittleEndian.Uint16(data[6:8]); blockSize != ftsPostingDocumentsBlock {
		return nil, fmt.Errorf("%w: unsupported block size %d", ErrCorruptFTSPosting, blockSize)
	}
	count := binary.LittleEndian.Uint32(data[8:12])
	blockCount := binary.LittleEndian.Uint32(data[12:16])
	directoryOffset := binary.LittleEndian.Uint32(data[16:20])
	blocksOffset := binary.LittleEndian.Uint32(data[20:24])
	positionsOffset := binary.LittleEndian.Uint32(data[24:28])
	totalSize := binary.LittleEndian.Uint32(data[28:32])
	if totalSize != uint32(len(data)) || directoryOffset != ftsPostingHeaderSize {
		return nil, fmt.Errorf("%w: inconsistent header lengths", ErrCorruptFTSPosting)
	}
	wantBlocks := uint32(0)
	if count > 0 {
		wantBlocks = (count-1)/uint32(ftsPostingDocumentsBlock) + 1
	}
	if blockCount != wantBlocks {
		return nil, fmt.Errorf("%w: block count %d does not cover %d documents", ErrCorruptFTSPosting, blockCount, count)
	}
	directorySize := uint64(blockCount) * ftsPostingDirectorySize
	if uint64(blocksOffset) != uint64(ftsPostingHeaderSize)+directorySize || blocksOffset > positionsOffset || positionsOffset > totalSize {
		return nil, fmt.Errorf("%w: invalid section offsets", ErrCorruptFTSPosting)
	}
	if data[36] != 0 || data[37] != 0 || data[38] != 0 || data[39] != 0 || data[40] != 0 || data[41] != 0 || data[42] != 0 || data[43] != 0 {
		return nil, fmt.Errorf("%w: reserved header bytes are nonzero", ErrCorruptFTSPosting)
	}
	if got, want := hashutil.CRC32C(data[:44]), binary.LittleEndian.Uint32(data[44:48]); got != want {
		return nil, fmt.Errorf("%w: header CRC32C mismatch", ErrCorruptFTSPosting)
	}
	if got, want := hashutil.CRC32C(data[ftsPostingHeaderSize:]), binary.LittleEndian.Uint32(data[32:36]); got != want {
		return nil, fmt.Errorf("%w: payload CRC32C mismatch", ErrCorruptFTSPosting)
	}
	if clone {
		data = append([]byte(nil), data...)
	}
	list := &FTSPostingList{
		data:            data,
		count:           count,
		positionsOffset: positionsOffset,
		blocks:          make([]ftsPostingBlock, blockCount),
	}
	for index := range list.blocks {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		offset := ftsPostingHeaderSize + index*ftsPostingDirectorySize
		list.blocks[index] = ftsPostingBlock{
			maxDocumentID:  binary.LittleEndian.Uint32(data[offset : offset+4]),
			blockOffset:    binary.LittleEndian.Uint32(data[offset+4 : offset+8]),
			positionOffset: binary.LittleEndian.Uint32(data[offset+8 : offset+12]),
		}
		if index > 0 && list.blocks[index-1].maxDocumentID >= list.blocks[index].maxDocumentID {
			return nil, fmt.Errorf("%w: directory document IDs are not increasing", ErrCorruptFTSPosting)
		}
	}
	if err := list.validate(ctx, blocksOffset, positionsOffset); err != nil {
		return nil, err
	}
	return list, nil
}

func (l *FTSPostingList) validate(ctx context.Context, blocksOffset, positionsOffset uint32) error {
	expectedBlockOffset := blocksOffset
	expectedPositionOffset := uint32(0)
	decodedCount := uint64(0)
	var previousDocumentID uint32
	havePrevious := false
	for blockIndex, metadata := range l.blocks {
		if blockIndex&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if metadata.blockOffset != expectedBlockOffset || metadata.positionOffset != expectedPositionOffset {
			return fmt.Errorf("%w: non-contiguous block %d", ErrCorruptFTSPosting, blockIndex)
		}
		if uint64(metadata.blockOffset)+ftsPostingBlockHeaderSize > uint64(positionsOffset) {
			return fmt.Errorf("%w: block %d header is truncated", ErrCorruptFTSPosting, blockIndex)
		}
		header := l.data[metadata.blockOffset : metadata.blockOffset+ftsPostingBlockHeaderSize]
		minimumDocumentID := binary.LittleEndian.Uint32(header[0:4])
		blockCount := binary.LittleEndian.Uint16(header[4:6])
		if blockCount == 0 || blockCount > ftsPostingDocumentsBlock || blockIndex+1 < len(l.blocks) && blockCount != ftsPostingDocumentsBlock {
			return fmt.Errorf("%w: invalid document count in block %d", ErrCorruptFTSPosting, blockIndex)
		}
		if header[10] != 0 || header[11] != 0 {
			return fmt.Errorf("%w: block %d reserved bytes are nonzero", ErrCorruptFTSPosting, blockIndex)
		}
		widths := [4]uint8{header[6], header[7], header[8], header[9]}
		for _, width := range widths {
			if width > 32 {
				return fmt.Errorf("%w: block %d has bit width %d", ErrCorruptFTSPosting, blockIndex, width)
			}
		}
		cursor := uint64(metadata.blockOffset) + ftsPostingBlockHeaderSize
		arrays := make([][]uint32, 4)
		for arrayIndex, width := range widths {
			length := ftsPackedByteSize(width, uint32(blockCount))
			if cursor+length > uint64(positionsOffset) {
				return fmt.Errorf("%w: block %d packed data is truncated", ErrCorruptFTSPosting, blockIndex)
			}
			arrays[arrayIndex] = unpackFTSUint32(l.data[cursor:cursor+length], width, uint32(blockCount))
			cursor += length
		}
		expectedBlockOffset = uint32(cursor)
		documentDeltas, termFrequencies := arrays[0], arrays[1]
		documentLengths, positionLengths := arrays[2], arrays[3]
		if documentDeltas[0] != 0 {
			return fmt.Errorf("%w: block %d first document delta is nonzero", ErrCorruptFTSPosting, blockIndex)
		}
		documentID := minimumDocumentID
		if havePrevious && previousDocumentID >= documentID {
			return fmt.Errorf("%w: block %d overlaps its predecessor", ErrCorruptFTSPosting, blockIndex)
		}
		for index := range documentDeltas {
			if index > 0 {
				if documentDeltas[index] == 0 || math.MaxUint32-documentID < documentDeltas[index] {
					return fmt.Errorf("%w: block %d has an invalid document delta", ErrCorruptFTSPosting, blockIndex)
				}
				documentID += documentDeltas[index]
			}
			if termFrequencies[index] == 0 || documentLengths[index] < termFrequencies[index] || positionLengths[index] == 0 || termFrequencies[index] > positionLengths[index] {
				return fmt.Errorf("%w: block %d has invalid inline payloads", ErrCorruptFTSPosting, blockIndex)
			}
			positionStart := uint64(l.positionsOffset) + uint64(expectedPositionOffset)
			positionEnd := positionStart + uint64(positionLengths[index])
			if positionEnd > uint64(len(l.data)) {
				return fmt.Errorf("%w: block %d position payload is truncated", ErrCorruptFTSPosting, blockIndex)
			}
			if _, err := decodeFTSPositionDeltas(ctx, l.data[positionStart:positionEnd], termFrequencies[index]); err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					return ctxErr
				}
				return fmt.Errorf("%w: block %d: %v", ErrCorruptFTSPosting, blockIndex, err)
			}
			expectedPositionOffset += positionLengths[index]
		}
		if documentID != metadata.maxDocumentID {
			return fmt.Errorf("%w: block %d maximum document mismatch", ErrCorruptFTSPosting, blockIndex)
		}
		previousDocumentID = documentID
		havePrevious = true
		decodedCount += uint64(blockCount)
	}
	if expectedBlockOffset != positionsOffset || uint64(l.positionsOffset)+uint64(expectedPositionOffset) != uint64(len(l.data)) || decodedCount != uint64(l.count) {
		return fmt.Errorf("%w: section lengths do not match their contents", ErrCorruptFTSPosting)
	}
	return nil
}

// Bytes returns an owned copy of the native encoded posting list.
func (l *FTSPostingList) Bytes() []byte {
	if l == nil {
		return nil
	}
	return append([]byte(nil), l.data...)
}

// DocumentFrequency returns the number of documents in the list.
func (l *FTSPostingList) DocumentFrequency() uint32 {
	if l == nil {
		return 0
	}
	return l.count
}

func (l *FTSPostingList) prepareBlockMaxNormalizations(ctx context.Context, scorer *BM25Scorer, documentLengths []uint32) (uint32, error) {
	if ctx == nil || l == nil || scorer == nil {
		return 0, fmt.Errorf("%w: posting list, scorer, and context are required", ErrInvalidFTSPosting)
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	maximums := make([]float32, len(l.blocks))
	var maximumTF uint32
	iterator := l.scoringIterator()
	work := 0
	for iterator.Next() {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		work++
		documentID := iterator.DocumentID()
		if documentLengths != nil && (uint64(documentID) >= uint64(len(documentLengths)) || iterator.DocumentLength() != documentLengths[documentID]) {
			return 0, fmt.Errorf("%w: posting has inconsistent document metadata", ErrInvalidFTSPosting)
		}
		maximumTF = max(maximumTF, iterator.TermFrequency())
		normalization := scorer.termNormalization(iterator.TermFrequency(), iterator.DocumentLength())
		maximums[iterator.blockIndex] = max(maximums[iterator.blockIndex], normalization)
	}
	l.blockMaxNormalizations = maximums
	l.blockMaxParams = scorer.params
	l.blockMaxTotalDocuments = scorer.stats.TotalDocuments
	l.blockMaxTotalTokens = scorer.stats.TotalTokens
	return maximumTF, nil
}

func (l *FTSPostingList) cachedBlockMaxNormalizations(scorer *BM25Scorer) []float32 {
	if l == nil || scorer == nil || len(l.blockMaxNormalizations) != len(l.blocks) ||
		l.blockMaxParams != scorer.params || l.blockMaxTotalDocuments != scorer.stats.TotalDocuments ||
		l.blockMaxTotalTokens != scorer.stats.TotalTokens {
		return nil
	}
	return l.blockMaxNormalizations
}

// Iterator returns a new independent iterator positioned before the first
// posting.
func (l *FTSPostingList) Iterator() *FTSPostingIterator {
	return l.iterator(true)
}

func (l *FTSPostingList) scoringIterator() *FTSPostingIterator {
	return l.iterator(false)
}

func (l *FTSPostingList) iterator(decodePositions bool) *FTSPostingIterator {
	return &FTSPostingIterator{list: l, blockIndex: -1, indexInBlock: -1, decodePositions: decodePositions}
}

// FTSPostingIterator decodes one 128-document block at a time and supports
// skip-directory seeks.
type FTSPostingIterator struct {
	list            *FTSPostingList
	blockIndex      int
	indexInBlock    int
	globalIndex     uint32
	valid           bool
	decodePositions bool
	documentIDs     []uint32
	termFrequencies []uint32
	documentLengths []uint32
	positionOffsets []uint32
	positionLengths []uint32
}

// Next moves to the next posting and reports whether one exists.
func (i *FTSPostingIterator) Next() bool {
	if i == nil || i.list == nil || i.list.count == 0 {
		return false
	}
	if !i.valid {
		if i.blockIndex >= len(i.list.blocks) {
			return false
		}
		if i.blockIndex < 0 {
			i.loadBlock(0)
			i.indexInBlock = 0
			i.globalIndex = 0
			i.valid = true
			return true
		}
	}
	if i.indexInBlock+1 < len(i.documentIDs) {
		i.indexInBlock++
		i.globalIndex++
		i.valid = true
		return true
	}
	nextBlock := i.blockIndex + 1
	if nextBlock >= len(i.list.blocks) {
		i.valid = false
		i.blockIndex = len(i.list.blocks)
		return false
	}
	i.loadBlock(nextBlock)
	i.indexInBlock = 0
	i.globalIndex++
	i.valid = true
	return true
}

// Advance moves to the first posting with DocumentID >= target. If the
// current posting already satisfies target it remains current.
func (i *FTSPostingIterator) Advance(target uint32) bool {
	if i == nil || i.list == nil || i.list.count == 0 {
		return false
	}
	if i.valid && i.DocumentID() >= target {
		return true
	}
	startBlock := 0
	startIndex := 0
	if i.valid {
		startBlock = i.blockIndex
		startIndex = i.indexInBlock + 1
		if i.list.blocks[startBlock].maxDocumentID < target {
			startBlock++
			startIndex = 0
		}
	} else if i.blockIndex >= len(i.list.blocks) {
		return false
	}
	if startBlock >= len(i.list.blocks) {
		i.valid = false
		i.blockIndex = len(i.list.blocks)
		return false
	}
	targetBlock := searchFTSPostingBlock(i.list.blocks, startBlock, target)
	if targetBlock == len(i.list.blocks) {
		i.valid = false
		i.blockIndex = len(i.list.blocks)
		return false
	}
	if targetBlock != i.blockIndex {
		i.loadBlock(targetBlock)
		startIndex = 0
	}
	position := searchFTSDocumentID(i.documentIDs, startIndex, target)
	if position >= len(i.documentIDs) {
		i.valid = false
		return i.Advance(target)
	}
	i.indexInBlock = position
	i.globalIndex = uint32(targetBlock)*uint32(ftsPostingDocumentsBlock) + uint32(position)
	i.valid = true
	return true
}

func searchFTSPostingBlock(blocks []ftsPostingBlock, start int, target uint32) int {
	low, high := start, len(blocks)
	for low < high {
		middle := int(uint(low+high) >> 1)
		if blocks[middle].maxDocumentID < target {
			low = middle + 1
		} else {
			high = middle
		}
	}
	return low
}

func searchFTSDocumentID(documentIDs []uint32, start int, target uint32) int {
	low, high := start, len(documentIDs)
	for low < high {
		middle := int(uint(low+high) >> 1)
		if documentIDs[middle] < target {
			low = middle + 1
		} else {
			high = middle
		}
	}
	return low
}

func (i *FTSPostingIterator) loadBlock(blockIndex int) {
	metadata := i.list.blocks[blockIndex]
	header := i.list.data[metadata.blockOffset : metadata.blockOffset+ftsPostingBlockHeaderSize]
	minimumDocumentID := binary.LittleEndian.Uint32(header[0:4])
	count := binary.LittleEndian.Uint16(header[4:6])
	widths := [4]uint8{header[6], header[7], header[8], header[9]}
	cursor := uint64(metadata.blockOffset) + ftsPostingBlockHeaderSize
	countInt := int(count)
	i.documentIDs = resizeFTSUint32(i.documentIDs, countInt)
	documentIDBytes := ftsPackedByteSize(widths[0], uint32(count))
	unpackFTSUint32Into(i.list.data[cursor:cursor+documentIDBytes], widths[0], i.documentIDs)
	cursor += documentIDBytes
	cursor += ftsPackedByteSize(widths[1], uint32(count))
	cursor += ftsPackedByteSize(widths[2], uint32(count))
	i.termFrequencies = i.termFrequencies[:0]
	i.documentLengths = i.documentLengths[:0]
	if i.decodePositions {
		i.positionLengths = resizeFTSUint32(i.positionLengths, countInt)
		i.positionOffsets = resizeFTSUint32(i.positionOffsets, countInt)
		length := ftsPackedByteSize(widths[3], uint32(count))
		unpackFTSUint32Into(i.list.data[cursor:cursor+length], widths[3], i.positionLengths)
	} else {
		i.positionLengths = nil
		i.positionOffsets = nil
	}
	documentID := minimumDocumentID
	for index := range i.documentIDs {
		if index > 0 {
			documentID += i.documentIDs[index]
		}
		i.documentIDs[index] = documentID
	}
	if i.decodePositions {
		positionOffset := metadata.positionOffset
		for index := range i.positionOffsets {
			i.positionOffsets[index] = positionOffset
			positionOffset += i.positionLengths[index]
		}
	}
	i.blockIndex = blockIndex
	i.indexInBlock = -1
	i.valid = false
}

func (i *FTSPostingIterator) decodeTermFrequencies() {
	if i == nil || len(i.termFrequencies) == len(i.documentIDs) {
		return
	}
	count := len(i.documentIDs)
	i.termFrequencies = resizeFTSUint32(i.termFrequencies, count)
	offset, width := i.packedBlockArray(1)
	length := ftsPackedByteSize(width, uint32(count))
	unpackFTSUint32Into(i.list.data[offset:offset+length], width, i.termFrequencies)
}

func (i *FTSPostingIterator) decodeDocumentLengths() {
	if i == nil || len(i.documentLengths) == len(i.documentIDs) {
		return
	}
	count := len(i.documentIDs)
	i.documentLengths = resizeFTSUint32(i.documentLengths, count)
	offset, width := i.packedBlockArray(2)
	length := ftsPackedByteSize(width, uint32(count))
	unpackFTSUint32Into(i.list.data[offset:offset+length], width, i.documentLengths)
}

func (i *FTSPostingIterator) packedBlockArray(arrayIndex int) (uint64, uint8) {
	metadata := i.list.blocks[i.blockIndex]
	header := i.list.data[metadata.blockOffset : metadata.blockOffset+ftsPostingBlockHeaderSize]
	count := uint32(binary.LittleEndian.Uint16(header[4:6]))
	offset := uint64(metadata.blockOffset) + ftsPostingBlockHeaderSize
	for index := 0; index < arrayIndex; index++ {
		offset += ftsPackedByteSize(header[6+index], count)
	}
	return offset, header[6+arrayIndex]
}

// Valid reports whether the iterator addresses a posting.
func (i *FTSPostingIterator) Valid() bool { return i != nil && i.valid }

// DocumentID returns the current document ID, or zero when invalid.
func (i *FTSPostingIterator) DocumentID() uint32 {
	if !i.Valid() {
		return 0
	}
	return i.documentIDs[i.indexInBlock]
}

// TermFrequency returns the current term frequency, or zero when invalid.
func (i *FTSPostingIterator) TermFrequency() uint32 {
	if !i.Valid() {
		return 0
	}
	i.decodeTermFrequencies()
	return i.termFrequencies[i.indexInBlock]
}

// DocumentLength returns the current document token count, or zero when
// invalid.
func (i *FTSPostingIterator) DocumentLength() uint32 {
	if !i.Valid() {
		return 0
	}
	i.decodeDocumentLengths()
	return i.documentLengths[i.indexInBlock]
}

// Positions decodes and returns an owned copy of the current position list.
func (i *FTSPostingIterator) Positions() []uint32 {
	if !i.Valid() || !i.decodePositions {
		return nil
	}
	start := uint64(i.list.positionsOffset) + uint64(i.positionOffsets[i.indexInBlock])
	end := start + uint64(i.positionLengths[i.indexInBlock])
	positions, _ := decodeFTSPositionDeltas(context.Background(), i.list.data[start:end], i.TermFrequency())
	return positions
}

// Posting returns an owned representation of the current posting.
func (i *FTSPostingIterator) Posting() (FTSPosting, bool) {
	if !i.Valid() {
		return FTSPosting{}, false
	}
	return FTSPosting{
		DocumentID:     i.DocumentID(),
		TermFrequency:  i.TermFrequency(),
		DocumentLength: i.DocumentLength(),
		Positions:      i.Positions(),
	}, true
}

func appendFTSPositionDeltas(ctx context.Context, destination []byte, positions []uint32) ([]byte, error) {
	previous := uint32(0)
	for index, position := range positions {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		destination = binary.AppendUvarint(destination, uint64(position-previous))
		previous = position
	}
	return destination, nil
}

func decodeFTSPositionDeltas(ctx context.Context, data []byte, count uint32) ([]uint32, error) {
	if uint64(count) > uint64(len(data)) {
		return nil, errors.New("position count exceeds payload length")
	}
	positions := make([]uint32, count)
	cursor := 0
	previous := uint32(0)
	for index := range positions {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		value, length := binary.Uvarint(data[cursor:])
		if length <= 0 || value > math.MaxUint32 || math.MaxUint32-previous < uint32(value) {
			return nil, errors.New("invalid delta-varint positions")
		}
		cursor += length
		previous += uint32(value)
		positions[index] = previous
	}
	if cursor != len(data) {
		return nil, errors.New("position payload has trailing bytes")
	}
	return positions, nil
}

func ftsBitsNeeded(value uint32) uint8 { return uint8(bits.Len32(value)) }

func ftsPackedByteSize(width uint8, count uint32) uint64 {
	return (uint64(width)*uint64(count) + 7) / 8
}

func packFTSUint32(values []uint32, width uint8) []byte {
	output := make([]byte, ftsPackedByteSize(width, uint32(len(values))))
	if width == 0 {
		return output
	}
	bitOffset := uint64(0)
	for _, value := range values {
		remaining := width
		for remaining > 0 {
			byteIndex := bitOffset >> 3
			shift := uint8(bitOffset & 7)
			take := min(remaining, 8-shift)
			mask := uint32(1<<take) - 1
			output[byteIndex] |= byte(value&mask) << shift
			value >>= take
			remaining -= take
			bitOffset += uint64(take)
		}
	}
	return output
}

func unpackFTSUint32(data []byte, width uint8, count uint32) []uint32 {
	output := make([]uint32, count)
	unpackFTSUint32Into(data, width, output)
	return output
}

func unpackFTSUint32Into(data []byte, width uint8, output []uint32) {
	clear(output)
	if width == 0 {
		return
	}
	bitOffset := uint64(0)
	for index := range output {
		remaining := width
		shiftOut := uint8(0)
		for remaining > 0 {
			byteIndex := bitOffset >> 3
			shiftIn := uint8(bitOffset & 7)
			take := min(remaining, 8-shiftIn)
			mask := byte(0xff >> (8 - take))
			output[index] |= uint32((data[byteIndex]>>shiftIn)&mask) << shiftOut
			remaining -= take
			shiftOut += take
			bitOffset += uint64(take)
		}
	}
}

func resizeFTSUint32(values []uint32, count int) []uint32 {
	if cap(values) < count {
		return make([]uint32, count)
	}
	return values[:count]
}

func maxUint32Slice(values []uint32) uint32 {
	var maximum uint32
	for _, value := range values {
		maximum = max(maximum, value)
	}
	return maximum
}

func ftsMaximumInt() uint64 { return uint64(^uint(0) >> 1) }
