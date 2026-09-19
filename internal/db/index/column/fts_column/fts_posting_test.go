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
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/parquet-go/bitpack"
	"github.com/stretchr/testify/require"
)

func TestFTSBitPacking(t *testing.T) {
	{
		got := packFTSUint32([]uint32{1, 2, 3}, 2)
		require.Equal(t, []byte{0x39}, got)
	}

	for width := uint8(0); width <= 32; width++ {
		for _, count := range []int{0, 1, 3, 31, 32, 127, 128} {
			values := make([]uint32, count)
			var mask uint32
			if width == 32 {
				mask = math.MaxUint32
			} else if width > 0 {
				mask = uint32(1)<<width - 1
			}
			for index := range values {
				values[index] = rand.Uint32() & mask
			}
			packed := packFTSUint32(values, width)
			{
				got, want := uint64(len(packed)), ftsPackedByteSize(width, uint32(count))
				require.Equal(t, want, got)
			}
			{
				decoded := unpackFTSUint32(packed, width, uint32(count))
				require.Equal(t, values, decoded)
			}
		}
	}
	for _, test := range []struct {
		value uint32
		want  uint8
	}{{0, 0}, {1, 1}, {2, 2}, {3, 2}, {255, 8}, {256, 9}, {math.MaxUint32, 32}} {
		{
			got := ftsBitsNeeded(test.value)
			require.Equal(t, test.want, got)
		}
	}
}

func TestFTSPostingListEmpty(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), nil)
	require.NoError(t, err)
	require.True(t, list.DocumentFrequency() == 0)
	require.Len(t, list.Bytes(), ftsPostingHeaderSize)
	require.False(t, list.Iterator().Next())

	reopened, err := OpenFTSPostingList(context.Background(), list.Bytes())
	require.NoError(t, err)
	require.True(t, reopened.DocumentFrequency() == 0)
}

func TestFTSPostingListBlockRoundTrip(t *testing.T) {
	postings := makeFTSPostingTestData(301)
	want := cloneFTSPostings(postings)
	list, err := BuildFTSPostingList(context.Background(), postings)
	require.NoError(t, err)
	require.Equal(t, uint32(len(want)), list.DocumentFrequency())
	require.Len(t, list.blocks, 3)

	postings[0].Positions[0] = 999
	postings[0].DocumentID = 999
	got := collectFTSPostings(list.Iterator())
	require.Equal(t, want, got)

	encoded := list.Bytes()
	reopened, err := OpenFTSPostingList(context.Background(), encoded)
	require.NoError(t, err)

	encoded[0] ^= 0xff
	{
		got := collectFTSPostings(reopened.Iterator())
		require.Equal(t, want, got,
			"reopened list aliases caller bytes")
	}
	{
		got := reopened.Bytes()
		require.Equal(t, string(ftsPostingMagic[:]), string(got[0:4]),
			"Bytes did not return an independent copy")
	}
}

func TestFTSPostingListSIMDFullBlockLayout(t *testing.T) {
	postings := makeFTSPostingTestData(int(ftsPostingDocumentsBlock) + 1)
	list, err := BuildFTSPostingList(context.Background(), postings)
	require.NoError(t, err)

	data := list.Bytes()
	require.Equal(t, uint16(2), binary.LittleEndian.Uint16(data[4:6]))

	firstBlock := list.blocks[0]
	header := data[firstBlock.blockOffset : firstBlock.blockOffset+ftsPostingBlockHeaderSize]
	require.Equal(t, ftsPostingDocumentsBlock, binary.LittleEndian.Uint16(header[4:6]))

	wantArrays := makeFTSPostingBlockArrays(postings[:ftsPostingDocumentsBlock])
	cursor := uint64(firstBlock.blockOffset) + ftsPostingBlockHeaderSize
	for arrayIndex, want := range wantArrays {
		width := header[6+arrayIndex]
		packedSize := ftsPackedByteSize(width, uint32(len(want)))
		end := cursor + packedSize
		paddingEnd := end + bitpack.PaddingInt32
		require.LessOrEqual(t, paddingEnd, uint64(len(data)))
		require.Equal(t, make([]byte, bitpack.PaddingInt32), data[end:paddingEnd])

		got := make([]uint32, len(want))
		bitpack.Unpack(got, data[cursor:paddingEnd], uint(width))
		require.Equal(t, want, got)
		cursor = paddingEnd
	}
	require.Equal(t, uint64(list.blocks[1].blockOffset), cursor)

	tailHeader := data[list.blocks[1].blockOffset : list.blocks[1].blockOffset+ftsPostingBlockHeaderSize]
	tailEnd := uint64(list.blocks[1].blockOffset) + ftsPostingBlockHeaderSize
	for arrayIndex := range wantArrays {
		tailEnd += ftsPackedByteSize(tailHeader[6+arrayIndex], 1)
	}
	require.Equal(t, uint64(list.positionsOffset), tailEnd,
		"partial posting blocks must retain the compact scalar layout")
}

func TestFTSPostingListRejectsPreviousVersion(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(1))
	require.NoError(t, err)

	data := list.Bytes()
	binary.LittleEndian.PutUint16(data[4:6], 1)
	repairFTSPostingCRCs(data)

	opened, err := OpenFTSPostingList(context.Background(), data)
	require.Nil(t, opened)
	require.ErrorIs(t, err, ErrCorruptFTSPosting)
}

func TestFTSPostingIteratorAdvance(t *testing.T) {
	postings := makeFTSPostingTestData(301)
	list, err := BuildFTSPostingList(context.Background(), postings)
	require.NoError(t, err)

	iterator := list.Iterator()
	for _, target := range []uint32{0, 1, 100, 510, 511, 512, 900, math.MaxUint32} {
		wantIndex := sortFTSPostingIndex(postings, target)
		found := iterator.Advance(target)
		if wantIndex == len(postings) {
			require.False(t, found)

			break
		}
		require.True(t, found)
		require.Equal(t, postings[wantIndex].DocumentID, iterator.DocumentID())
		require.True(t, iterator.Advance(target))
		require.Equal(t, postings[wantIndex].DocumentID, iterator.DocumentID())
	}

	iterator = list.Iterator()
	require.True(t, iterator.Next(),
		"Next then Advance failed")
	require.Equal(t, postings[0].DocumentID, iterator.DocumentID(),
		"Next then Advance failed")
	require.True(t, iterator.Advance(postings[129].DocumentID),
		"Next then Advance failed")
	require.Equal(t, postings[129].DocumentID, iterator.DocumentID())
}

func TestFTSPostingListInvalidInput(t *testing.T) {
	valid := FTSPosting{DocumentID: 1, TermFrequency: 1, DocumentLength: 1, Positions: []uint32{0}}
	tests := [][]FTSPosting{
		{{DocumentID: 1, TermFrequency: 0, DocumentLength: 1}},
		{{DocumentID: 1, TermFrequency: 2, DocumentLength: 2, Positions: []uint32{0}}},
		{{DocumentID: 1, TermFrequency: 2, DocumentLength: 1, Positions: []uint32{0, 1}}},
		{{DocumentID: 1, TermFrequency: 2, DocumentLength: 2, Positions: []uint32{2, 1}}},
		{valid, valid},
		{valid, {DocumentID: 0, TermFrequency: 1, DocumentLength: 1, Positions: []uint32{0}}},
	}
	for _, postings := range tests {
		{
			list, err := BuildFTSPostingList(context.Background(), postings)
			require.Nil(t, list)
			require.ErrorIs(t, err, ErrInvalidFTSPosting)
		}
	}
	{
		_, err := BuildFTSPostingList(nil, nil)
		require.ErrorIs(t, err, ErrInvalidFTSPosting)
	}
	{
		_, err := OpenFTSPostingList(nil, nil)
		require.ErrorIs(t, err, ErrCorruptFTSPosting)
	}
}

func TestFTSPostingListCorruption(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(150))
	require.NoError(t, err)

	valid := list.Bytes()
	tests := map[string]func([]byte) []byte{
		"truncated": func(data []byte) []byte { return data[:20] },
		"magic": func(data []byte) []byte {
			data[0] ^= 1
			return data
		},
		"version": func(data []byte) []byte {
			binary.LittleEndian.PutUint16(data[4:6], 99)
			repairFTSPostingCRCs(data)
			return data
		},
		"header crc": func(data []byte) []byte {
			data[44] ^= 1
			return data
		},
		"payload crc": func(data []byte) []byte {
			data[len(data)-1] ^= 1
			return data
		},
		"block count": func(data []byte) []byte {
			binary.LittleEndian.PutUint32(data[12:16], 1)
			repairFTSPostingCRCs(data)
			return data
		},
		"directory offset": func(data []byte) []byte {
			binary.LittleEndian.PutUint32(data[ftsPostingHeaderSize+4:ftsPostingHeaderSize+8], binary.LittleEndian.Uint32(data[20:24])+1)
			repairFTSPostingCRCs(data)
			return data
		},
		"bit width": func(data []byte) []byte {
			blockOffset := binary.LittleEndian.Uint32(data[20:24])
			data[blockOffset+6] = 33
			repairFTSPostingCRCs(data)
			return data
		},
		"SIMD padding": func(data []byte) []byte {
			blockOffset := binary.LittleEndian.Uint32(data[20:24])
			width := data[blockOffset+6]
			paddingOffset := uint64(blockOffset) + ftsPostingBlockHeaderSize + ftsPackedByteSize(width, uint32(ftsPostingDocumentsBlock))
			data[paddingOffset] = 1
			repairFTSPostingCRCs(data)
			return data
		},
		"position varint": func(data []byte) []byte {
			data[len(data)-1] = 0x80
			repairFTSPostingCRCs(data)
			return data
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			data := mutate(append([]byte(nil), valid...))
			{
				opened, err := OpenFTSPostingList(context.Background(), data)
				require.Nil(t, opened)
				require.ErrorIs(t, err, ErrCorruptFTSPosting)
			}
		})
	}
}

func TestFTSPostingListContextCancellation(t *testing.T) {
	postings := makeFTSPostingTestData(5000)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := BuildFTSPostingList(canceled, postings)
		require.ErrorIs(t, err, context.Canceled)
	}

	midBuild := newCancelAfterChecks(4)
	{
		_, err := BuildFTSPostingList(midBuild, postings)
		require.ErrorIs(t, err, context.Canceled)
	}

	list, err := BuildFTSPostingList(context.Background(), postings)
	require.NoError(t, err)

	midOpen := newCancelAfterChecks(4)
	{
		_, err := OpenFTSPostingList(midOpen, list.Bytes())
		require.ErrorIs(t, err, context.Canceled)
	}
}

func FuzzFTSPostingListOpen(f *testing.F) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(140))
	require.NoError(f, err)

	f.Add(list.Bytes())
	f.Add([]byte{})
	f.Add([]byte("ZVFP"))
	f.Fuzz(func(t *testing.T, data []byte) {
		opened, err := OpenFTSPostingList(context.Background(), data)
		if err != nil {
			return
		}
		require.True(t, opened.DocumentFrequency() <= uint32(len(data)))

		iterator := opened.Iterator()
		var previous uint32
		first := true
		for iterator.Next() {
			require.False(t, !first && iterator.DocumentID() <= previous,
				"document IDs are not increasing")
			require.Equal(t, iterator.TermFrequency(), uint32(len(iterator.Positions())),
				"position count differs from term frequency")

			previous = iterator.DocumentID()
			first = false
		}
	})
}

func BenchmarkFTSPostingIterator(b *testing.B) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(10000))
	if err != nil {
		require.NoError(b, err)
	}

	b.SetBytes(int64(len(list.data)))
	b.ReportAllocs()
	for b.Loop() {
		iterator := list.Iterator()
		var count int
		for iterator.Next() {
			count++
		}
		if count != 10000 {
			require.Equal(b, 10000, count)
		}
	}
}

func TestFTSPostingIteratorReusesBlockBuffers(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(300))
	require.NoError(t, err)
	iterator := list.Iterator()
	iterator.loadBlock(0)
	iterator.decodeTermFrequencies()
	iterator.decodeDocumentLengths()
	documentIDs := &iterator.documentIDs[0]
	termFrequencies := &iterator.termFrequencies[0]
	documentLengths := &iterator.documentLengths[0]
	positionOffsets := &iterator.positionOffsets[0]
	positionLengths := &iterator.positionLengths[0]

	iterator.loadBlock(1)
	iterator.decodeTermFrequencies()
	iterator.decodeDocumentLengths()
	require.Same(t, documentIDs, &iterator.documentIDs[0])
	require.Same(t, termFrequencies, &iterator.termFrequencies[0])
	require.Same(t, documentLengths, &iterator.documentLengths[0])
	require.Same(t, positionOffsets, &iterator.positionOffsets[0])
	require.Same(t, positionLengths, &iterator.positionLengths[0])
}

func TestFTSPostingIteratorDecodesScoringPayloadLazily(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(300))
	require.NoError(t, err)
	iterator := list.scoringIterator()
	require.True(t, iterator.Next())
	require.Empty(t, iterator.termFrequencies)
	require.Empty(t, iterator.documentLengths)

	require.NotZero(t, iterator.TermFrequency())
	require.NotEmpty(t, iterator.termFrequencies)
	require.Empty(t, iterator.documentLengths)
	require.NotZero(t, iterator.DocumentLength())
	require.NotEmpty(t, iterator.documentLengths)

	require.True(t, iterator.Advance(list.blocks[0].maxDocumentID+1))
	require.Empty(t, iterator.termFrequencies)
	require.Empty(t, iterator.documentLengths)
}

func TestFTSScoringIteratorSkipsPositionPayload(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(300))
	require.NoError(t, err)
	iterator := list.scoringIterator()
	require.True(t, iterator.Next())
	require.NotZero(t, iterator.TermFrequency())
	require.NotZero(t, iterator.DocumentLength())
	require.Nil(t, iterator.positionOffsets)
	require.Nil(t, iterator.positionLengths)
	require.Nil(t, iterator.Positions())
	require.True(t, iterator.Advance(500))
	require.Nil(t, iterator.positionOffsets)
	require.Nil(t, iterator.positionLengths)
}

func makeFTSPostingTestData(count int) []FTSPosting {
	postings := make([]FTSPosting, count)
	documentID := uint32(0)
	for index := range postings {
		documentID += uint32(index%7 + 1)
		frequency := uint32(index%5 + 1)
		positions := make([]uint32, frequency)
		position := uint32(index % 3)
		for positionIndex := range positions {
			position += uint32(positionIndex*130 + 1)
			positions[positionIndex] = position
		}
		postings[index] = FTSPosting{
			DocumentID:     documentID,
			TermFrequency:  frequency,
			DocumentLength: frequency + uint32(index%19),
			Positions:      positions,
		}
	}
	return postings
}

func makeFTSPostingBlockArrays(postings []FTSPosting) [4][]uint32 {
	var arrays [4][]uint32
	for index, posting := range postings {
		if index == 0 {
			arrays[0] = append(arrays[0], 0)
		} else {
			arrays[0] = append(arrays[0], posting.DocumentID-postings[index-1].DocumentID)
		}
		arrays[1] = append(arrays[1], posting.TermFrequency)
		arrays[2] = append(arrays[2], posting.DocumentLength)

		encoded, err := appendFTSPositionDeltas(context.Background(), nil, posting.Positions)
		if err != nil {
			panic(err)
		}
		arrays[3] = append(arrays[3], uint32(len(encoded)))
	}
	return arrays
}

func cloneFTSPostings(postings []FTSPosting) []FTSPosting {
	result := append([]FTSPosting(nil), postings...)
	for index := range result {
		result[index].Positions = append([]uint32(nil), result[index].Positions...)
	}
	return result
}

func collectFTSPostings(iterator *FTSPostingIterator) []FTSPosting {
	var result []FTSPosting
	for iterator.Next() {
		posting, _ := iterator.Posting()
		result = append(result, posting)
	}
	return result
}

func sortFTSPostingIndex(postings []FTSPosting, target uint32) int {
	low, high := 0, len(postings)
	for low < high {
		middle := low + (high-low)/2
		if postings[middle].DocumentID < target {
			low = middle + 1
		} else {
			high = middle
		}
	}
	return low
}

func repairFTSPostingCRCs(data []byte) {
	if len(data) < ftsPostingHeaderSize {
		return
	}
	binary.LittleEndian.PutUint32(data[32:36], hashutil.CRC32C(data[ftsPostingHeaderSize:]))
	binary.LittleEndian.PutUint32(data[44:48], hashutil.CRC32C(data[:44]))
}
