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
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestFTSBitPacking(t *testing.T) {
	if got := packFTSUint32([]uint32{1, 2, 3}, 2); !reflect.DeepEqual(got, []byte{0x39}) {
		t.Fatalf("packed example = %x", got)
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
			if got, want := uint64(len(packed)), ftsPackedByteSize(width, uint32(count)); got != want {
				t.Fatalf("width %d count %d size = %d, want %d", width, count, got, want)
			}
			if decoded := unpackFTSUint32(packed, width, uint32(count)); !reflect.DeepEqual(decoded, values) {
				t.Fatalf("width %d count %d round trip differs", width, count)
			}
		}
	}
	for _, test := range []struct {
		value uint32
		want  uint8
	}{{0, 0}, {1, 1}, {2, 2}, {3, 2}, {255, 8}, {256, 9}, {math.MaxUint32, 32}} {
		if got := ftsBitsNeeded(test.value); got != test.want {
			t.Fatalf("ftsBitsNeeded(%d) = %d, want %d", test.value, got, test.want)
		}
	}
}

func TestFTSPostingListEmpty(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if list.DocumentFrequency() != 0 || len(list.Bytes()) != ftsPostingHeaderSize || list.Iterator().Next() {
		t.Fatalf("unexpected empty list: %#v", list)
	}
	reopened, err := OpenFTSPostingList(context.Background(), list.Bytes())
	if err != nil || reopened.DocumentFrequency() != 0 {
		t.Fatalf("reopen = %#v, %v", reopened, err)
	}
}

func TestFTSPostingListBlockRoundTrip(t *testing.T) {
	postings := makeFTSPostingTestData(301)
	want := cloneFTSPostings(postings)
	list, err := BuildFTSPostingList(context.Background(), postings)
	if err != nil {
		t.Fatal(err)
	}
	if list.DocumentFrequency() != uint32(len(want)) || len(list.blocks) != 3 {
		t.Fatalf("list count/blocks = %d/%d", list.DocumentFrequency(), len(list.blocks))
	}
	postings[0].Positions[0] = 999
	postings[0].DocumentID = 999
	got := collectFTSPostings(list.Iterator())
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip differs\ngot  %#v\nwant %#v", got[:3], want[:3])
	}

	encoded := list.Bytes()
	reopened, err := OpenFTSPostingList(context.Background(), encoded)
	if err != nil {
		t.Fatal(err)
	}
	encoded[0] ^= 0xff
	if got := collectFTSPostings(reopened.Iterator()); !reflect.DeepEqual(got, want) {
		t.Fatal("reopened list aliases caller bytes")
	}
	if got := reopened.Bytes(); string(got[0:4]) != string(ftsPostingMagic[:]) {
		t.Fatal("Bytes did not return an independent copy")
	}
}

func TestFTSPostingIteratorAdvance(t *testing.T) {
	postings := makeFTSPostingTestData(301)
	list, err := BuildFTSPostingList(context.Background(), postings)
	if err != nil {
		t.Fatal(err)
	}
	iterator := list.Iterator()
	for _, target := range []uint32{0, 1, 100, 510, 511, 512, 900, math.MaxUint32} {
		wantIndex := sortFTSPostingIndex(postings, target)
		found := iterator.Advance(target)
		if wantIndex == len(postings) {
			if found {
				t.Fatalf("Advance(%d) found %d", target, iterator.DocumentID())
			}
			break
		}
		if !found || iterator.DocumentID() != postings[wantIndex].DocumentID {
			t.Fatalf("Advance(%d) = %v/%d, want %d", target, found, iterator.DocumentID(), postings[wantIndex].DocumentID)
		}
		if !iterator.Advance(target) || iterator.DocumentID() != postings[wantIndex].DocumentID {
			t.Fatalf("repeated Advance(%d) moved", target)
		}
	}

	iterator = list.Iterator()
	if !iterator.Next() || iterator.DocumentID() != postings[0].DocumentID || !iterator.Advance(postings[129].DocumentID) {
		t.Fatal("Next then Advance failed")
	}
	if iterator.DocumentID() != postings[129].DocumentID {
		t.Fatalf("advanced document = %d", iterator.DocumentID())
	}
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
	for index, postings := range tests {
		if list, err := BuildFTSPostingList(context.Background(), postings); list != nil || !errors.Is(err, ErrInvalidFTSPosting) {
			t.Fatalf("case %d = %#v, %v", index, list, err)
		}
	}
	if _, err := BuildFTSPostingList(nil, nil); !errors.Is(err, ErrInvalidFTSPosting) {
		t.Fatalf("nil context error = %v", err)
	}
	if _, err := OpenFTSPostingList(nil, nil); !errors.Is(err, ErrCorruptFTSPosting) {
		t.Fatalf("nil open context error = %v", err)
	}
}

func TestFTSPostingListCorruption(t *testing.T) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(150))
	if err != nil {
		t.Fatal(err)
	}
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
		"position varint": func(data []byte) []byte {
			data[len(data)-1] = 0x80
			repairFTSPostingCRCs(data)
			return data
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			data := mutate(append([]byte(nil), valid...))
			if opened, err := OpenFTSPostingList(context.Background(), data); opened != nil || !errors.Is(err, ErrCorruptFTSPosting) {
				t.Fatalf("Open = %#v, %v", opened, err)
			}
		})
	}
}

func TestFTSPostingListContextCancellation(t *testing.T) {
	postings := makeFTSPostingTestData(5000)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := BuildFTSPostingList(canceled, postings); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled build = %v", err)
	}
	midBuild := newCancelAfterChecks(4)
	if _, err := BuildFTSPostingList(midBuild, postings); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-build = %v", err)
	}
	list, err := BuildFTSPostingList(context.Background(), postings)
	if err != nil {
		t.Fatal(err)
	}
	midOpen := newCancelAfterChecks(4)
	if _, err := OpenFTSPostingList(midOpen, list.Bytes()); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-open = %v", err)
	}
}

func FuzzFTSPostingListOpen(f *testing.F) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(140))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(list.Bytes())
	f.Add([]byte{})
	f.Add([]byte("ZVFP"))
	f.Fuzz(func(t *testing.T, data []byte) {
		opened, err := OpenFTSPostingList(context.Background(), data)
		if err != nil {
			return
		}
		if opened.DocumentFrequency() > uint32(len(data)) {
			t.Fatalf("impossible count %d for %d bytes", opened.DocumentFrequency(), len(data))
		}
		iterator := opened.Iterator()
		var previous uint32
		first := true
		for iterator.Next() {
			if !first && iterator.DocumentID() <= previous {
				t.Fatal("document IDs are not increasing")
			}
			if uint32(len(iterator.Positions())) != iterator.TermFrequency() {
				t.Fatal("position count differs from term frequency")
			}
			previous = iterator.DocumentID()
			first = false
		}
	})
}

func BenchmarkFTSPostingIterator(b *testing.B) {
	list, err := BuildFTSPostingList(context.Background(), makeFTSPostingTestData(10000))
	if err != nil {
		b.Fatal(err)
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
			b.Fatal(count)
		}
	}
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
	binary.LittleEndian.PutUint32(data[32:36], ailego.CRC32C(data[ftsPostingHeaderSize:]))
	binary.LittleEndian.PutUint32(data[44:48], ailego.CRC32C(data[:44]))
}
