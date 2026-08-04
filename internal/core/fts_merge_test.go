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
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestMergeFTSTermDictionariesDenseRemapPositionsAndReopen(t *testing.T) {
	segment0 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "apple", Position: 0}, {Text: "banana", Position: 1}},
		nil,
		{{Text: "apple", Position: 0}, {Text: "apple", Position: 1}},
	})
	segment1 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "banana", Position: 0}, {Text: "carrot", Position: 1}},
		{{Text: "apple", Position: 0}, {Text: "banana", Position: 1}},
	})
	deleted0 := ailego.NewBitmap(3)
	deleted0.Set(1)
	deleted1 := ailego.NewBitmap(2)
	deleted1.Set(0)
	before0, err := segment0.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	before1, err := segment1.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{
		{Dictionary: segment0, DeletedDocuments: deleted0},
		{Dictionary: segment1, DeletedDocuments: deleted1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := merged.Stats(), (FTSSegmentStats{TotalDocuments: 3, TotalTokens: 6}); got != want {
		t.Fatalf("stats = %#v, want %#v", got, want)
	}
	if got, want := merged.Terms(), []string{"apple", "banana"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("terms = %q, want %q", got, want)
	}
	for documentID := uint32(0); documentID < 3; documentID++ {
		if length, ok := merged.DocumentLength(documentID); !ok || length != 2 {
			t.Fatalf("document %d length = %d, %v", documentID, length, ok)
		}
	}
	wantPostings := map[string][]FTSPosting{
		"apple": {
			{DocumentID: 0, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{0}},
			{DocumentID: 1, TermFrequency: 2, DocumentLength: 2, Positions: []uint32{0, 1}},
			{DocumentID: 2, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{0}},
		},
		"banana": {
			{DocumentID: 0, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{1}},
			{DocumentID: 2, TermFrequency: 1, DocumentLength: 2, Positions: []uint32{1}},
		},
	}
	for term, want := range wantPostings {
		info, postings, found := merged.Lookup(term)
		if !found || info.DocumentFrequency != uint32(len(want)) || !reflect.DeepEqual(collectFTSPostings(postings.Iterator()), want) {
			t.Fatalf("%q = %#v, %#v, %v; want %#v", term, info, collectFTSPostings(postings.Iterator()), found, want)
		}
	}
	if info, _, found := merged.Lookup("carrot"); found || info != (FTSTermInfo{}) {
		t.Fatalf("deleted-only term survived: %#v", info)
	}
	appleInfo, _, _ := merged.Lookup("apple")
	if appleInfo.MaximumTermFrequency != 2 {
		t.Fatalf("apple max tf = %d", appleInfo.MaximumTermFrequency)
	}

	pipeline := newFTSStandardTestPipeline(t)
	node, err := ParseFTSQuery(context.Background(), `"apple banana"`, pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	iterator, err := NewFTSQueryIterator(context.Background(), merged, node, FTSQueryExecutionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := collectFTSQueryDocuments(t, iterator), []uint32{0, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("phrase IDs = %v, want %v", got, want)
	}

	encoded, err := merged.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenFTSTermDictionary(context.Background(), encoded)
	if err != nil {
		t.Fatal(err)
	}
	assertFTSDictionariesEqual(t, reopened, merged)
	after0, _ := segment0.Encode(context.Background())
	after1, _ := segment1.Encode(context.Background())
	if !reflect.DeepEqual(before0, after0) || !reflect.DeepEqual(before1, after1) {
		t.Fatal("merge mutated a source dictionary")
	}
	deleted0.Clear(1)
	deleted1.Clear(0)
	assertFTSDictionariesEqual(t, reopened, merged)
}

func TestMergeFTSTermDictionariesEmptyAllDeletedAndMaximumTF(t *testing.T) {
	empty, err := MergeFTSTermDictionaries(context.Background(), nil)
	if err != nil || empty.Stats() != (FTSSegmentStats{}) || empty.TermCount() != 0 {
		t.Fatalf("empty merge = %#v, %v", empty, err)
	}
	encoded, err := empty.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if reopened, err := OpenFTSTermDictionary(context.Background(), encoded); err != nil || reopened.Stats() != (FTSSegmentStats{}) {
		t.Fatalf("reopen empty = %#v, %v", reopened, err)
	}

	source := buildFTSTestDictionary(t, [][]Token{
		{{Text: "x", Position: 0}, {Text: "x", Position: 1}, {Text: "x", Position: 2}},
		{{Text: "x", Position: 0}},
	})
	deleteHighest := ailego.NewBitmap(2)
	deleteHighest.Set(0)
	merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source, DeletedDocuments: deleteHighest}})
	if err != nil {
		t.Fatal(err)
	}
	info, postings, found := merged.Lookup("x")
	if !found || info.MaximumTermFrequency != 1 || !reflect.DeepEqual(collectFTSPostings(postings.Iterator()), []FTSPosting{
		{DocumentID: 0, TermFrequency: 1, DocumentLength: 1, Positions: []uint32{0}},
	}) {
		t.Fatalf("max-tf merge = %#v, %#v, %v", info, collectFTSPostings(postings.Iterator()), found)
	}
	deleteAll := ailego.NewBitmap(2)
	deleteAll.Set(0)
	deleteAll.Set(1)
	allDeleted, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source, DeletedDocuments: deleteAll}})
	if err != nil || allDeleted.Stats() != (FTSSegmentStats{}) || allDeleted.TermCount() != 0 {
		t.Fatalf("all-deleted merge = %#v, %v", allDeleted, err)
	}
}

func TestMergeFTSTermDictionariesValidationCancellationAndConcurrency(t *testing.T) {
	source := buildFTSTestDictionary(t, [][]Token{{{Text: "x", Position: 0}}})
	if merged, err := MergeFTSTermDictionaries(nil, nil); merged != nil || !errors.Is(err, ErrInvalidFTSMerge) {
		t.Fatalf("nil context = %#v, %v", merged, err)
	}
	if merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{}}); merged != nil || !errors.Is(err, ErrInvalidFTSMerge) {
		t.Fatalf("nil dictionary = %#v, %v", merged, err)
	}
	inconsistent := buildFTSTestDictionary(t, [][]Token{{{Text: "x", Position: 0}}})
	inconsistent.stats.TotalTokens++
	if merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: inconsistent}}); merged != nil || !errors.Is(err, ErrInvalidFTSMerge) {
		t.Fatalf("inconsistent dictionary = %#v, %v", merged, err)
	}
	outside := ailego.NewBitmap(1)
	outside.Set(1)
	if merged, err := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source, DeletedDocuments: outside}}); merged != nil || !errors.Is(err, ErrInvalidFTSMerge) {
		t.Fatalf("outside deletion = %#v, %v", merged, err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if merged, err := MergeFTSTermDictionaries(canceled, []FTSSegmentView{{Dictionary: source}}); merged != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled merge = %#v, %v", merged, err)
	}
	largeDocuments := make([][]Token, 5000)
	for index := range largeDocuments {
		largeDocuments[index] = []Token{{Text: "x", Position: 0}}
	}
	large := buildFTSTestDictionary(t, largeDocuments)
	if merged, err := MergeFTSTermDictionaries(newCancelAfterChecks(3), []FTSSegmentView{{Dictionary: large}}); merged != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-merge cancellation = %#v, %v", merged, err)
	}

	var wait sync.WaitGroup
	errorsChannel := make(chan error, 12)
	for range 12 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			merged, mergeErr := MergeFTSTermDictionaries(context.Background(), []FTSSegmentView{{Dictionary: source}})
			if mergeErr != nil {
				errorsChannel <- mergeErr
				return
			}
			if info, _, found := merged.Lookup("x"); !found || info.DocumentFrequency != 1 {
				errorsChannel <- errors.New("unexpected concurrent merge")
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		t.Error(err)
	}
}

func FuzzMergeFTSTermDictionaries(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5}, uint64(0b001001))
	f.Add([]byte{}, uint64(0))
	f.Fuzz(func(t *testing.T, data []byte, deletionMask uint64) {
		if len(data) > 32 {
			data = data[:32]
		}
		documents := make([][]Token, len(data))
		for documentID, value := range data {
			tokenCount := int(value % 4)
			documents[documentID] = make([]Token, tokenCount)
			for position := range tokenCount {
				documents[documentID][position] = Token{
					Text: string(rune('a' + (int(value)+position)%4)), Position: uint32(position),
				}
			}
		}
		split := len(documents) / 2
		parts := [][][]Token{documents[:split], documents[split:]}
		views := make([]FTSSegmentView, len(parts))
		survivors := make([][]Token, 0, len(documents))
		global := 0
		for partIndex, part := range parts {
			views[partIndex].Dictionary = buildFTSTestDictionary(t, part)
			deleted := ailego.NewBitmap(uint64(len(part)))
			for local := range part {
				isDeleted := global < 64 && deletionMask&(uint64(1)<<global) != 0
				if isDeleted {
					deleted.Set(uint64(local))
				} else {
					survivors = append(survivors, part[local])
				}
				global++
			}
			views[partIndex].DeletedDocuments = deleted
		}
		merged, err := MergeFTSTermDictionaries(context.Background(), views)
		if err != nil {
			t.Fatal(err)
		}
		want := buildFTSTestDictionary(t, survivors)
		assertFTSDictionariesEqual(t, merged, want)
	})
}

func BenchmarkMergeFTSTermDictionaries(b *testing.B) {
	documents := make([][]Token, 5000)
	for documentID := range documents {
		documents[documentID] = []Token{
			{Text: "common", Position: 0},
			{Text: string(rune('a' + documentID%26)), Position: 1},
		}
	}
	left := buildFTSTestDictionary(b, documents[:2500])
	right := buildFTSTestDictionary(b, documents[2500:])
	views := []FTSSegmentView{{Dictionary: left}, {Dictionary: right}}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := MergeFTSTermDictionaries(context.Background(), views); err != nil {
			b.Fatal(err)
		}
	}
}

func assertFTSDictionariesEqual(t testing.TB, got, want *FTSTermDictionary) {
	t.Helper()
	if got == nil || want == nil {
		if got != want {
			t.Fatalf("dictionary nil mismatch: got %#v, want %#v", got, want)
		}
		return
	}
	if got.Stats() != want.Stats() || !slices.Equal(got.Terms(), want.Terms()) || !slices.Equal(got.documentLengths, want.documentLengths) {
		t.Fatalf("dictionary metadata = %#v/%q/%v, want %#v/%q/%v", got.Stats(), got.Terms(), got.documentLengths, want.Stats(), want.Terms(), want.documentLengths)
	}
	for _, term := range want.Terms() {
		gotInfo, gotPostings, gotFound := got.Lookup(term)
		wantInfo, wantPostings, wantFound := want.Lookup(term)
		if gotFound != wantFound || gotInfo != wantInfo ||
			!reflect.DeepEqual(collectFTSPostings(gotPostings.Iterator()), collectFTSPostings(wantPostings.Iterator())) {
			t.Fatalf("term %q mismatch: %#v/%#v/%v, want %#v/%#v/%v", term, gotInfo, collectFTSPostings(gotPostings.Iterator()), gotFound, wantInfo, collectFTSPostings(wantPostings.Iterator()), wantFound)
		}
	}
}
