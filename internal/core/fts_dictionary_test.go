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
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

type ftsPostingFixture struct {
	BaselineCommit      string `json:"baseline_commit"`
	PostingHeaderSHA256 string `json:"posting_header_sha256"`
	PostingSourceSHA256 string `json:"posting_source_sha256"`
	IndexerHeaderSHA256 string `json:"indexer_header_sha256"`
	IndexerSourceSHA256 string `json:"indexer_source_sha256"`
	ReducerHeaderSHA256 string `json:"reducer_header_sha256"`
	ReducerSourceSHA256 string `json:"reducer_source_sha256"`
	PhraseSourceSHA256  string `json:"phrase_source_sha256"`
	PositionDeltaHex    string `json:"position_delta_hex"`
	TotalDocuments      uint64 `json:"total_documents"`
	TotalTokens         uint64 `json:"total_tokens"`
	Documents           []struct {
		DocumentID uint32 `json:"document_id"`
		Tokens     []struct {
			Term     string `json:"term"`
			Position uint32 `json:"position"`
		} `json:"tokens"`
	} `json:"documents"`
	Terms []struct {
		Term                 string       `json:"term"`
		DocumentFrequency    uint32       `json:"document_frequency"`
		MaximumTermFrequency uint32       `json:"maximum_term_frequency"`
		Postings             []FTSPosting `json:"postings"`
	} `json:"terms"`
}

func loadFTSPostingFixture(t testing.TB) ftsPostingFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/fts_posting_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture ftsPostingFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func TestFTSTermDictionaryBaselineFixture(t *testing.T) {
	fixture := loadFTSPostingFixture(t)
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.PostingHeaderSHA256 != "78b8e7e6af6ae7279eb7561fc4aec53b2f7811a209f3e32306f89eb9430f92b5" ||
		fixture.PostingSourceSHA256 != "ed99e2c626429926afc6633c1f4d3d6ae89ac32bb584860306b215302756180c" ||
		fixture.IndexerHeaderSHA256 != "83baa255dad8f86e18a9c49091bcbcf015d5c066d75e726332eb2dddf7a31056" ||
		fixture.IndexerSourceSHA256 != "dbe3bffb8fef7d15a4be09babd5c7d3706dcdde4343d1318a310ba24662d5cab" ||
		fixture.ReducerHeaderSHA256 != "b87f60888e230f39268dea6614d7aadca60f8945bc3bc0862f2fa026d1aeb43b" ||
		fixture.ReducerSourceSHA256 != "7bebfd9d410c598e2970c0d3b7331b9623e2b481b28466b2eae7543af7d58b2e" ||
		fixture.PhraseSourceSHA256 != "6316f87dab229ba02fbb588f0047f23a9b988aab584d0d87fb9987896dd565f7" {
		t.Fatalf("unexpected fixture identity: %#v", fixture)
	}
	positions, err := appendFTSPositionDeltas(context.Background(), nil, []uint32{0, 2, 130})
	if err != nil {
		t.Fatal(err)
	}
	if got := hex.EncodeToString(positions); got != fixture.PositionDeltaHex {
		t.Fatalf("position encoding = %s, want %s", got, fixture.PositionDeltaHex)
	}

	builder := NewFTSFieldBuilder()
	for _, document := range fixture.Documents {
		tokens := make([]Token, len(document.Tokens))
		for index, token := range document.Tokens {
			tokens[index] = Token{Text: token.Term, Position: token.Position}
		}
		if err := builder.AddDocument(context.Background(), document.DocumentID, tokens); err != nil {
			t.Fatal(err)
		}
	}
	dictionary, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := dictionary.Stats(), (FTSSegmentStats{TotalDocuments: fixture.TotalDocuments, TotalTokens: fixture.TotalTokens}); got != want {
		t.Fatalf("stats = %#v, want %#v", got, want)
	}
	if dictionary.Stats().AverageDocumentLength() != float64(5)/3 {
		t.Fatalf("average length = %v", dictionary.Stats().AverageDocumentLength())
	}
	if dictionary.TermCount() != len(fixture.Terms) {
		t.Fatalf("term count = %d", dictionary.TermCount())
	}
	for _, term := range fixture.Terms {
		info, postingList, found := dictionary.Lookup(term.Term)
		if !found || info != (FTSTermInfo{Term: term.Term, DocumentFrequency: term.DocumentFrequency, MaximumTermFrequency: term.MaximumTermFrequency}) {
			t.Fatalf("Lookup(%q) = %#v, %v", term.Term, info, found)
		}
		if got := collectFTSPostings(postingList.Iterator()); !reflect.DeepEqual(got, term.Postings) {
			t.Fatalf("postings for %q = %#v, want %#v", term.Term, got, term.Postings)
		}
	}
	if _, _, found := dictionary.Lookup("missing"); found {
		t.Fatal("missing term found")
	}
}

func TestFTSTermDictionaryPrefixAndSnapshot(t *testing.T) {
	builder := NewFTSFieldBuilder()
	if err := builder.AddDocument(context.Background(), 0, []Token{{Text: "banana", Position: 0}, {Text: "band", Position: 1}, {Text: "apple", Position: 2}}); err != nil {
		t.Fatal(err)
	}
	first, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := first.Terms(), []string{"apple", "banana", "band"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("terms = %#v, want %#v", got, want)
	}
	terms := first.Terms()
	terms[0] = "changed"
	if first.Terms()[0] != "apple" {
		t.Fatal("Terms aliases dictionary state")
	}
	if got := first.Prefix("ban", 0); !reflect.DeepEqual(got, []FTSTermInfo{
		{Term: "banana", DocumentFrequency: 1, MaximumTermFrequency: 1},
		{Term: "band", DocumentFrequency: 1, MaximumTermFrequency: 1},
	}) {
		t.Fatalf("prefix = %#v", got)
	}
	if got := first.Prefix("ban", 1); len(got) != 1 || got[0].Term != "banana" {
		t.Fatalf("limited prefix = %#v", got)
	}
	if got := first.Prefix("", -1); len(got) != 0 {
		t.Fatalf("negative limit = %#v", got)
	}
	if err := builder.AddDocument(context.Background(), 1, []Token{{Text: "apricot", Position: 0}}); err != nil {
		t.Fatal(err)
	}
	second, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if first.Stats().TotalDocuments != 1 || second.Stats().TotalDocuments != 2 || first.TermCount() != 3 || second.TermCount() != 4 {
		t.Fatalf("snapshot stats/terms = %#v/%d %#v/%d", first.Stats(), first.TermCount(), second.Stats(), second.TermCount())
	}
}

func TestFTSTermDictionaryEncodeReopen(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "", Position: 0}, {Text: "alpha", Position: 1}, {Text: "alpha", Position: 2}},
		nil,
		{{Text: "alphabet", Position: 0}, {Text: string([]byte{0xff, 'x'}), Position: 1}},
	})
	encoded, err := dictionary.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	encodedAgain, err := dictionary.Encode(context.Background())
	if err != nil || !reflect.DeepEqual(encodedAgain, encoded) {
		t.Fatal("encoding is not deterministic")
	}
	reopened, err := OpenFTSTermDictionary(context.Background(), encoded)
	if err != nil {
		t.Fatal(err)
	}
	encoded[0] ^= 0xff
	if reopened.TermCount() != dictionary.TermCount() || !reflect.DeepEqual(reopened.Terms(), dictionary.Terms()) || reopened.Stats() != dictionary.Stats() {
		t.Fatal("reopened dictionary differs")
	}
	for _, term := range dictionary.Terms() {
		wantInfo, wantList, _ := dictionary.Lookup(term)
		gotInfo, gotList, found := reopened.Lookup(term)
		if !found || gotInfo != wantInfo || !reflect.DeepEqual(collectFTSPostings(gotList.Iterator()), collectFTSPostings(wantList.Iterator())) {
			t.Fatalf("reopened term %x differs", term)
		}
	}
	reencoded, err := reopened.Encode(context.Background())
	if err != nil || !reflect.DeepEqual(reencoded, encodedAgain) {
		t.Fatal("reopened encoding is not byte-identical")
	}
	if length, found := reopened.DocumentLength(1); !found || length != 0 {
		t.Fatalf("empty document length = %d, %v", length, found)
	}
	if _, found := reopened.DocumentLength(3); found {
		t.Fatal("out-of-range document length found")
	}
}

func TestFTSTermDictionaryAllEmptyDocuments(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{nil, nil})
	if dictionary.TermCount() != 0 || dictionary.Stats() != (FTSSegmentStats{TotalDocuments: 2}) {
		t.Fatalf("empty-doc dictionary = %d terms, %#v", dictionary.TermCount(), dictionary.Stats())
	}
	encoded, err := dictionary.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenFTSTermDictionary(context.Background(), encoded)
	if err != nil || reopened.TermCount() != 0 || reopened.Stats() != dictionary.Stats() {
		t.Fatalf("reopen = %#v, %v", reopened, err)
	}
}

func TestFTSFieldBuilderInvalidInputAndCancellation(t *testing.T) {
	if err := (*FTSFieldBuilder)(nil).AddDocument(context.Background(), 0, nil); !errors.Is(err, ErrInvalidFTSDocument) {
		t.Fatalf("nil builder error = %v", err)
	}
	builder := NewFTSFieldBuilder()
	if err := builder.AddDocument(nil, 0, nil); !errors.Is(err, ErrInvalidFTSDocument) {
		t.Fatalf("nil context error = %v", err)
	}
	if err := builder.AddDocument(context.Background(), 1, nil); !errors.Is(err, ErrInvalidFTSDocument) {
		t.Fatalf("sparse document error = %v", err)
	}
	if err := builder.AddDocument(context.Background(), 0, []Token{{Text: "a", Position: 2}, {Text: "b", Position: 1}}); !errors.Is(err, ErrInvalidFTSDocument) {
		t.Fatalf("decreasing position error = %v", err)
	}
	if err := builder.AddDocument(context.Background(), 0, nil); err != nil {
		t.Fatalf("builder mutated after failure: %v", err)
	}
	if _, err := builder.Build(nil); !errors.Is(err, ErrInvalidFTSDictionary) {
		t.Fatalf("nil build context error = %v", err)
	}
	if _, err := (*FTSFieldBuilder)(nil).Build(context.Background()); !errors.Is(err, ErrInvalidFTSDictionary) {
		t.Fatalf("nil build error = %v", err)
	}

	tokens := make([]Token, 20000)
	for index := range tokens {
		tokens[index] = Token{Text: fmt.Sprintf("term-%05d", index), Position: uint32(index)}
	}
	midAdd := newCancelAfterChecks(3)
	if err := NewFTSFieldBuilder().AddDocument(midAdd, 0, tokens); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-add error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := NewFTSFieldBuilder().AddDocument(canceled, 0, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled add = %v", err)
	}
	buildBuilder := NewFTSFieldBuilder()
	if err := buildBuilder.AddDocument(context.Background(), 0, []Token{{Text: "alpha", Position: 0}}); err != nil {
		t.Fatal(err)
	}
	if _, err := buildBuilder.Build(newCancelAfterChecks(4)); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-build error = %v", err)
	}
}

func TestFTSTermDictionaryEncodeOpenCancellation(t *testing.T) {
	tokens := make([]Token, 4200)
	for index := range tokens {
		tokens[index] = Token{Text: fmt.Sprintf("term-%05d", index), Position: uint32(index)}
	}
	dictionary := buildFTSTestDictionary(t, [][]Token{tokens})
	if _, err := dictionary.Encode(newCancelAfterChecks(3)); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-encode error = %v", err)
	}
	encoded, err := dictionary.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := OpenFTSTermDictionary(newCancelAfterChecks(3), encoded); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-open error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := OpenFTSTermDictionary(canceled, encoded); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled open error = %v", err)
	}
}

func TestFTSTermDictionaryCorruption(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "alpha", Position: 0}, {Text: "alphabet", Position: 1}},
		{{Text: "alpha", Position: 0}},
	})
	valid, err := dictionary.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]func([]byte) []byte{
		"truncated": func(data []byte) []byte { return data[:20] },
		"magic": func(data []byte) []byte {
			data[0] ^= 1
			return data
		},
		"version": func(data []byte) []byte {
			binary.LittleEndian.PutUint16(data[4:6], 99)
			repairFTSDictionaryCRCs(data)
			return data
		},
		"header crc": func(data []byte) []byte {
			data[60] ^= 1
			return data
		},
		"payload crc": func(data []byte) []byte {
			data[len(data)-1] ^= 1
			return data
		},
		"reserved": func(data []byte) []byte {
			data[44] = 1
			repairFTSDictionaryCRCs(data)
			return data
		},
		"impossible term count": func(data []byte) []byte {
			binary.LittleEndian.PutUint32(data[8:12], ^uint32(0))
			repairFTSDictionaryCRCs(data)
			return data
		},
		"token total": func(data []byte) []byte {
			binary.LittleEndian.PutUint32(data[ftsDictionaryHeaderSize:ftsDictionaryHeaderSize+4], 99)
			repairFTSDictionaryCRCs(data)
			return data
		},
		"first prefix": func(data []byte) []byte {
			termsOffset := binary.LittleEndian.Uint32(data[28:32])
			data[termsOffset] = 1
			repairFTSDictionaryCRCs(data)
			return data
		},
		"nested posting": func(data []byte) []byte {
			postingsOffset := binary.LittleEndian.Uint32(data[32:36])
			data[postingsOffset] ^= 1
			repairFTSDictionaryCRCs(data)
			return data
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			data := mutate(append([]byte(nil), valid...))
			if opened, err := OpenFTSTermDictionary(context.Background(), data); opened != nil || !errors.Is(err, ErrCorruptFTSDictionary) {
				t.Fatalf("Open = %#v, %v", opened, err)
			}
		})
	}
	if _, err := OpenFTSTermDictionary(nil, nil); !errors.Is(err, ErrCorruptFTSDictionary) {
		t.Fatalf("nil context error = %v", err)
	}
}

func TestFTSTermDictionaryConcurrentUse(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "alpha", Position: 0}, {Text: "beta", Position: 1}, {Text: "alpha", Position: 2}},
		{{Text: "beta", Position: 0}},
	})
	want, err := dictionary.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for worker := 0; worker < 32; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 50; iteration++ {
				_, posting, found := dictionary.Lookup("alpha")
				if !found || len(collectFTSPostings(posting.Iterator())) != 1 {
					errorsChannel <- errors.New("lookup differs")
					return
				}
				encoded, err := dictionary.Encode(context.Background())
				if err != nil || !reflect.DeepEqual(encoded, want) {
					errorsChannel <- errors.New("encoding differs")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		t.Fatal(err)
	}
}

func FuzzFTSTermDictionaryOpen(f *testing.F) {
	dictionary := buildFTSTestDictionary(f, [][]Token{
		{{Text: "alpha", Position: 0}, {Text: "beta", Position: 1}},
		{{Text: "alpha", Position: 0}},
	})
	encoded, err := dictionary.Encode(context.Background())
	if err != nil {
		f.Fatal(err)
	}
	f.Add(encoded)
	f.Add([]byte{})
	f.Add([]byte("ZVFD"))
	f.Fuzz(func(t *testing.T, data []byte) {
		dictionary, err := OpenFTSTermDictionary(context.Background(), data)
		if err != nil {
			return
		}
		if uint64(dictionary.TermCount()) > uint64(len(data)) || dictionary.Stats().TotalDocuments > uint64(len(data))/4 {
			t.Fatalf("impossible counts for %d bytes", len(data))
		}
		if terms := dictionary.Terms(); !sortStringsStrict(terms) {
			t.Fatal("terms are not sorted")
		}
	})
}

func BenchmarkFTSTermDictionaryBuild(b *testing.B) {
	documents := make([][]Token, 1000)
	for documentID := range documents {
		documents[documentID] = make([]Token, 20)
		for position := range documents[documentID] {
			documents[documentID][position] = Token{Text: fmt.Sprintf("term-%03d", (documentID+position)%500), Position: uint32(position)}
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		builder := NewFTSFieldBuilder()
		for documentID, tokens := range documents {
			if err := builder.AddDocument(context.Background(), uint32(documentID), tokens); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := builder.Build(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
}

func buildFTSTestDictionary(t testing.TB, documents [][]Token) *FTSTermDictionary {
	t.Helper()
	builder := NewFTSFieldBuilder()
	for documentID, tokens := range documents {
		if err := builder.AddDocument(context.Background(), uint32(documentID), tokens); err != nil {
			t.Fatal(err)
		}
	}
	dictionary, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return dictionary
}

func repairFTSDictionaryCRCs(data []byte) {
	if len(data) < ftsDictionaryHeaderSize {
		return
	}
	binary.LittleEndian.PutUint32(data[40:44], ailego.CRC32C(data[ftsDictionaryHeaderSize:]))
	binary.LittleEndian.PutUint32(data[60:64], ailego.CRC32C(data[:60]))
}

func sortStringsStrict(values []string) bool {
	for index := 1; index < len(values); index++ {
		if values[index-1] >= values[index] {
			return false
		}
	}
	return true
}

func TestFTSSegmentStatsAverageEmpty(t *testing.T) {
	if got := (FTSSegmentStats{}).AverageDocumentLength(); got != 1 {
		t.Fatalf("empty average = %v", got)
	}
	if got := (FTSCorpusStats{}).AverageDocumentLength(); got != 1 {
		t.Fatalf("empty corpus average = %v", got)
	}
	if got := (FTSCorpusStats{}).Terms(); len(got) != 0 {
		t.Fatalf("empty corpus terms = %#v", got)
	}
	if got := (FTSCorpusStats{}).DocumentFrequency("x"); got != 0 {
		t.Fatalf("empty corpus df = %d", got)
	}
}

func TestAggregateFTSCorpusStats(t *testing.T) {
	segment0 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "alpha", Position: 0}, {Text: "beta", Position: 1}},
		nil,
		{{Text: "alpha", Position: 0}, {Text: "alpha", Position: 1}, {Text: "only-deleted", Position: 2}},
	})
	segment1 := buildFTSTestDictionary(t, [][]Token{
		{{Text: "alpha", Position: 0}, {Text: "gamma", Position: 1}},
		{{Text: "beta", Position: 0}, {Text: "gamma", Position: 1}, {Text: "gamma", Position: 2}},
	})
	deleted0 := ailego.NewBitmap(3)
	deleted0.Set(2)
	deleted1 := ailego.NewBitmap(2)
	deleted1.Set(0)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{
		{Dictionary: segment0, DeletedDocuments: deleted0},
		{Dictionary: segment1, DeletedDocuments: deleted1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.TotalDocuments != 3 || stats.TotalTokens != 5 || stats.AverageDocumentLength() != float64(5)/3 {
		t.Fatalf("corpus totals = %#v", stats)
	}
	want := map[string]uint64{"alpha": 1, "beta": 2, "gamma": 1}
	if got := stats.DocumentFrequencies(); !reflect.DeepEqual(got, want) {
		t.Fatalf("frequencies = %#v, want %#v", got, want)
	}
	if got, wantTerms := stats.Terms(), []string{"alpha", "beta", "gamma"}; !reflect.DeepEqual(got, wantTerms) {
		t.Fatalf("terms = %#v, want %#v", got, wantTerms)
	}
	copy := stats.DocumentFrequencies()
	copy["alpha"] = 99
	if stats.DocumentFrequency("alpha") != 1 {
		t.Fatal("frequency map aliases stats")
	}
}

func TestAggregateFTSCorpusStatsValidationAndCancellation(t *testing.T) {
	if _, err := AggregateFTSCorpusStats(nil, nil); !errors.Is(err, ErrInvalidFTSStats) {
		t.Fatalf("nil context error = %v", err)
	}
	if _, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{}}); !errors.Is(err, ErrInvalidFTSStats) {
		t.Fatalf("nil dictionary error = %v", err)
	}
	dictionary := buildFTSTestDictionary(t, [][]Token{{{Text: "alpha", Position: 0}}})
	deleted := ailego.NewBitmap(65)
	deleted.Set(64)
	if _, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary, DeletedDocuments: deleted}}); !errors.Is(err, ErrInvalidFTSStats) {
		t.Fatalf("out-of-domain deletion error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := AggregateFTSCorpusStats(canceled, []FTSSegmentView{{Dictionary: dictionary}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled stats error = %v", err)
	}

	largeDocuments := make([][]Token, 5000)
	for index := range largeDocuments {
		largeDocuments[index] = []Token{{Text: "alpha", Position: 0}}
	}
	large := buildFTSTestDictionary(t, largeDocuments)
	midway := newCancelAfterChecks(3)
	if _, err := AggregateFTSCorpusStats(midway, []FTSSegmentView{{Dictionary: large}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-stats error = %v", err)
	}
}

func TestFTSTermDictionaryLongSharedPrefix(t *testing.T) {
	prefix := strings.Repeat("x", 10000)
	dictionary := buildFTSTestDictionary(t, [][]Token{{
		{Text: prefix + "a", Position: 0},
		{Text: prefix + "b", Position: 1},
	}})
	encoded, err := dictionary.Encode(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenFTSTermDictionary(context.Background(), encoded)
	if err != nil || !reflect.DeepEqual(reopened.Terms(), dictionary.Terms()) {
		t.Fatalf("long-prefix reopen = %#v, %v", reopened, err)
	}
}
