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
	"math"
	"reflect"
	"sort"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestBM25ScorerBaselineFormula(t *testing.T) {
	stats := FTSCorpusStats{
		TotalDocuments: 3, TotalTokens: 8,
		documentFrequencies: map[string]uint64{"apple": 2, "grape": 1},
	}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := scorer.Params(), (BM25Params{K1: 1.2, B: 0.75}); got != want {
		t.Fatalf("params = %#v, want %#v", got, want)
	}
	assertBM25Close(t, "idf", scorer.IDF(2), 0.470003629246, 1e-7)
	assertBM25Close(t, "term idf", scorer.TermIDF("apple"), 0.470003629246, 1e-7)
	assertBM25Close(t, "score", scorer.Score(2, 2, 3), 0.624306707526, 1e-6)
	assertBM25Close(t, "precomputed score", scorer.ScoreWithIDF(scorer.IDF(2), 2, 3), 0.624306707526, 1e-6)
	assertBM25Close(t, "boosted score", scorer.ScoreWithIDFAndBoost(scorer.IDF(2), 2, 3, 2), 1.248613415052, 2e-6)
	assertBM25Close(t, "rare short document", scorer.Score(1, 1, 1), 1.317755332291, 1e-6)
	assertBM25Close(t, "upper bound", scorer.MaxScoreBound(2), 1.034007984341, 1e-6)
	if scorer.Score(2, 0, 3) != 0 || scorer.ScoreWithIDF(-1, 2, 3) != 0 {
		t.Fatal("non-occurring or non-positive-IDF terms must score zero")
	}
}

func TestBM25ScorerValidationOwnershipAndEmptyCorpus(t *testing.T) {
	invalid := []BM25Params{
		{K1: -1, B: 0.75},
		{K1: float32(math.NaN()), B: 0.75},
		{K1: 1.2, B: -0.1},
		{K1: 1.2, B: 1.1},
		{K1: 1.2, B: float32(math.Inf(1))},
	}
	for _, params := range invalid {
		if scorer, err := NewBM25Scorer(params, FTSCorpusStats{}); scorer != nil || !errors.Is(err, ErrInvalidBM25) {
			t.Fatalf("NewBM25Scorer(%#v) = %#v, %v", params, scorer, err)
		}
	}
	for _, stats := range []FTSCorpusStats{
		{TotalTokens: 1},
		{TotalDocuments: 1, documentFrequencies: map[string]uint64{"x": 0}},
		{TotalDocuments: 1, documentFrequencies: map[string]uint64{"x": 2}},
		{TotalDocuments: 1, documentFrequencies: map[string]uint64{"x": 1}},
		{TotalDocuments: 2, TotalTokens: 1, documentFrequencies: map[string]uint64{"x": 2}},
	} {
		if scorer, err := NewBM25Scorer(DefaultBM25Params(), stats); scorer != nil || !errors.Is(err, ErrInvalidBM25) {
			t.Fatalf("invalid stats = %#v, %v", scorer, err)
		}
	}

	empty, err := NewBM25Scorer(DefaultBM25Params(), FTSCorpusStats{})
	if err != nil || empty.IDF(1) != 0 || empty.Score(1, 1, 1) != 0 || empty.MaxScoreBound(1) != 0 {
		t.Fatalf("empty scorer = %#v, %v", empty, err)
	}
	emptyDocuments, err := NewBM25Scorer(DefaultBM25Params(), FTSCorpusStats{TotalDocuments: 2})
	if err != nil || emptyDocuments.Score(1, 1, 1) != 0 {
		t.Fatalf("token-empty scorer = %#v, %v", emptyDocuments, err)
	}
	stats := FTSCorpusStats{TotalDocuments: 2, TotalTokens: 3, documentFrequencies: map[string]uint64{"x": 1}}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	if err != nil {
		t.Fatal(err)
	}
	stats.documentFrequencies["x"] = 2
	copyStats := scorer.Stats()
	copyStats.documentFrequencies["x"] = 2
	if scorer.DocumentFrequency("x") != 1 || scorer.Stats().DocumentFrequency("x") != 1 {
		t.Fatal("scorer statistics alias caller memory")
	}
	if (*BM25Scorer)(nil).IDF(1) != 0 || (*BM25Scorer)(nil).Score(1, 1, 1) != 0 {
		t.Fatal("nil scorer getters must be safe")
	}
}

func TestSearchFTSBM25BooleanPhraseBoostAndTopK(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "apple", Position: 0}, {Text: "apple", Position: 1}, {Text: "banana", Position: 2}},
		{{Text: "apple", Position: 0}, {Text: "banana", Position: 1}, {Text: "banana", Position: 2}, {Text: "banana", Position: 3}},
		{{Text: "grape", Position: 0}},
		{{Text: "apple", Position: 0}, {Text: "quick", Position: 1}, {Text: "brown", Position: 2}, {Text: "fox", Position: 3}},
		{{Text: "quick", Position: 0}, {Text: "brown", Position: 1}},
	})
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary}})
	if err != nil {
		t.Fatal(err)
	}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	if err != nil {
		t.Fatal(err)
	}
	pipeline := newFTSStandardTestPipeline(t)
	parse := func(query string) FTSQueryNode {
		node, parseErr := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		return node
	}
	search := func(query string, topK int) []FTSResult {
		results, searchErr := SearchFTS(context.Background(), dictionary, parse(query), scorer, FTSSearchOptions{TopK: topK})
		if searchErr != nil {
			t.Fatal(searchErr)
		}
		return results
	}

	apple := search("apple", 10)
	if got, want := ftsResultDocumentIDs(apple), []uint32{0, 1, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("apple IDs = %v, want %v (%#v)", got, want, apple)
	}
	assertBM25Close(t, "apple doc 0", apple[0].Score, float64(scorer.Score(3, 2, 3)), 1e-6)

	orResults := search("apple OR banana", 10)
	if got, want := ftsResultDocumentIDs(orResults), []uint32{1, 0, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("OR IDs = %v, want %v (%#v)", got, want, orResults)
	}
	assertBM25Close(t, "OR sum", orResults[0].Score, float64(scorer.Score(3, 1, 4)+scorer.Score(2, 3, 4)), 1e-6)

	shouldResults := search("+apple banana", 10)
	if !reflect.DeepEqual(shouldResults, orResults) {
		t.Fatalf("required+optional results = %#v, want %#v", shouldResults, orResults)
	}
	mustNot := search("apple AND NOT banana", 10)
	if got, want := ftsResultDocumentIDs(mustNot), []uint32{3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("must-not IDs = %v, want %v", got, want)
	}

	phrase := search(`"quick brown"`, 10)
	if got, want := ftsResultDocumentIDs(phrase), []uint32{4, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("phrase IDs = %v, want %v (%#v)", got, want, phrase)
	}
	duplicatePhrase := search(`"quick brown" OR "quick brown"`, 10)
	if got, want := ftsResultDocumentIDs(duplicatePhrase), []uint32{4, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("duplicate phrase IDs = %v, want %v", got, want)
	}
	for index := range phrase {
		assertBM25Close(t, "duplicate phrase boost", duplicatePhrase[index].Score, float64(phrase[index].Score*2), 2e-6)
	}
	optionalPhrase := search(`+apple "quick brown"`, 10)
	document3, found := findFTSResult(optionalPhrase, 3)
	if !found {
		t.Fatalf("optional phrase omitted required document: %#v", optionalPhrase)
	}
	wantDocument3 := scorer.Score(3, 1, 4) + scorer.Score(2, 1, 4) + scorer.Score(2, 1, 4)
	assertBM25Close(t, "optional phrase score", document3.Score, float64(wantDocument3), 1e-6)
	if got := search("apple OR banana", 2); !reflect.DeepEqual(got, orResults[:2]) {
		t.Fatalf("top-2 = %#v, want %#v", got, orResults[:2])
	}
}

func TestSearchFTSTiesDeletionAdvanceAndValidation(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "same", Position: 0}},
		{{Text: "same", Position: 0}},
		{{Text: "same", Position: 0}},
	})
	deleted := ailego.NewBitmap(3)
	deleted.Set(1)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary, DeletedDocuments: deleted}})
	if err != nil {
		t.Fatal(err)
	}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	if err != nil {
		t.Fatal(err)
	}
	node := &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "same"}
	options := FTSQueryExecutionOptions{DeletedDocuments: deleted}
	iterator, err := NewFTSScoredQueryIterator(context.Background(), dictionary, node, scorer, options)
	if err != nil {
		t.Fatal(err)
	}
	if !iterator.Advance(context.Background(), 2) || iterator.DocumentID() != 2 || iterator.Score() <= 0 {
		t.Fatalf("Advance = %v, %d, %g", iterator.Valid(), iterator.DocumentID(), iterator.Score())
	}
	if !iterator.Advance(context.Background(), 1) || iterator.DocumentID() != 2 {
		t.Fatal("Advance moved a suitable current result")
	}
	results, err := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: 1, FTSQueryExecutionOptions: options})
	if err != nil || !reflect.DeepEqual(ftsResultDocumentIDs(results), []uint32{0}) {
		t.Fatalf("tie/deletion search = %#v, %v", results, err)
	}
	if _, err := NewFTSScoredQueryIterator(context.Background(), dictionary, node, nil, options); !errors.Is(err, ErrInvalidFTSQueryExecution) {
		t.Fatalf("nil scorer = %v", err)
	}
	if _, err := NewFTSScoredQueryIterator(nil, dictionary, node, scorer, options); !errors.Is(err, ErrInvalidFTSQueryExecution) {
		t.Fatalf("nil scored-iterator context = %v", err)
	}
	if _, err := SearchFTS(nil, dictionary, node, scorer, FTSSearchOptions{TopK: 1}); !errors.Is(err, ErrInvalidFTSSearch) {
		t.Fatalf("nil context = %v", err)
	}
	if _, err := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: -1}); !errors.Is(err, ErrInvalidFTSSearch) {
		t.Fatalf("negative TopK = %v", err)
	}
	if results, err := SearchFTS(context.Background(), dictionary, nil, scorer, FTSSearchOptions{TopK: 0}); err != nil || len(results) != 0 {
		t.Fatalf("zero TopK = %#v, %v", results, err)
	}
	if _, err := SearchFTS(context.Background(), nil, nil, scorer, FTSSearchOptions{TopK: 0}); !errors.Is(err, ErrInvalidFTSQueryExecution) {
		t.Fatalf("zero TopK nil dictionary = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SearchFTS(canceled, dictionary, node, scorer, FTSSearchOptions{TopK: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled search = %v", err)
	}
}

func TestBM25ScorerAndSearchConcurrentUse(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "x", Position: 0}}, {{Text: "x", Position: 0}}, {{Text: "y", Position: 0}},
	})
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary}})
	if err != nil {
		t.Fatal(err)
	}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	if err != nil {
		t.Fatal(err)
	}
	node := &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "x"}
	var wait sync.WaitGroup
	errorsChannel := make(chan error, 16)
	for range 16 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			results, searchErr := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: 2})
			if searchErr != nil {
				errorsChannel <- searchErr
				return
			}
			if got := ftsResultDocumentIDs(results); !reflect.DeepEqual(got, []uint32{0, 1}) {
				errorsChannel <- errors.New("unexpected concurrent result")
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		t.Error(err)
	}
}

func FuzzBM25Scorer(f *testing.F) {
	f.Add(uint64(3), uint64(8), uint64(2), uint32(2), uint32(3))
	f.Add(uint64(1), uint64(0), uint64(1), uint32(1), uint32(0))
	f.Fuzz(func(t *testing.T, documents, tokens, documentFrequency uint64, termFrequency, documentLength uint32) {
		documents = documents%1_000_000 + 1
		documentFrequency = documentFrequency%documents + 1
		stats := FTSCorpusStats{
			TotalDocuments: documents, TotalTokens: tokens%1_000_000 + documentFrequency,
			documentFrequencies: map[string]uint64{"x": documentFrequency},
		}
		scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
		if err != nil {
			t.Fatal(err)
		}
		score := scorer.Score(documentFrequency, termFrequency, documentLength)
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) || score < 0 {
			t.Fatalf("score = %g", score)
		}
	})
}

func FuzzSearchFTSBM25(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4}, uint8(0b0011), false, uint16(0), uint8(3))
	f.Add([]byte{7, 7, 7}, uint8(0b0101), true, uint16(0b010), uint8(2))
	f.Fuzz(func(t *testing.T, data []byte, queryMask uint8, requireAll bool, deletionMask uint16, topKByte uint8) {
		if len(data) > 16 {
			data = data[:16]
		}
		if len(data) == 0 {
			return
		}
		queryMask &= 0x0f
		if queryMask == 0 {
			queryMask = 1
		}
		documents := make([][]Token, len(data))
		termCounts := make([][4]uint32, len(data))
		for documentID, value := range data {
			tokenCount := int(value%4) + 1
			documents[documentID] = make([]Token, tokenCount)
			for position := range tokenCount {
				termIndex := (int(value) + position) % 4
				documents[documentID][position] = Token{Text: string(rune('a' + termIndex)), Position: uint32(position)}
				termCounts[documentID][termIndex]++
			}
		}
		dictionary := buildFTSTestDictionary(t, documents)
		deleted := ailego.NewBitmap(uint64(len(documents)))
		for documentID := range documents {
			if deletionMask&(uint16(1)<<documentID) != 0 {
				deleted.Set(uint64(documentID))
			}
		}
		stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary, DeletedDocuments: deleted}})
		if err != nil {
			t.Fatal(err)
		}
		scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
		if err != nil {
			t.Fatal(err)
		}
		children := make([]FTSQueryNode, 0, 4)
		selected := make([]int, 0, 4)
		for termIndex := range 4 {
			if queryMask&(1<<termIndex) == 0 {
				continue
			}
			selected = append(selected, termIndex)
			children = append(children, &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: string(rune('a' + termIndex))})
		}
		var node FTSQueryNode
		if len(children) == 1 {
			node = children[0]
		} else if requireAll {
			node = &FTSAndQueryNode{Flags: defaultFTSQueryModifier(), Children: children}
		} else {
			node = &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: children}
		}
		topK := int(topKByte%uint8(len(documents))) + 1
		got, err := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{
			TopK: topK, FTSQueryExecutionOptions: FTSQueryExecutionOptions{DeletedDocuments: deleted},
		})
		if err != nil {
			t.Fatal(err)
		}
		want := make([]FTSResult, 0, len(documents))
		for documentID, counts := range termCounts {
			if deleted.Contains(uint64(documentID)) {
				continue
			}
			matched := !requireAll
			if requireAll {
				matched = true
			}
			var score float32
			for _, termIndex := range selected {
				if counts[termIndex] == 0 {
					if requireAll {
						matched = false
					}
					continue
				}
				if !requireAll {
					matched = true
				}
				term := string(rune('a' + termIndex))
				score += scorer.Score(stats.DocumentFrequency(term), counts[termIndex], uint32(len(documents[documentID])))
			}
			if matched && score > 0 {
				want = append(want, FTSResult{DocumentID: uint32(documentID), Score: score})
			}
		}
		sort.Slice(want, func(left, right int) bool { return ftsResultBetter(want[left], want[right]) })
		if len(want) > topK {
			want = want[:topK]
		}
		if len(got) != len(want) {
			t.Fatalf("result count = %d, want %d: %#v / %#v", len(got), len(want), got, want)
		}
		for index := range want {
			if got[index].DocumentID != want[index].DocumentID {
				t.Fatalf("result %d = %#v, want %#v", index, got[index], want[index])
			}
			assertBM25Close(t, "fuzz result", got[index].Score, float64(want[index].Score), 1e-6)
		}
	})
}

func BenchmarkSearchFTSBM25(b *testing.B) {
	documents := make([][]Token, 10_000)
	for documentID := range documents {
		documents[documentID] = []Token{{Text: "common", Position: 0}, {Text: "term", Position: 1}}
		if documentID%5 == 0 {
			documents[documentID] = append(documents[documentID], Token{Text: "rare", Position: 2})
		}
	}
	dictionary := buildFTSTestDictionary(b, documents)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary}})
	if err != nil {
		b.Fatal(err)
	}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	if err != nil {
		b.Fatal(err)
	}
	node := &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{
		&FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "common"},
		&FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "rare"},
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: 10}); err != nil {
			b.Fatal(err)
		}
	}
}

func assertBM25Close(t testing.TB, name string, got float32, want, tolerance float64) {
	t.Helper()
	if difference := math.Abs(float64(got) - want); difference > tolerance {
		t.Fatalf("%s = %.12g, want %.12g (difference %.12g)", name, got, want, difference)
	}
}

func ftsResultDocumentIDs(results []FTSResult) []uint32 {
	documentIDs := make([]uint32, len(results))
	for index, result := range results {
		documentIDs[index] = result.DocumentID
	}
	return documentIDs
}

func findFTSResult(results []FTSResult, documentID uint32) (FTSResult, bool) {
	for _, result := range results {
		if result.DocumentID == documentID {
			return result, true
		}
	}
	return FTSResult{}, false
}
