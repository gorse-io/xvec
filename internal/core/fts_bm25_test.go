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

package core

import (
	"context"
	"errors"
	"math"
	"sort"
	"sync"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBM25ScorerBaselineFormula(t *testing.T) {
	stats := FTSCorpusStats{
		TotalDocuments: 3, TotalTokens: 8,
		documentFrequencies: map[string]uint64{"apple": 2, "grape": 1},
	}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)
	{
		got, want := scorer.Params(), (BM25Params{K1: 1.2, B: 0.75})
		require.Equal(t, want, got)
	}

	assertBM25Close(t, "idf", scorer.IDF(2), 0.470003629246, 1e-7)
	assertBM25Close(t, "term idf", scorer.TermIDF("apple"), 0.470003629246, 1e-7)
	assertBM25Close(t, "score", scorer.Score(2, 2, 3), 0.624306707526, 1e-6)
	assertBM25Close(t, "precomputed score", scorer.ScoreWithIDF(scorer.IDF(2), 2, 3), 0.624306707526, 1e-6)
	assertBM25Close(t, "boosted score", scorer.ScoreWithIDFAndBoost(scorer.IDF(2), 2, 3, 2), 1.248613415052, 2e-6)
	assertBM25Close(t, "rare short document", scorer.Score(1, 1, 1), 1.317755332291, 1e-6)
	assertBM25Close(t, "upper bound", scorer.MaxScoreBound(2), 1.034007984341, 1e-6)
	require.True(t, scorer.Score(2, 0, 3) == 0,
		"non-occurring or non-positive-IDF terms must score zero")
	require.True(t, scorer.ScoreWithIDF(-1, 2, 3) == 0,
		"non-occurring or non-positive-IDF terms must score zero")
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
		{
			scorer, err := NewBM25Scorer(params, FTSCorpusStats{})
			require.Nil(t, scorer)
			require.ErrorIs(t, err, ErrInvalidBM25)
		}
	}
	for _, stats := range []FTSCorpusStats{
		{TotalTokens: 1},
		{TotalDocuments: 1, documentFrequencies: map[string]uint64{"x": 0}},
		{TotalDocuments: 1, documentFrequencies: map[string]uint64{"x": 2}},
		{TotalDocuments: 1, documentFrequencies: map[string]uint64{"x": 1}},
		{TotalDocuments: 2, TotalTokens: 1, documentFrequencies: map[string]uint64{"x": 2}},
	} {
		{
			scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
			require.Nil(t, scorer)
			require.ErrorIs(t, err, ErrInvalidBM25)
		}
	}

	empty, err := NewBM25Scorer(DefaultBM25Params(), FTSCorpusStats{})
	require.NoError(t, err)
	require.True(t, empty.IDF(1) == 0)
	require.True(t, empty.Score(1, 1, 1) == 0)
	require.True(t, empty.MaxScoreBound(1) == 0)

	emptyDocuments, err := NewBM25Scorer(DefaultBM25Params(), FTSCorpusStats{TotalDocuments: 2})
	require.NoError(t, err)
	require.True(t, emptyDocuments.Score(1, 1, 1) == 0)

	stats := FTSCorpusStats{TotalDocuments: 2, TotalTokens: 3, documentFrequencies: map[string]uint64{"x": 1}}
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)

	stats.documentFrequencies["x"] = 2
	copyStats := scorer.Stats()
	copyStats.documentFrequencies["x"] = 2
	require.True(t, scorer.DocumentFrequency("x") == 1,
		"scorer statistics alias caller memory")
	require.True(t, scorer.Stats().DocumentFrequency("x") == 1,
		"scorer statistics alias caller memory")
	require.True(t, (*BM25Scorer)(nil).IDF(1) == 0,
		"nil scorer getters must be safe")
	require.True(t, (*BM25Scorer)(nil).Score(1, 1, 1) == 0,
		"nil scorer getters must be safe")
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
	require.NoError(t, err)

	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)

	pipeline := newFTSStandardTestPipeline(t)
	parse := func(query string) FTSQueryNode {
		node, parseErr := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
		require.NoError(t, parseErr)

		return node
	}
	search := func(query string, topK int) []FTSResult {
		results, searchErr := SearchFTS(context.Background(), dictionary, parse(query), scorer, FTSSearchOptions{TopK: topK})
		require.NoError(t, searchErr)

		return results
	}

	apple := search("apple", 10)
	{
		got, want := ftsResultDocumentIDs(apple), []uint32{0, 1, 3}
		require.Equal(t, want, got)
	}

	assertBM25Close(t, "apple doc 0", apple[0].Score, float64(scorer.Score(3, 2, 3)), 1e-6)

	orResults := search("apple OR banana", 10)
	{
		got, want := ftsResultDocumentIDs(orResults), []uint32{1, 0, 3}
		require.Equal(t, want, got)
	}

	assertBM25Close(t, "OR sum", orResults[0].Score, float64(scorer.Score(3, 1, 4)+scorer.Score(2, 3, 4)), 1e-6)

	shouldResults := search("+apple banana", 10)
	require.Equal(t, orResults, shouldResults)

	mustNot := search("apple AND NOT banana", 10)
	{
		got, want := ftsResultDocumentIDs(mustNot), []uint32{3}
		require.Equal(t, want, got)
	}

	phrase := search(`"quick brown"`, 10)
	{
		got, want := ftsResultDocumentIDs(phrase), []uint32{4, 3}
		require.Equal(t, want, got)
	}

	duplicatePhrase := search(`"quick brown" OR "quick brown"`, 10)
	{
		got, want := ftsResultDocumentIDs(duplicatePhrase), []uint32{4, 3}
		require.Equal(t, want, got)
	}

	for index := range phrase {
		assertBM25Close(t, "duplicate phrase boost", duplicatePhrase[index].Score, float64(phrase[index].Score*2), 2e-6)
	}
	optionalPhrase := search(`+apple "quick brown"`, 10)
	document3, found := findFTSResult(optionalPhrase, 3)
	require.True(t, found)

	wantDocument3 := scorer.Score(3, 1, 4) + scorer.Score(2, 1, 4) + scorer.Score(2, 1, 4)
	assertBM25Close(t, "optional phrase score", document3.Score, float64(wantDocument3), 1e-6)
	{
		got := search("apple OR banana", 2)
		require.Equal(t, orResults[:2], got)
	}
}

func TestSearchFTSBlockMaxSkipsNonCompetitivePostingBlocks(t *testing.T) {
	documents := make([][]Token, 256)
	for documentID := range documents {
		if documentID < 128 {
			documents[documentID] = repeatedFTSTestTokens("target", 16)
			continue
		}
		documents[documentID] = append(repeatedFTSTestTokens("target", 1), repeatedFTSTestTokens("filler", 63)...)
		for position := range documents[documentID] {
			documents[documentID][position].Position = uint32(position)
		}
	}
	dictionary := buildFTSTestDictionary(t, documents)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary}})
	require.NoError(t, err)
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)
	node := &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "target"}

	results, searchStats, err := searchFTSWithStats(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: 1})
	require.NoError(t, err)
	require.Equal(t, []uint32{0}, ftsResultDocumentIDs(results))
	require.Positive(t, searchStats.blockMaxSkips)
	require.Equal(t, uint64(128), searchStats.scoredDocuments,
		"the non-competitive second posting block should not be scored")
}

func TestSearchFTSBlockMaxMatchesExhaustiveBooleanAndPhraseSearch(t *testing.T) {
	documents := make([][]Token, 640)
	for documentID := range documents {
		terms := make([]string, 0, 16)
		for range 1 + documentID%5 {
			terms = append(terms, "apple")
		}
		if documentID%2 == 0 {
			terms = append(terms, "banana")
		}
		if documentID%3 == 0 {
			terms = append(terms, "quick", "brown")
		}
		for range documentID % 9 {
			terms = append(terms, "filler")
		}
		tokens := make([]Token, len(terms))
		for position, term := range terms {
			tokens[position] = Token{Text: term, Position: uint32(position)}
		}
		documents[documentID] = tokens
	}
	dictionary := buildFTSTestDictionary(t, documents)
	deleted := container.NewBitmap(uint64(len(documents)))
	for documentID := 17; documentID < len(documents); documentID += 71 {
		deleted.Set(uint64(documentID))
	}
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary, DeletedDocuments: deleted}})
	require.NoError(t, err)
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)
	pipeline := newFTSStandardTestPipeline(t)
	options := FTSSearchOptions{TopK: 7, FTSQueryExecutionOptions: FTSQueryExecutionOptions{DeletedDocuments: deleted}}

	var totalSkips uint64
	for _, query := range []string{
		"apple", "apple OR banana", "apple AND banana", `"quick brown"`,
		"+apple banana", "apple AND NOT banana",
	} {
		node, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
		require.NoError(t, err)
		want := exhaustiveFTSSearchForTest(t, dictionary, node, scorer, options)
		got, searchStats, err := searchFTSWithStats(context.Background(), dictionary, node, scorer, options)
		require.NoError(t, err)
		require.Equal(t, want, got, query)
		totalSkips += searchStats.blockMaxSkips
	}
	require.Positive(t, totalSkips)
}

func TestSearchFTSBlockMaxDoesNotHideNonFiniteScores(t *testing.T) {
	documents := make([][]Token, 256)
	for documentID := range documents {
		if documentID < 128 {
			documents[documentID] = repeatedFTSTestTokens("strong", 16)
		} else {
			documents[documentID] = append(repeatedFTSTestTokens("strong", 1), repeatedFTSTestTokens("filler", 63)...)
		}
		if documentID == 200 {
			documents[documentID] = append(documents[documentID], Token{Text: "toxic"})
		}
		for position := range documents[documentID] {
			documents[documentID][position].Position = uint32(position)
		}
	}
	dictionary := buildFTSTestDictionary(t, documents)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary}})
	require.NoError(t, err)
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)
	node := &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{
		&FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "strong"},
		&FTSTermQueryNode{Flags: FTSQueryModifier{Boost: -math.MaxFloat32}, Term: "toxic"},
	}}

	_, err = SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: 1})
	require.ErrorIs(t, err, ErrInvalidFTSSearch)
}

func TestSearchFTSWANDSkipsCandidatesBeforePivot(t *testing.T) {
	documents := make([][]Token, 201)
	documents[0] = repeatedFTSTestTokens("winner", 16)
	for documentID := 1; documentID <= 128; documentID++ {
		documents[documentID] = append(repeatedFTSTestTokens("low", 1), repeatedFTSTestTokens("filler", 63)...)
	}
	for documentID := 129; documentID < 200; documentID++ {
		documents[documentID] = repeatedFTSTestTokens("filler", 1)
	}
	documents[200] = repeatedFTSTestTokens("future", 16)
	for documentID := range documents {
		for position := range documents[documentID] {
			documents[documentID][position].Position = uint32(position)
		}
	}
	dictionary := buildFTSTestDictionary(t, documents)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary}})
	require.NoError(t, err)
	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)
	node := &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{
		&FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "winner"},
		&FTSTermQueryNode{Flags: FTSQueryModifier{Boost: 0.05}, Term: "low"},
		&FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "future"},
	}}
	options := FTSSearchOptions{TopK: 1}

	want := exhaustiveFTSSearchForTest(t, dictionary, node, scorer, options)
	got, searchStats, err := searchFTSWithStats(context.Background(), dictionary, node, scorer, options)
	require.NoError(t, err)
	require.Equal(t, want, got)
	require.Equal(t, []uint32{0}, ftsResultDocumentIDs(got))
	require.Positive(t, searchStats.wandSkips)
	require.Equal(t, uint64(2), searchStats.scoredDocuments,
		"WAND should jump from the first low-score candidate to the future pivot")
}

func repeatedFTSTestTokens(term string, count int) []Token {
	tokens := make([]Token, count)
	for position := range tokens {
		tokens[position] = Token{Text: term, Position: uint32(position)}
	}
	return tokens
}

func exhaustiveFTSSearchForTest(
	t testing.TB,
	dictionary *FTSTermDictionary,
	node FTSQueryNode,
	scorer *BM25Scorer,
	options FTSSearchOptions,
) []FTSResult {
	t.Helper()
	iterator, err := NewFTSScoredQueryIterator(context.Background(), dictionary, node, scorer, options.FTSQueryExecutionOptions)
	require.NoError(t, err)
	results := make([]FTSResult, 0)
	for iterator.Next(context.Background()) {
		if iterator.Score() > 0 {
			results = append(results, FTSResult{DocumentID: iterator.DocumentID(), Score: iterator.Score()})
		}
	}
	require.NoError(t, iterator.Err())
	sort.Slice(results, func(left, right int) bool { return ftsResultBetter(results[left], results[right]) })
	if len(results) > options.TopK {
		results = results[:options.TopK]
	}
	return results
}

func TestSearchFTSTiesDeletionAdvanceAndValidation(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "same", Position: 0}},
		{{Text: "same", Position: 0}},
		{{Text: "same", Position: 0}},
	})
	deleted := container.NewBitmap(3)
	deleted.Set(1)
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary, DeletedDocuments: deleted}})
	require.NoError(t, err)

	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)

	node := &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "same"}
	options := FTSQueryExecutionOptions{DeletedDocuments: deleted}
	iterator, err := NewFTSScoredQueryIterator(context.Background(), dictionary, node, scorer, options)
	require.NoError(t, err)
	require.True(t, iterator.Advance(context.Background(), 2))
	require.True(t, iterator.DocumentID() == 2)
	require.True(t, iterator.Score() > 0)
	require.True(t, iterator.Advance(context.Background(), 1),
		"Advance moved a suitable current result")
	require.True(t, iterator.DocumentID() == 2,
		"Advance moved a suitable current result")

	results, err := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: 1, FTSQueryExecutionOptions: options})
	require.NoError(t, err)
	require.Equal(t, []uint32{0}, ftsResultDocumentIDs(results))
	{
		_, err := NewFTSScoredQueryIterator(context.Background(), dictionary, node, nil, options)
		require.ErrorIs(t, err, ErrInvalidFTSQueryExecution)
	}
	{
		_, err := NewFTSScoredQueryIterator(nil, dictionary, node, scorer, options)
		require.ErrorIs(t, err, ErrInvalidFTSQueryExecution)
	}
	{
		_, err := SearchFTS(nil, dictionary, node, scorer, FTSSearchOptions{TopK: 1})
		require.ErrorIs(t, err, ErrInvalidFTSSearch)
	}
	{
		_, err := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: -1})
		require.ErrorIs(t, err, ErrInvalidFTSSearch)
	}
	{
		results, err := SearchFTS(context.Background(), dictionary, nil, scorer, FTSSearchOptions{TopK: 0})
		require.NoError(t, err)
		require.Len(t, results, 0)
	}
	{
		_, err := SearchFTS(context.Background(), nil, nil, scorer, FTSSearchOptions{TopK: 0})
		require.ErrorIs(t, err, ErrInvalidFTSQueryExecution)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := SearchFTS(canceled, dictionary, node, scorer, FTSSearchOptions{TopK: 1})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestBM25ScorerAndSearchConcurrentUse(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "x", Position: 0}}, {{Text: "x", Position: 0}}, {{Text: "y", Position: 0}},
	})
	stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary}})
	require.NoError(t, err)

	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	require.NoError(t, err)

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
			if got := ftsResultDocumentIDs(results); !assert.Equal(t, []uint32{0, 1}, got) {
				errorsChannel <- errors.New("unexpected concurrent result")
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		assert.NoError(t, err)
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
		require.NoError(t, err)

		score := scorer.Score(documentFrequency, termFrequency, documentLength)
		require.False(t, math.IsNaN(float64(score)))
		require.False(t, math.IsInf(float64(score), 0))
		require.True(t, score >= 0)
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
		deleted := container.NewBitmap(uint64(len(documents)))
		for documentID := range documents {
			if deletionMask&(uint16(1)<<documentID) != 0 {
				deleted.Set(uint64(documentID))
			}
		}
		stats, err := AggregateFTSCorpusStats(context.Background(), []FTSSegmentView{{Dictionary: dictionary, DeletedDocuments: deleted}})
		require.NoError(t, err)

		scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
		require.NoError(t, err)

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
		require.NoError(t, err)

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
		require.Len(t, got, len(want))

		for index := range want {
			require.Equal(t, want[index].DocumentID, got[index].DocumentID)

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
		require.NoError(b, err)
	}

	scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
	if err != nil {
		require.NoError(b, err)
	}

	node := &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: []FTSQueryNode{
		&FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "common"},
		&FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: "rare"},
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		{
			_, err := SearchFTS(context.Background(), dictionary, node, scorer, FTSSearchOptions{TopK: 10})
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func assertBM25Close(t testing.TB, name string, got float32, want, tolerance float64) {
	t.Helper()
	require.InDelta(t, want, got, tolerance, name)
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
