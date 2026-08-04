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

package zvec

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/core"
)

type testRerankerFunc func(context.Context, []RerankBatch, int) ([]Document, error)

func (f testRerankerFunc) Rerank(ctx context.Context, batches []RerankBatch, topK int) ([]Document, error) {
	return f(ctx, batches, topK)
}

type testNilReranker struct{}

func (*testNilReranker) Rerank(context.Context, []RerankBatch, int) ([]Document, error) {
	return nil, nil
}

func TestCollectionMultiQueryDenseSparseFTSFilterProjectionAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "hybrid")
	collection, err := CreateAndOpen(ctx, path, testMultiQuerySchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, testMultiQueryDocuments()); err != nil {
		t.Fatal(err)
	}
	if completeness := collection.Stats().IndexCompleteness["title"]; completeness != 1 {
		t.Fatalf("FTS completeness = %v", completeness)
	}

	assertBatches := func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		if topK != 2 || len(batches) != 3 {
			t.Fatalf("rerank shape = topK %d, batches %d", topK, len(batches))
		}
		if got := []string{batches[0].Field.Name, batches[1].Field.Name, batches[2].Field.Name}; !reflect.DeepEqual(got, []string{"embedding", "sparse", "title"}) {
			t.Fatalf("batch fields = %#v", got)
		}
		if got := documentKeys(batches[0].Documents); !reflect.DeepEqual(got, []string{"a", "b", "d"}) {
			t.Fatalf("dense candidates = %#v", got)
		}
		if got := documentKeys(batches[1].Documents); !reflect.DeepEqual(got, []string{"b", "a"}) {
			t.Fatalf("sparse candidates = %#v", got)
		}
		if got := documentKeys(batches[2].Documents); !reflect.DeepEqual(got, []string{"b", "a"}) {
			t.Fatalf("FTS candidates = %#v", got)
		}
		for _, batch := range batches {
			for _, document := range batch.Documents {
				if len(document.Fields) != 1 {
					t.Fatalf("candidate projection = %#v", document)
				}
			}
		}
		first := batches[1].Documents[0]
		second := batches[1].Documents[1]
		first.Score, second.Score = 42, 41
		first.Fields["title"] = "forged by reranker"
		return []Document{first, second}, nil
	}
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 3},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 3},
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 3},
		},
		TopK: 2, Filter: "category = 'keep'",
		Projection: Projection{OutputFields: []string{"title"}},
		Reranker:   testRerankerFunc(assertBatches),
	}
	results, err := collection.MultiQuery(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	assertHybridResults(t, results)
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}

	reopenOptions := NewCollectionOptions()
	reopenOptions.ReadOnly = true
	reopened, err := Open(ctx, path, reopenOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	results, err = reopened.MultiQuery(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	assertHybridResults(t, results)
}

func TestCollectionMultiQueryFTSExpressionDefaultOperatorAndFilteredBM25(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "fts"), testMultiQuerySchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, testMultiQueryDocuments()); err != nil {
		t.Fatal(err)
	}

	params := NewFTSQueryParams()
	params.DefaultOperator = "and"
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Query: `"go search"`}, NumCandidates: 4},
			{Field: "title", FTS: &FTSClause{Match: "go database"}, Params: params, NumCandidates: 4},
		},
		TopK: 2,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			if topK != 2 || !reflect.DeepEqual(documentKeys(batches[0].Documents), []string{"b"}) {
				t.Fatalf("phrase batch = %#v", documentKeys(batches[0].Documents))
			}
			if !reflect.DeepEqual(documentKeys(batches[1].Documents), []string{"a", "c"}) {
				t.Fatalf("AND match batch = %#v", documentKeys(batches[1].Documents))
			}
			return []Document{batches[0].Documents[0], batches[1].Documents[0]}, nil
		}),
	}
	results, err := collection.MultiQuery(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"b", "a"}) {
		t.Fatalf("FTS multi-query = %#v, %v", results, err)
	}

	var unfilteredScore, filteredScore float32
	scoreReranker := func(destination *float32) Reranker {
		return testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
			for _, document := range batches[0].Documents {
				if document.PrimaryKey == "a" {
					*destination = document.Score
					return []Document{document}, nil
				}
			}
			t.Fatal("document a missing from FTS candidates")
			return nil, nil
		})
	}
	base := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 4},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
		},
		TopK: 1, Reranker: scoreReranker(&unfilteredScore),
	}
	if _, err := collection.MultiQuery(ctx, base); err != nil {
		t.Fatal(err)
	}
	base.Filter = "category = 'keep'"
	base.Reranker = scoreReranker(&filteredScore)
	if _, err := collection.MultiQuery(ctx, base); err != nil {
		t.Fatal(err)
	}
	if unfilteredScore == 0 || filteredScore != unfilteredScore {
		t.Fatalf("BM25 score changed under candidate filter: %v -> %v", unfilteredScore, filteredScore)
	}
}

func TestCollectionMultiQueryVectorParamsAndEmptySnapshot(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "params"), testMultiQuerySchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, testMultiQueryDocuments()); err != nil {
		t.Fatal(err)
	}
	denseParams := NewFlatQueryParams()
	denseParams.Radius = 0.9
	sparseParams := NewFlatQueryParams()
	sparseParams.Radius = 2
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, Params: denseParams},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, Params: sparseParams},
		},
		Filter: "category = 'keep'",
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			if topK != DefaultMultiQueryTopK {
				t.Fatalf("default TopK = %d", topK)
			}
			if got := documentKeys(batches[0].Documents); !reflect.DeepEqual(got, []string{"a"}) {
				t.Fatalf("dense radius candidates = %#v", got)
			}
			if got := documentKeys(batches[1].Documents); !reflect.DeepEqual(got, []string{"b"}) {
				t.Fatalf("sparse radius candidates = %#v", got)
			}
			return []Document{batches[0].Documents[0], batches[1].Documents[0]}, nil
		}),
	}
	if results, err := collection.MultiQuery(ctx, query); err != nil || !reflect.DeepEqual(documentKeys(results), []string{"a", "b"}) {
		t.Fatalf("parameterized query = %#v, %v", results, err)
	}

	empty, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "empty"), testMultiQuerySchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	emptyQuery := query
	emptyQuery.Filter = ""
	emptyQuery.Queries[0].Params = nil
	emptyQuery.Queries[1] = SubQuery{Field: "title", FTS: &FTSClause{Match: "go"}}
	emptyQuery.Reranker = testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		if topK != DefaultMultiQueryTopK || len(batches) != 2 || len(batches[0].Documents) != 0 || len(batches[1].Documents) != 0 {
			t.Fatalf("empty batches = %#v, topK %d", batches, topK)
		}
		return nil, nil
	})
	if results, err := empty.MultiQuery(ctx, emptyQuery); err != nil || len(results) != 0 {
		t.Fatalf("empty query = %#v, %v", results, err)
	}
}

func TestCollectionFTSAnalyzerConfiguration(t *testing.T) {
	ctx := context.Background()
	tests := []struct {
		name       string
		params     FTSIndexParams
		text       string
		wantTokens []string
	}{
		{
			name: "whitespace filters",
			params: FTSIndexParams{
				Tokenizer: "whitespace", Filters: []string{"lowercase", "ascii_folding"},
			},
			text: "CAFÉ Go", wantTokens: []string{"cafe", "go"},
		},
		{
			name: "standard max token length",
			params: FTSIndexParams{
				Tokenizer: "standard", ExtraParams: `{"max_token_length":2}`,
			},
			text: "abc", wantTokens: []string{"ab", "c"},
		},
		{
			name: "ngram options",
			params: FTSIndexParams{
				Tokenizer: "ngram", ExtraParams: `{"ngram_min":1,"ngram_max":2,"token_chars":["letter"]}`,
			},
			text: "a1b", wantTokens: []string{"a", "b"},
		},
		{
			name: "stemmer language",
			params: FTSIndexParams{
				Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"}, ExtraParams: `{"stemmer_lang":"english"}`,
			},
			text: "RUNNING", wantTokens: []string{"run"},
		},
	}
	jiebaDirectory, err := filepath.Abs(filepath.Join("internal", "core", "testdata", "jieba"))
	if err != nil {
		t.Fatal(err)
	}
	jiebaExtra, _ := json.Marshal(map[string]string{
		"jieba_dict_dir": jiebaDirectory,
		"user_dict_path": filepath.Join(jiebaDirectory, "user.dict.utf8"),
		"cut_mode":       "search",
	})
	tests = append(tests, struct {
		name       string
		params     FTSIndexParams
		text       string
		wantTokens []string
	}{
		name: "jieba resources", params: FTSIndexParams{Tokenizer: "jieba", ExtraParams: string(jiebaExtra)},
		text: "中华人民共和国", wantTokens: []string{"中华", "人民", "共和", "共和国", "中华人民共和国"},
	})
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			if err := testCase.params.Validate(); err != nil {
				t.Fatal(err)
			}
			analyzer, err := newCollectionFTSAnalyzer(ctx, testCase.params)
			if err != nil {
				t.Fatal(err)
			}
			tokens, err := analyzer.Analyze(ctx, testCase.text)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]string, len(tokens))
			for index := range tokens {
				got[index] = tokens[index].Text
			}
			if !reflect.DeepEqual(got, testCase.wantTokens) {
				t.Fatalf("tokens = %#v, want %#v", got, testCase.wantTokens)
			}
		})
	}

	defaults, err := newCollectionFTSAnalyzer(ctx, NewFTSIndexParams())
	if err != nil {
		t.Fatal(err)
	}
	pipeline, ok := defaults.(*core.FTSTokenizerPipeline)
	if !ok || pipeline.TokenizerName() != "standard" || !reflect.DeepEqual(pipeline.FilterNames(), []string{"lowercase"}) {
		t.Fatalf("default pipeline = %#v", defaults)
	}
}

func TestCollectionMultiQueryValidationAndRerankerBoundaries(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "validation")
	collection, err := CreateAndOpen(ctx, path, testMultiQuerySchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, testMultiQueryDocuments()); err != nil {
		t.Fatal(err)
	}
	validReranker := testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		return firstDistinctDocuments(batches, topK), nil
	})
	valid := func() MultiQuery {
		return MultiQuery{
			Queries: []SubQuery{
				{Field: "embedding", DenseVector: VectorFP32{1, 0}},
				{Field: "title", FTS: &FTSClause{Match: "go"}},
			},
			Reranker: validReranker,
		}
	}
	tests := []struct {
		name   string
		mutate func(*MultiQuery)
	}{
		{name: "one sub-query", mutate: func(query *MultiQuery) { query.Queries = query.Queries[:1] }},
		{name: "negative top-k", mutate: func(query *MultiQuery) { query.TopK = -1 }},
		{name: "oversized top-k", mutate: func(query *MultiQuery) { query.TopK = MaxQueryTopK + 1 }},
		{name: "negative candidates", mutate: func(query *MultiQuery) { query.Queries[0].NumCandidates = -1 }},
		{name: "oversized candidates", mutate: func(query *MultiQuery) { query.Queries[0].NumCandidates = MaxQueryTopK + 1 }},
		{name: "missing field", mutate: func(query *MultiQuery) { query.Queries[0].Field = "missing" }},
		{name: "missing target", mutate: func(query *MultiQuery) { query.Queries[0].DenseVector = nil }},
		{name: "multiple targets", mutate: func(query *MultiQuery) { query.Queries[0].FTS = &FTSClause{Match: "go"} }},
		{name: "dense target on FTS", mutate: func(query *MultiQuery) { query.Queries[0].Field = "title" }},
		{name: "FTS target on vector", mutate: func(query *MultiQuery) { query.Queries[1].Field = "embedding" }},
		{name: "empty FTS clause", mutate: func(query *MultiQuery) { query.Queries[1].FTS = &FTSClause{} }},
		{name: "two FTS strings", mutate: func(query *MultiQuery) { query.Queries[1].FTS = &FTSClause{Query: "go", Match: "go"} }},
		{name: "wrong FTS params", mutate: func(query *MultiQuery) { value := NewFlatQueryParams(); query.Queries[1].Params = value }},
		{name: "wrong vector params", mutate: func(query *MultiQuery) { value := NewFTSQueryParams(); query.Queries[0].Params = value }},
		{name: "malformed FTS expression", mutate: func(query *MultiQuery) { query.Queries[1].FTS = &FTSClause{Query: "(go"} }},
		{name: "invalid filter", mutate: func(query *MultiQuery) { query.Filter = "rating >>> 1" }},
		{name: "invalid projection", mutate: func(query *MultiQuery) { query.Projection.OutputFields = []string{"missing"} }},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			query := valid()
			testCase.mutate(&query)
			if _, err := collection.MultiQuery(ctx, query); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("error = %v", err)
			}
		})
	}

	if _, err := collection.MultiQuery(ctx, valid()); err != nil {
		t.Fatalf("zero defaults should be valid: %v", err)
	}
	explicitRRF := valid()
	explicitRRF.Reranker = NewRRFReranker()
	wantRRF, err := collection.MultiQuery(ctx, explicitRRF)
	if err != nil {
		t.Fatal(err)
	}
	for name, reranker := range map[string]Reranker{"nil": nil, "typed nil": (*testNilReranker)(nil)} {
		t.Run(name+" reranker", func(t *testing.T) {
			query := valid()
			query.Reranker = reranker
			got, err := collection.MultiQuery(ctx, query)
			if err != nil || !reflect.DeepEqual(got, wantRRF) {
				t.Fatalf("default RRF = %#v, %v; want %#v", got, err, wantRRF)
			}
		})
	}
	var nilCollection *Collection
	if _, err := nilCollection.MultiQuery(ctx, valid()); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection = %v", err)
	}
	if _, err := collection.MultiQuery(nil, valid()); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context = %v", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := collection.MultiQuery(canceled, valid()); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context = %v", err)
	}

	foreign := Document{PrimaryKey: "foreign", DocID: math.MaxUint64, Score: 1}
	duplicateQuery := valid()
	duplicateQuery.TopK = 2
	tooManyQuery := valid()
	tooManyQuery.TopK = 1
	boundaryTests := []struct {
		name  string
		query MultiQuery
		make  func([]RerankBatch) []Document
	}{
		{name: "foreign document", query: valid(), make: func([]RerankBatch) []Document { return []Document{foreign} }},
		{name: "duplicate document", query: duplicateQuery, make: func(b []RerankBatch) []Document { return []Document{b[0].Documents[0], b[0].Documents[0]} }},
		{name: "non-finite score", query: valid(), make: func(b []RerankBatch) []Document {
			value := b[0].Documents[0]
			value.Score = float32(math.NaN())
			return []Document{value}
		}},
		{name: "too many documents", query: tooManyQuery, make: func(b []RerankBatch) []Document { return []Document{b[0].Documents[0], b[0].Documents[1]} }},
		{name: "wrong primary key", query: valid(), make: func(b []RerankBatch) []Document {
			value := b[0].Documents[0]
			value.PrimaryKey = "forged"
			return []Document{value}
		}},
	}
	for _, testCase := range boundaryTests {
		t.Run(testCase.name, func(t *testing.T) {
			query := testCase.query
			query.Reranker = testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
				return testCase.make(batches), nil
			})
			if _, err := collection.MultiQuery(ctx, query); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("error = %v", err)
			}
		})
	}

	sentinel := errors.New("reranker failed")
	errorQuery := valid()
	errorQuery.Reranker = testRerankerFunc(func(context.Context, []RerankBatch, int) ([]Document, error) {
		return nil, sentinel
	})
	if _, err := collection.MultiQuery(ctx, errorQuery); !errors.Is(err, sentinel) {
		t.Fatalf("reranker error = %v", err)
	}
	cancelContext, cancelDuringRerank := context.WithCancel(ctx)
	cancelQuery := valid()
	cancelQuery.Reranker = testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
		cancelDuringRerank()
		return batches[0].Documents[:1], nil
	})
	if _, err := collection.MultiQuery(cancelContext, cancelQuery); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation after rerank = %v", err)
	}

	// Caller code must run without the Collection read lock: a write from the
	// reranker completes and does not enter the current candidate snapshot.
	writeQuery := valid()
	writeQuery.TopK = 1
	writeQuery.Reranker = testRerankerFunc(func(callbackContext context.Context, batches []RerankBatch, _ int) ([]Document, error) {
		_, err := collection.Insert(callbackContext, []Document{{PrimaryKey: "later", Fields: map[string]any{
			"title": "later", "category": "keep", "rating": int32(9),
			"embedding": VectorFP32{9, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{9}},
		}}})
		if err != nil {
			return nil, err
		}
		return batches[0].Documents[:1], nil
	})
	results, err := collection.MultiQuery(ctx, writeQuery)
	if err != nil || len(results) != 1 || results[0].PrimaryKey == "later" || collection.Stats().DocumentCount != 5 {
		t.Fatalf("write during rerank = %#v, %v, stats %#v", results, err, collection.Stats())
	}

	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.MultiQuery(ctx, valid()); !errors.Is(err, ErrFailedPrecondition) {
		t.Fatalf("closed collection = %v", err)
	}
}

func TestCollectionMultiQueryConcurrentSnapshotSearch(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "concurrent"), testMultiQuerySchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, testMultiQueryDocuments()); err != nil {
		t.Fatal(err)
	}
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 4},
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 4},
		},
		TopK: 3,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			return firstDistinctDocuments(batches, topK), nil
		}),
	}
	var wait sync.WaitGroup
	errorsFound := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 12; iteration++ {
				results, err := collection.MultiQuery(ctx, query)
				if err != nil {
					errorsFound <- err
					return
				}
				if got := documentKeys(results); !reflect.DeepEqual(got, []string{"c", "a", "b"}) {
					errorsFound <- errors.New("non-deterministic MultiQuery result")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsFound)
	for err := range errorsFound {
		t.Fatal(err)
	}
}

func TestMultiQueryPinnedCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/multi_query_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		BaselineCommit     string `json:"baseline_commit"`
		QueryHeaderHash    string `json:"query_header_sha256"`
		CollectionHash     string `json:"collection_source_sha256"`
		RerankerHeaderHash string `json:"reranker_header_sha256"`
		MaxTopK            int    `json:"max_topk"`
		MinimumSubQueries  int    `json:"minimum_sub_queries"`
		DefaultTopK        int    `json:"default_topk"`
		DefaultCandidates  int    `json:"default_num_candidates"`
		DefaultFTSOperator string `json:"default_fts_operator"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.QueryHeaderHash != "2c482b4c9832ffb07086e9789c88a4f7de6bc278c3f7ae901b4091e7acbdd193" ||
		fixture.CollectionHash != "cf4145fa9cbed9bf8975c440f024ce98359b2c6792f008e630db5da7f6422493" ||
		fixture.RerankerHeaderHash != "bc1949536968bc27f0cb11026d0ab8633dbb46641365455c20b433367837c7d6" ||
		fixture.MaxTopK != MaxQueryTopK || fixture.MinimumSubQueries != 2 ||
		fixture.DefaultTopK != DefaultMultiQueryTopK || fixture.DefaultCandidates != DefaultSubQueryCandidates ||
		fixture.DefaultFTSOperator != NewFTSQueryParams().DefaultOperator {
		t.Fatalf("compatibility fixture mismatch: %#v", fixture)
	}
}

func FuzzMultiQueryTargetKind(f *testing.F) {
	f.Add(uint8(0))
	f.Add(uint8(1))
	f.Add(uint8(2))
	f.Add(uint8(4))
	f.Add(uint8(7))
	f.Fuzz(func(t *testing.T, flags uint8) {
		flags &= 7
		query := SubQuery{}
		if flags&1 != 0 {
			query.DenseVector = VectorFP32{1}
		}
		if flags&2 != 0 {
			query.SparseVector = SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}}
		}
		if flags&4 != 0 {
			query.FTS = &FTSClause{Match: "x"}
		}
		_, err := multiQueryTargetKind(query)
		valid := flags == 1 || flags == 2 || flags == 4
		if (err == nil) != valid {
			t.Fatalf("flags %03b: err = %v", flags, err)
		}
	})
}

func BenchmarkV05HybridMultiQuery(b *testing.B) {
	ctx := context.Background()
	schema := testMultiQuerySchema()
	collection, err := CreateAndOpen(ctx, filepath.Join(b.TempDir(), "benchmark"), schema, NewCollectionOptions())
	if err != nil {
		b.Fatal(err)
	}
	defer collection.Close()
	documents := make([]Document, 256)
	for index := range documents {
		documents[index] = Document{PrimaryKey: "doc-" + benchmarkNumber(index), Fields: map[string]any{
			"title": "go vector search", "category": "keep", "rating": int32(index),
			"embedding": VectorFP32{float32(index) / 256, 1},
			"sparse":    SparseVectorFP32{Indices: []uint32{2}, Values: []float32{float32(index) / 256}},
		}}
	}
	if _, err := collection.Insert(ctx, documents); err != nil {
		b.Fatal(err)
	}
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 20},
			{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 20},
			{Field: "title", FTS: &FTSClause{Match: "go search"}, NumCandidates: 20},
		},
		TopK: 10,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			return firstDistinctDocuments(batches, topK), nil
		}),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, err := collection.MultiQuery(ctx, query); err != nil {
			b.Fatal(err)
		}
	}
}

func testMultiQuerySchema() CollectionSchema {
	fts := NewFTSIndexParams()
	schema := NewCollectionSchema("hybrid",
		FieldSchema{Name: "title", DataType: DataTypeString, Nullable: true, Index: fts},
		FieldSchema{Name: "category", DataType: DataTypeString, Index: NewInvertIndexParams()},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Nullable: true, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func testMultiQueryDocuments() []Document {
	return []Document{
		{PrimaryKey: "a", Fields: map[string]any{
			"title": "Go database", "category": "keep", "rating": int32(1),
			"embedding": VectorFP32{1, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}},
		}},
		{PrimaryKey: "b", Fields: map[string]any{
			"title": "Go Go search", "category": "keep", "rating": int32(2),
			"embedding": VectorFP32{0.8, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{3}},
		}},
		{PrimaryKey: "c", Fields: map[string]any{
			"title": "Go database search", "category": "drop", "rating": int32(3),
			"embedding": VectorFP32{2, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{5}},
		}},
		{PrimaryKey: "d", Fields: map[string]any{
			"title": nil, "category": "keep", "rating": int32(4),
			"embedding": VectorFP32{0.2, 0}, "sparse": nil,
		}},
	}
}

func assertHybridResults(t *testing.T, results []Document) {
	t.Helper()
	if got := documentKeys(results); !reflect.DeepEqual(got, []string{"b", "a"}) {
		t.Fatalf("hybrid keys = %#v", got)
	}
	if results[0].Score != 42 || results[1].Score != 41 {
		t.Fatalf("hybrid scores = %v, %v", results[0].Score, results[1].Score)
	}
	if !reflect.DeepEqual(results[0].Fields, map[string]any{"title": "Go Go search"}) ||
		!reflect.DeepEqual(results[1].Fields, map[string]any{"title": "Go database"}) {
		t.Fatalf("hybrid projection = %#v", results)
	}
}

func firstDistinctDocuments(batches []RerankBatch, topK int) []Document {
	seen := make(map[uint64]struct{})
	result := make([]Document, 0, topK)
	for _, batch := range batches {
		for _, document := range batch.Documents {
			if _, found := seen[document.DocID]; found {
				continue
			}
			seen[document.DocID] = struct{}{}
			result = append(result, document)
			if len(result) == topK {
				return result
			}
		}
	}
	return result
}

func benchmarkNumber(value int) string {
	const digits = "0123456789abcdef"
	return string([]byte{digits[(value>>8)&15], digits[(value>>4)&15], digits[value&15]})
}
