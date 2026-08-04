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
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
)

type callbackContextKey struct{}

func TestCallbackRerankerDelegatesExactInputs(t *testing.T) {
	ctx := context.WithValue(context.Background(), callbackContextKey{}, "owned")
	batches := callbackTestBatches()
	want := []Document{batches[1].Documents[0], batches[0].Documents[0]}
	invocations := 0
	reranker := NewCallbackReranker(func(gotContext context.Context, gotBatches []RerankBatch, topK int) ([]Document, error) {
		invocations++
		if gotContext.Value(callbackContextKey{}) != "owned" || topK != 2 || !reflect.DeepEqual(gotBatches, batches) {
			t.Fatalf("callback inputs = %#v, %d", gotBatches, topK)
		}
		return want, nil
	})
	if err := reranker.Validate(); err != nil {
		t.Fatal(err)
	}
	got, err := reranker.Rerank(ctx, batches, 2)
	if err != nil || !reflect.DeepEqual(got, want) || invocations != 1 {
		t.Fatalf("callback output = %#v, %v, invocations %d", got, err, invocations)
	}

	zeroInvoked := false
	zero := NewCallbackReranker(func(_ context.Context, _ []RerankBatch, topK int) ([]Document, error) {
		zeroInvoked = true
		if topK != 0 {
			t.Fatalf("topK = %d", topK)
		}
		return []Document{}, nil
	})
	if got, err := zero.Rerank(context.Background(), nil, 0); err != nil || got == nil || !zeroInvoked {
		t.Fatalf("zero callback = %#v, %v, invoked %v", got, err, zeroInvoked)
	}
}

func TestCallbackRerankerValidationErrorsCancellationAndPanic(t *testing.T) {
	var empty CallbackReranker
	if err := empty.Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty validation = %v", err)
	}
	if _, err := empty.Rerank(context.Background(), nil, 1); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty callback = %v", err)
	}
	valid := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		return []Document{}, nil
	})
	if _, err := valid.Rerank(nil, nil, 1); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context = %v", err)
	}

	invoked := false
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	preCanceled := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		invoked = true
		return nil, nil
	})
	if _, err := preCanceled.Rerank(canceled, nil, 1); !errors.Is(err, context.Canceled) || invoked {
		t.Fatalf("pre-canceled callback = %v, invoked %v", err, invoked)
	}

	sentinel := errors.New("callback failed")
	failing := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		return nil, sentinel
	})
	if _, err := failing.Rerank(context.Background(), nil, 1); !errors.Is(err, sentinel) {
		t.Fatalf("callback error = %v", err)
	}

	panicSentinel := errors.New("callback panic")
	panicking := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		panic(panicSentinel)
	})
	if _, err := panicking.Rerank(context.Background(), nil, 1); !errors.Is(err, ErrInternal) || !errors.Is(err, panicSentinel) {
		t.Fatalf("panic error = %v", err)
	}
	panickingString := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		panic("bad callback")
	})
	if _, err := panickingString.Rerank(context.Background(), nil, 1); !errors.Is(err, ErrInternal) {
		t.Fatalf("string panic error = %v", err)
	}

	duringContext, cancelDuring := context.WithCancel(context.Background())
	canceling := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		cancelDuring()
		return []Document{}, nil
	})
	if _, err := canceling.Rerank(duringContext, nil, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("post-callback cancellation = %v", err)
	}
}

func TestCollectionMultiQueryCallbackReranker(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "callback"), testMultiQuerySchema(), NewCollectionOptions())
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
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 4},
		},
		TopK: 2, Filter: "category = 'keep'", Projection: Projection{OutputFields: []string{"title"}},
	}
	query.Reranker = NewCallbackReranker(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		if topK != 2 || batches[0].Field.Name != "embedding" || batches[1].Field.Name != "title" {
			t.Fatalf("callback batches = %#v, topK %d", batches, topK)
		}
		first := batches[1].Documents[0]
		second := batches[0].Documents[0]
		first.Score, second.Score = 9, 8
		return []Document{first, second}, nil
	})
	results, err := collection.MultiQuery(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"b", "a"}) || results[0].Score != 9 || results[1].Score != 8 {
		t.Fatalf("callback MultiQuery = %#v, %v", results, err)
	}
	for _, document := range results {
		if len(document.Fields) != 1 {
			t.Fatalf("callback projection = %#v", document)
		}
	}

	query.Reranker = NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		panic("collection callback panic")
	})
	if _, err := collection.MultiQuery(ctx, query); !errors.Is(err, ErrInternal) {
		t.Fatalf("collection callback panic = %v", err)
	}
}

func TestCallbackRerankerConcurrentUse(t *testing.T) {
	batches := callbackTestBatches()
	var invocations atomic.Int64
	reranker := NewCallbackReranker(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		invocations.Add(1)
		return firstDistinctDocuments(batches, topK), nil
	})
	want, err := reranker.Rerank(context.Background(), batches, 2)
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	errorsFound := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				got, err := reranker.Rerank(context.Background(), batches, 2)
				if err != nil {
					errorsFound <- err
					return
				}
				if !reflect.DeepEqual(got, want) {
					errorsFound <- errors.New("non-deterministic callback result")
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
	if invocations.Load() != 801 {
		t.Fatalf("invocations = %d", invocations.Load())
	}
}

func TestCallbackRerankerCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/callback_reranker_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		BaselineCommit string   `json:"baseline_commit"`
		HeaderHash     string   `json:"header_sha256"`
		SourceHash     string   `json:"source_sha256"`
		TestsHash      string   `json:"tests_sha256"`
		Arguments      []string `json:"arguments"`
		EmptyIsError   bool     `json:"empty_callback_is_error"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.HeaderHash != "bc1949536968bc27f0cb11026d0ab8633dbb46641365455c20b433367837c7d6" ||
		fixture.SourceHash != "3c93edc12303898af52911589c46c720072f9470446858fc36a61206d538aa1e" ||
		fixture.TestsHash != "05a03cacf74e7615661cec3153b2d2307f6a991510018386f6650f2625cb9a7d" ||
		!reflect.DeepEqual(fixture.Arguments, []string{"results", "fields", "topn"}) || !fixture.EmptyIsError {
		t.Fatalf("callback compatibility fixture mismatch: %#v", fixture)
	}
}

func FuzzCallbackRerankerPanicBoundary(f *testing.F) {
	f.Add(uint8(0), uint8(10))
	f.Add(uint8(1), uint8(0))
	f.Fuzz(func(t *testing.T, mode, topK uint8) {
		invoked := false
		reranker := NewCallbackReranker(func(_ context.Context, batches []RerankBatch, gotTopK int) ([]Document, error) {
			invoked = true
			if mode&1 != 0 {
				panic("fuzz callback")
			}
			if gotTopK == 0 {
				return []Document{}, nil
			}
			return batches[0].Documents[:1], nil
		})
		results, err := reranker.Rerank(context.Background(), callbackTestBatches(), int(topK))
		if !invoked {
			t.Fatal("callback was not invoked")
		}
		if mode&1 != 0 {
			if !errors.Is(err, ErrInternal) || results != nil {
				t.Fatalf("panic result = %#v, %v", results, err)
			}
		} else if err != nil || (topK == 0 && (results == nil || len(results) != 0)) || (topK != 0 && len(results) != 1) {
			t.Fatalf("callback result = %#v, %v", results, err)
		}
	})
}

func BenchmarkCallbackReranker(b *testing.B) {
	batches := callbackTestBatches()
	reranker := NewCallbackReranker(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		return firstDistinctDocuments(batches, topK), nil
	})
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		if _, err := reranker.Rerank(context.Background(), batches, 2); err != nil {
			b.Fatal(err)
		}
	}
}

func callbackTestBatches() []RerankBatch {
	return []RerankBatch{
		{Field: weightedVectorField("embedding", MetricTypeIP), Documents: []Document{
			{PrimaryKey: "a", DocID: 1, Score: 0.9},
			{PrimaryKey: "b", DocID: 2, Score: 0.8},
		}},
		{Field: FieldSchema{Name: "body", DataType: DataTypeString, Index: NewFTSIndexParams()}, Documents: []Document{
			{PrimaryKey: "b", DocID: 2, Score: 2},
			{PrimaryKey: "c", DocID: 3, Score: 1},
		}},
	}
}
