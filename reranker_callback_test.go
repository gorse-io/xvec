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

package xvec

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type callbackContextKey struct{}

func TestCallbackRerankerDelegatesExactInputs(t *testing.T) {
	ctx := context.WithValue(context.Background(), callbackContextKey{}, "owned")
	batches := callbackTestBatches()
	want := []Document{batches[1].Documents[0], batches[0].Documents[0]}
	invocations := 0
	reranker := NewCallbackReranker(func(gotContext context.Context, gotBatches []RerankBatch, topK int) ([]Document, error) {
		invocations++
		require.True(t, gotContext.Value(callbackContextKey{}) == "owned")
		require.True(t, topK == 2)
		require.Equal(t, batches, gotBatches)

		return want, nil
	})
	{
		err := reranker.Validate()
		require.NoError(t, err)
	}

	got, err := reranker.Rerank(ctx, batches, 2)
	require.NoError(t, err)
	require.Equal(t, want, got)
	require.True(t, invocations == 1)

	zeroInvoked := false
	zero := NewCallbackReranker(func(_ context.Context, _ []RerankBatch, topK int) ([]Document, error) {
		zeroInvoked = true
		require.True(t, topK == 0)

		return []Document{}, nil
	})
	{
		got, err := zero.Rerank(context.Background(), nil, 0)
		require.NoError(t, err)
		require.NotNil(t, got)
		require.True(t, zeroInvoked)
	}
}

func TestCallbackRerankerValidationErrorsCancellationAndPanic(t *testing.T) {
	var empty CallbackReranker
	{
		err := empty.Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := empty.Rerank(context.Background(), nil, 1)
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	valid := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		return []Document{}, nil
	})
	{
		_, err := valid.Rerank(nil, nil, 1)
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	invoked := false
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	preCanceled := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		invoked = true
		return nil, nil
	})
	{
		_, err := preCanceled.Rerank(canceled, nil, 1)
		require.ErrorIs(t, err, context.Canceled)
		require.False(t, invoked)
	}

	sentinel := errors.New("callback failed")
	failing := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		return nil, sentinel
	})
	{
		_, err := failing.Rerank(context.Background(), nil, 1)
		require.ErrorIs(t, err, sentinel)
	}

	panicSentinel := errors.New("callback panic")
	panicking := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		panic(panicSentinel)
	})
	{
		_, err := panicking.Rerank(context.Background(), nil, 1)
		require.ErrorIs(t, err, ErrInternal)
		require.ErrorIs(t, err, panicSentinel)
	}

	panickingString := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		panic("bad callback")
	})
	{
		_, err := panickingString.Rerank(context.Background(), nil, 1)
		require.ErrorIs(t, err, ErrInternal)
	}

	duringContext, cancelDuring := context.WithCancel(context.Background())
	canceling := NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		cancelDuring()
		return []Document{}, nil
	})
	{
		_, err := canceling.Rerank(duringContext, nil, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestCollectionMultiQueryCallbackReranker(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "callback"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer func() { require.NoError(t, collection.Close()) }()
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 4},
		},
		TopK: 2, Filter: "category = 'keep'", Projection: Projection{OutputFields: []string{"title"}},
	}
	query.Reranker = NewCallbackReranker(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		require.True(t, topK == 2)
		require.True(t, batches[0].Field.Name == "embedding")
		require.True(t, batches[1].Field.Name == "title")

		first := batches[1].Documents[0]
		second := batches[0].Documents[0]
		first.Score, second.Score = 9, 8
		return []Document{first, second}, nil
	})
	results, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(results))
	require.True(t, results[0].Score == 9)
	require.True(t, results[1].Score == 8)

	for _, document := range results {
		require.Len(t, document.Fields, 1)
	}

	query.Reranker = NewCallbackReranker(func(context.Context, []RerankBatch, int) ([]Document, error) {
		panic("collection callback panic")
	})
	{
		_, err := collection.MultiQuery(ctx, query)
		require.ErrorIs(t, err, ErrInternal)
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
	require.NoError(t, err)

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
				if !assert.Equal(t, want, got) {
					errorsFound <- errors.New("non-deterministic callback result")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsFound)
	for err := range errorsFound {
		require.NoError(t, err)
	}
	require.True(t, invocations.Load() == 801)
}

func TestCallbackRerankerCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/callback_reranker_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		BaselineCommit string   `json:"baseline_commit"`
		HeaderHash     string   `json:"header_sha256"`
		SourceHash     string   `json:"source_sha256"`
		TestsHash      string   `json:"tests_sha256"`
		Arguments      []string `json:"arguments"`
		EmptyIsError   bool     `json:"empty_callback_is_error"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.HeaderHash == "bc1949536968bc27f0cb11026d0ab8633dbb46641365455c20b433367837c7d6")
	require.True(t, fixture.SourceHash == "3c93edc12303898af52911589c46c720072f9470446858fc36a61206d538aa1e")
	require.True(t, fixture.TestsHash == "05a03cacf74e7615661cec3153b2d2307f6a991510018386f6650f2625cb9a7d")
	require.Equal(t, []string{"results", "fields", "topn"}, fixture.Arguments)
	require.True(t, fixture.EmptyIsError)
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
		require.True(t, invoked,
			"callback was not invoked")

		if mode&1 != 0 {
			require.ErrorIs(t, err, ErrInternal)
			require.Nil(t, results)
		} else {
			require.NoError(t, err)
			if topK == 0 {
				require.NotNil(t, results)
				require.Len(t, results, 0)
			} else {
				require.Len(t, results, 1)
			}
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
		{
			_, err := reranker.Rerank(context.Background(), batches, 2)
			if err != nil {
				require.NoError(b, err)
			}
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
