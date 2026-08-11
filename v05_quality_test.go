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
	"math"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestV05HybridReleaseReopenMatrix is the release-level behavior gate for all
// three retrieval sources and every public fusion strategy. Results must remain
// value-identical after the durable collection is reopened read-only.
func TestV05HybridReleaseReopenMatrix(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "v05-hybrid")
	collection, err := CreateAndOpen(ctx, path, testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}

	queries := []struct {
		name     string
		reranker Reranker
	}{
		{name: "default RRF"},
		{name: "explicit RRF", reranker: NewRRFReranker()},
		{name: "weighted", reranker: NewWeightedReranker(0.4, 0.2, 0.4)},
		{name: "callback", reranker: NewCallbackReranker(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			return firstDistinctDocuments(batches, topK), nil
		})},
	}
	makeQuery := func(reranker Reranker) MultiQuery {
		return MultiQuery{
			Queries: []SubQuery{
				{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
				{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 4},
				{Field: "title", FTS: &FTSClause{Query: `go AND (database OR search)`}, NumCandidates: 4},
			},
			TopK: 2, Filter: "category = 'keep'",
			Projection: Projection{OutputFields: []string{"title"}},
			Reranker:   reranker,
		}
	}
	run := func(handle *Collection) map[string][]Document {
		t.Helper()
		results := make(map[string][]Document, len(queries))
		for _, testCase := range queries {
			documents, queryErr := handle.MultiQuery(ctx, makeQuery(testCase.reranker))
			require.NoError(t, queryErr)
			require.Len(t, documents, 2)

			for _, document := range documents {
				require.False(t, document.PrimaryKey == "c")
				require.Len(t, document.Fields, 1)
				{
					score := float64(document.Score)
					require.False(t, math.IsNaN(score))
					require.False(t, math.IsInf(score, 0))
				}
			}
			results[testCase.name] = documents
		}
		require.Equal(t, results["explicit RRF"], results["default RRF"])

		return results
	}

	writableResults := run(collection)
	stats := collection.Stats()
	require.True(t, stats.DocumentCount == 4)
	require.True(t, stats.ImmutableSegments == 1)
	require.True(t, stats.MutableDocuments == 0)
	require.False(t, stats.StorageMemoryBytes == 0)
	require.True(t, stats.IndexCompleteness["title"] == 1)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	reopened, err := Open(ctx, path, options)
	require.NoError(t, err)

	defer func() { require.NoError(t, reopened.Close()) }()
	{
		reopenedResults := run(reopened)
		require.Equal(t, writableResults, reopenedResults)
	}
}
