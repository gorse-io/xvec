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
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOptimizeFTSCompactsDeletesAndReopens(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize-fts")
	fts := FTSIndexParams{
		Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"},
		ExtraParams: `{"stemmer_lang":"english"}`,
	}
	schema := NewCollectionSchema("optimize_fts",
		FieldSchema{Name: "title", DataType: DataTypeString, Index: fts},
		FieldSchema{
			Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
			Index: NewFlatIndexParams(MetricTypeIP),
		},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "a", Fields: map[string]any{"title": "Go searching", "embedding": VectorFP32{1, 0}}},
			{PrimaryKey: "b", Fields: map[string]any{"title": "Database search", "embedding": VectorFP32{0.7, 0}}},
			{PrimaryKey: "remove", Fields: map[string]any{"title": "Go removed", "embedding": VectorFP32{0.9, 0}}},
		})
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"title": "Go optimized searching"}}})
		require.NoError(t, err)
	}
	{
		_, err := collection.Delete(ctx, []string{"remove"})
		require.NoError(t, err)
	}

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "optimized search"}, NumCandidates: 3},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 3},
		},
		TopK: 2, Projection: Projection{OutputFields: []string{"title"}},
	}
	want, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Len(t, want, 2)
	require.True(t, want[0].PrimaryKey == "a")

	before := collection.Stats()
	require.True(t, before.DeletedDocuments >= 2)
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	after := collection.Stats()
	require.True(t, after.DocumentCount == 2)
	require.True(t, after.DeletedDocuments == 0)
	require.True(t, after.MutableDocuments == 0)
	require.True(t, after.ImmutableSegments == 2)

	got, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	reopened, err := Open(ctx, path, options)
	require.NoError(t, err)

	defer reopened.Close()
	got, err = reopened.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)
}
