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

func TestCreateFTSIndexBackfillQueryAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "fts-backfill")
	schema := NewCollectionSchema("fts_backfill",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{
			Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
			Index: NewFlatIndexParams(MetricTypeIP),
		},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "go", Fields: map[string]any{"title": "Go vector search", "embedding": VectorFP32{1, 0}}},
			{PrimaryKey: "db", Fields: map[string]any{"title": "Database internals", "embedding": VectorFP32{0.5, 0}}},
		})
		require.NoError(t, err)
	}

	params := FTSIndexParams{Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"}, ExtraParams: `{"stemmer_lang":"english"}`}
	{
		err := collection.CreateIndex(ctx, "title", params, CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	field, found := collection.Schema().Field("title")
	require.True(t, found)
	require.True(t, equalIndexParams(field.Index, params))
	require.True(t, collection.Stats().IndexCompleteness["title"] == 1)

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "searching"}, NumCandidates: 2},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 2},
		},
		TopK: 1, Projection: Projection{OutputFields: []string{"title"}},
	}
	want, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Len(t, want, 1)
	require.True(t, want[0].PrimaryKey == "go")
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	reopened, err := Open(ctx, path, options)
	require.NoError(t, err)

	defer reopened.Close()
	got, err := reopened.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestCreateFTSIndexBackfillFailureRollsBack(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "fts-rollback")
	schema := NewCollectionSchema("fts_rollback", FieldSchema{Name: "title", DataType: DataTypeString})
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"title": "中文"}}})
		require.NoError(t, err)
	}

	beforeSchema := collection.Schema()
	beforeGeneration := collection.store.Manifest().Generation
	params := FTSIndexParams{Tokenizer: "jieba", ExtraParams: `{"jieba_dict_dir":"missing-jieba-resources"}`}
	{
		err := collection.CreateIndex(ctx, "title", params, CreateIndexOptions{})
		require.Error(t, err,
			"missing Jieba resources unexpectedly succeeded")
	}
	require.Equal(t, beforeSchema, collection.Schema(),
		"failed FTS backfill changed schema or manifest")
	require.Equal(t, beforeGeneration, collection.store.Manifest().Generation,
		"failed FTS backfill changed schema or manifest")
}
