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

	"github.com/gorse-io/zvec/internal/db"
	"github.com/stretchr/testify/require"
)

func TestCreateScalarIndexPublishesSchemaAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-scalar-index")
	schema := createIndexSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			createIndexDocument("a", "alpha", 1, 3),
			createIndexDocument("b", "alphabet", 2, 2),
			createIndexDocument("c", "beta", 3, 1),
		})
		require.NoError(t, err)
	}

	initialGeneration := collection.store.Manifest().Generation
	params := NewInvertIndexParams()
	{
		err := collection.CreateIndex(ctx, "rating", &params, CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	createdGeneration := collection.store.Manifest().Generation
	require.True(t, createdGeneration > initialGeneration)

	rating, _ := collection.Schema().Field("rating")
	require.True(t, equalIndexParams(rating.Index, NewInvertIndexParams()))

	params.EnableExtendedWildcard = true
	rating, _ = collection.Schema().Field("rating")
	stored := rating.Index.(InvertIndexParams)
	require.False(t, stored.EnableExtendedWildcard,
		"CreateIndex retained caller-owned parameter pointer")

	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating>=2",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(results))
	{
		err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, createdGeneration, got)
	}

	changed := NewInvertIndexParams()
	changed.EnableRangeOptimization = false
	{
		err := collection.CreateIndex(ctx, "rating", changed, CreateIndexOptions{Concurrency: 1})
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > createdGeneration,
		"changed index parameters did not publish a generation")

	extended := NewInvertIndexParams()
	extended.EnableExtendedWildcard = true
	{
		err := collection.CreateIndex(ctx, "title", extended, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	results, err = collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "title LIKE '%bet'",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b"}, documentKeys(results))
	{
		// The schema manifest and existing write WAL are independently durable;
		// neither needs a Flush before reopening.
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	rating, _ = collection.Schema().Field("rating")
	title, _ := collection.Schema().Field("title")
	require.NotNil(t, rating.Index)
	require.False(t, rating.Index.(InvertIndexParams).EnableRangeOptimization)
	require.NotNil(t, title.Index)
	require.True(t, title.Index.(InvertIndexParams).EnableExtendedWildcard)
	require.True(t, collection.Stats().DocumentCount == 3)
}

func TestCreateFlatIndexChangesMetricAtomically(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-flat-index")
	schema := NewCollectionSchema("create_flat",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "far", Fields: map[string]any{"embedding": VectorFP32{10, 0}}},
			{PrimaryKey: "near", Fields: map[string]any{"embedding": VectorFP32{1, 0}}},
		})
		require.NoError(t, err)
	}

	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2}
	before, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"far", "near"}, documentKeys(before))
	{
		err := collection.CreateIndex(ctx, "embedding", NewFlatIndexParams(MetricTypeL2), CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	after, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"near", "far"}, documentKeys(after))
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	after, err = collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"near", "far"}, documentKeys(after))
}

func TestCreateIndexValidationAndRollback(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-index-errors")
	schema := NewCollectionSchema("index_errors",
		FieldSchema{Name: "text", DataType: DataTypeString},
		FieldSchema{Name: "already_fts", DataType: DataTypeString, Index: NewFTSIndexParams()},
		FieldSchema{Name: "binary", DataType: DataTypeBinary},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	var typedNil *InvertIndexParams
	invalidFlat := NewFlatIndexParams(MetricTypeUndefined)
	tests := []struct {
		name    string
		column  string
		index   IndexParams
		options CreateIndexOptions
		want    error
	}{
		{"empty-column", "", NewInvertIndexParams(), CreateIndexOptions{}, ErrInvalidArgument},
		{"nil-index", "text", nil, CreateIndexOptions{}, ErrInvalidArgument},
		{"typed-nil-index", "text", typedNil, CreateIndexOptions{}, ErrInvalidArgument},
		{"negative-concurrency", "text", NewInvertIndexParams(), CreateIndexOptions{Concurrency: -1}, ErrInvalidArgument},
		{"missing-field", "missing", NewInvertIndexParams(), CreateIndexOptions{}, ErrNotFound},
		{"invert-vector", "embedding", NewInvertIndexParams(), CreateIndexOptions{}, ErrInvalidArgument},
		{"flat-scalar", "text", NewFlatIndexParams(MetricTypeIP), CreateIndexOptions{}, ErrInvalidArgument},
		{"scalar-index-conflict", "already_fts", NewInvertIndexParams(), CreateIndexOptions{}, ErrNotSupported},
		{"invalid-index-params", "embedding", invalidFlat, CreateIndexOptions{}, ErrInvalidArgument},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			beforeSchema := collection.Schema()
			beforeGeneration := collection.store.Manifest().Generation
			err := collection.CreateIndex(ctx, testCase.column, testCase.index, testCase.options)
			require.ErrorIs(t, err, testCase.want)
			require.Equal(t, beforeSchema, collection.Schema(),
				"failed CreateIndex changed schema or manifest")
			require.Equal(t, beforeGeneration, collection.store.Manifest().Generation,
				"failed CreateIndex changed schema or manifest")
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.CreateIndex(canceled, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := collection.CreateIndex(nil, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	var nilCollection *Collection
	{
		err := nilCollection.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		err := readOnly.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{})
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
}

func TestCreateIndexBackfillFailureLeavesSchemaUnchanged(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "create-index-rollback"), createIndexSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.store.Insert(ctx, []db.WriteInput{{PrimaryKey: "corrupt", Payload: []byte("not-a-document")}})
		require.NoError(t, err)
	}

	before := collection.Schema()
	generation := collection.store.Manifest().Generation
	{
		err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2})
		require.Error(t, err,
			"CreateIndex unexpectedly accepted corrupt backfill data")
	}
	require.Equal(t, before, collection.Schema(),
		"failed backfill changed published schema")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed backfill changed published schema")
}

func createIndexSchema() CollectionSchema {
	schema := NewCollectionSchema("create_index",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func createIndexDocument(primaryKey, title string, rating int32, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "embedding": VectorFP32{score, 0},
	}}
}
