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

func TestDropScalarIndexPublishesAndPreservesForwardResults(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-scalar-index")
	schema := createIndexSchema()
	schema.Fields[1].Index = NewInvertIndexParams()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{
			createIndexDocument("a", "alpha", 1, 3),
			createIndexDocument("b", "beta", 2, 2),
			createIndexDocument("c", "gamma", 3, 1),
		})
		require.NoError(t, err)
	}

	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating>=2"}
	before, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(before))

	generation := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "rating")
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > generation,
		"DropIndex did not publish a manifest generation")

	rating, _ := collection.Schema().Field("rating")
	require.Nil(t, rating.Index)
	require.Equal(t, IndexTypeUndefined, rating.IndexType())

	after, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(after))

	idempotentGeneration := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "rating")
		require.NoError(t, err)
	}
	require.Equal(t, idempotentGeneration, collection.store.Manifest().Generation,
		"idempotent scalar DropIndex advanced generation")
	{
		err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "rating")
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	rating, _ = collection.Schema().Field("rating")
	require.Nil(t, rating.Index)
	require.True(t, collection.Stats().DocumentCount == 3)
}

func TestDropVectorIndexRestoresDefaultFlatIP(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-vector-index")
	schema := NewCollectionSchema("drop_vector",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeL2)},
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
	require.Equal(t, []string{"near", "far"}, documentKeys(before))
	{
		err := collection.DropIndex(ctx, "embedding")
		require.NoError(t, err)
	}

	field, _ := collection.Schema().Field("embedding")
	flat, ok := field.Index.(FlatIndexParams)
	require.True(t, ok)
	require.Equal(t, MetricTypeIP, flat.Metric)
	require.Equal(t, QuantizeTypeUndefined, flat.Quantize)

	after, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"far", "near"}, documentKeys(after))

	generation := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "embedding")
		require.NoError(t, err)
	}
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"idempotent vector DropIndex advanced generation")
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	field, _ = collection.Schema().Field("embedding")
	{
		flat, ok = field.Index.(FlatIndexParams)
		require.True(t, ok)
		require.Equal(t, MetricTypeIP, flat.Metric)
	}
}

func TestDropUnsupportedOrFTSIndexRemovesMetadata(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-later-index")
	schema := NewCollectionSchema("drop_later",
		FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{
			"text": "hello", "embedding": VectorFP32{1, 0},
		}}})
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "text")
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "embedding")
		require.NoError(t, err)
	}

	text, _ := collection.Schema().Field("text")
	embedding, _ := collection.Schema().Field("embedding")
	require.Nil(t, text.Index)
	require.Equal(t, IndexTypeFlat, embedding.IndexType())

	results, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.True(t, results[0].PrimaryKey == "a")
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	text, _ = collection.Schema().Field("text")
	embedding, _ = collection.Schema().Field("embedding")
	require.Nil(t, text.Index)
	require.Equal(t, IndexTypeFlat, embedding.IndexType())
}

func TestDropIndexValidationLifecycleAndRollback(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-index-errors")
	collection, err := CreateAndOpen(ctx, path, createIndexSchema(), NewCollectionOptions())
	require.NoError(t, err)

	generation := collection.store.Manifest().Generation
	{
		err := collection.DropIndex(ctx, "title")
		require.NoError(t, err)
	}
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"unindexed scalar no-op advanced generation")

	for _, testCase := range []struct {
		name   string
		column string
		want   error
	}{
		{"empty", "", ErrInvalidArgument},
		{"missing", "missing", ErrNotFound},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			before := collection.Schema()
			beforeGeneration := collection.store.Manifest().Generation
			{
				err := collection.DropIndex(ctx, testCase.column)
				require.ErrorIs(t, err, testCase.want)
			}
			require.Equal(t, before, collection.Schema(),
				"failed DropIndex changed schema")
			require.Equal(t, beforeGeneration, collection.store.Manifest().Generation,
				"failed DropIndex changed schema")
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.DropIndex(canceled, "embedding")
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := collection.DropIndex(nil, "embedding")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	var nilCollection *Collection
	{
		err := nilCollection.DropIndex(ctx, "embedding")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.DropIndex(ctx, "embedding")
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		err := readOnly.DropIndex(ctx, "embedding")
		require.ErrorIs(t, err, ErrPermissionDenied)
	}

	corrupt, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "drop-index-rollback"), NewCollectionSchema("drop_rollback",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeL2)},
	), NewCollectionOptions())
	require.NoError(t, err)

	defer corrupt.Close()
	{
		_, err := corrupt.store.Insert(ctx, []db.WriteInput{{PrimaryKey: "corrupt", Payload: []byte("bad")}})
		require.NoError(t, err)
	}

	before := corrupt.Schema()
	beforeGeneration := corrupt.store.Manifest().Generation
	{
		err := corrupt.DropIndex(ctx, "embedding")
		require.Error(t, err,
			"DropIndex accepted corrupt vector backfill")
	}
	require.Equal(t, before, corrupt.Schema(),
		"failed vector DropIndex changed schema")
	require.Equal(t, beforeGeneration, corrupt.store.Manifest().Generation,
		"failed vector DropIndex changed schema")
}
