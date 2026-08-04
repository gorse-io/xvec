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
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestDropColumnRemovesPayloadsAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-column")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	inserted, err := collection.Insert(ctx, []Document{
		dropColumnDocument("a", 5, float64(1.25), true, []float32{3, 0}),
		dropColumnDocument("b", 3, nil, true, []float32{2, 0}),
		dropColumnDocument("c", 1, nil, false, []float32{1, 0}),
	})
	require.NoError(t, err)

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"rating": int32(4)}}})
	require.NoError(t, err)

	wantIDs := map[string]uint64{"a": inserted[0].DocID, "b": updated[0].DocID, "c": inserted[2].DocID}
	{
		err := collection.DropColumn(ctx, "rating")
		require.NoError(t, err)
	}
	{
		_, found := collection.Schema().Field("rating")
		require.False(t, found,
			"dropped indexed field remains in schema")
	}

	assertStoredFieldAbsent(t, ctx, collection, "rating", wantIDs)
	{
		_, err := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 3, Filter: "rating >= 2",
		})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.DropColumn(ctx, "optional")
		require.NoError(t, err)
	}

	assertStoredFieldAbsent(t, ctx, collection, "optional", wantIDs)
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 3,
	})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b", "c"}, documentKeys(results))

	for index := range results {
		{
			_, found := results[index].Fields["rating"]
			require.False(t, found)
		}
		{
			_, found := results[index].Fields["optional"]
			require.False(t, found)
		}
	}
	require.True(t, collection.Stats().DocumentCount == 3)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	assertStoredFieldAbsent(t, ctx, collection, "rating", wantIDs)
	assertStoredFieldAbsent(t, ctx, collection, "optional", wantIDs)
	withDropped := dropColumnDocument("bad", 2, nil, false, []float32{1, 0})
	{
		_, err := collection.Insert(ctx, []Document{withDropped})
		require.Error(t, err,
			"insert containing dropped fields succeeded")
	}

	valid := Document{PrimaryKey: "d", Fields: map[string]any{
		"text": "d", "embedding": VectorFP32{0.5, 0},
	}}
	{
		_, err := collection.Insert(ctx, []Document{valid})
		require.NoError(t, err)
	}
}

func TestDropColumnValidationAndPublicationRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.DropColumn(ctx, "rating")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "drop-column-errors")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{dropColumnDocument("one", 2, nil, false, []float32{1, 0})})
		require.NoError(t, err)
	}
	{
		err := collection.DropColumn(nil, "rating")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	initialSchema := collection.Schema()
	initialGeneration := collection.store.Manifest().Generation
	for _, column := range []string{"", "missing", "text", "embedding"} {
		t.Run(column, func(t *testing.T) {
			{
				err := collection.DropColumn(ctx, column)
				require.Error(t, err,
					"DropColumn succeeded")
			}
			require.Equal(t, initialSchema, collection.Schema())
			require.Equal(t, initialGeneration, collection.store.Manifest().Generation)
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.DropColumn(canceled, "rating")
		require.ErrorIs(t, err, context.Canceled)
	}

	versionLock, err := ailego.AcquireFileLock(ctx, filepath.Join(path, ".version.lock"), ailego.LockExclusive)
	require.NoError(t, err)

	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.DropColumn(deadline, "rating")
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		_ = versionLock.Close()
	}
	require.ErrorIs(t, err, context.DeadlineExceeded)
	{
		err := versionLock.Close()
		require.NoError(t, err)
	}
	require.Equal(t, initialSchema, collection.Schema())
	require.Equal(t, initialGeneration, collection.store.Manifest().Generation)

	fetched, err := collection.Fetch(ctx, []string{"one"}, Projection{})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, int32(2), fetched[0].Fields["rating"])
}

func TestDropColumnRejectsLastFieldAndReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	t.Run("last field", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "drop-last-column")
		schema := NewCollectionSchema("drop_last", FieldSchema{Name: "only", DataType: DataTypeInt32})
		schema.MaxDocsPerSegment = MinMaxDocsPerSegment
		collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
		require.NoError(t, err)

		defer collection.Close()
		{
			err := collection.DropColumn(ctx, "only")
			require.ErrorIs(t, err, ErrInvalidArgument)
		}
	})
	t.Run("read only", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "drop-column-read-only")
		collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
		require.NoError(t, err)
		{
			err := collection.Close()
			require.NoError(t, err)
		}

		options := NewCollectionOptions()
		options.ReadOnly = true
		collection, err = Open(ctx, path, options)
		require.NoError(t, err)

		defer collection.Close()
		{
			err := collection.DropColumn(ctx, "rating")
			require.ErrorIs(t, err, ErrPermissionDenied)
		}
	})
}

func TestDropColumnEmptyCollectionPublishesSchemaOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-column-empty")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	initialGeneration := collection.store.Manifest().Generation
	{
		err := collection.DropColumn(ctx, "rating")
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > initialGeneration)
	require.Nil(t, collection.store.Manifest().PersistedSegments)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, found := collection.Schema().Field("rating")
		require.False(t, found,
			"reopened schema contains dropped field")
	}
}

func assertStoredFieldAbsent(t *testing.T, ctx context.Context, collection *Collection, field string, wantIDs map[string]uint64) {
	t.Helper()
	stored, err := collection.store.LiveDocuments(ctx)
	require.NoError(t, err)

	for _, item := range stored {
		require.Equal(t, wantIDs[item.PrimaryKey], item.DocID)

		fields, decodeErr := unmarshalDocumentPayload(item.Payload)
		require.NoError(t, decodeErr)
		{
			_, found := fields[field]
			require.False(t, found)
		}
	}
}

func dropColumnSchema() CollectionSchema {
	schema := NewCollectionSchema("drop_columns",
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Index: NewInvertIndexParams()},
		FieldSchema{Name: "optional", DataType: DataTypeDouble, Nullable: true},
		FieldSchema{Name: "text", DataType: DataTypeString},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func dropColumnDocument(primaryKey string, rating int32, optional any, includeOptional bool, embedding []float32) Document {
	fields := map[string]any{
		"rating": rating, "text": primaryKey, "embedding": VectorFP32(embedding),
	}
	if includeOptional {
		fields["optional"] = optional
	}
	return Document{PrimaryKey: primaryKey, Fields: fields}
}
