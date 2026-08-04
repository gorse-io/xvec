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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeleteByFilterAcrossSegmentsAndWALRecovery(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "delete-filter")
	schema := deleteFilterSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	first := []Document{
		deleteFilterDocument("a", "alpha", int32(1), StringArray{"red"}, 5),
		deleteFilterDocument("b", "beta", int32(2), StringArray{"blue"}, 4),
		deleteFilterDocument("c", "gamma", nil, StringArray{"red", "blue"}, 3),
	}
	{
		_, err := collection.Insert(ctx, first)
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}

	second := []Document{
		deleteFilterDocument("d", "delta", int32(4), StringArray{}, 2),
		deleteFilterDocument("e", "omega", int32(5), nil, 1),
	}
	{
		_, err := collection.Insert(ctx, second)
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating >")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 5)
	}
	{
		err := collection.DeleteByFilter(ctx, "(rating>=2 AND title LIKE 'd%') OR tags CONTAIN_ALL ('red', 'blue')")
		require.NoError(t, err)
	}
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 3)
	}

	assertFetchedPresence(t, collection, []string{"a", "b", "c", "d", "e"}, []bool{true, true, false, false, true})
	{
		err := collection.DeleteByFilter(ctx, "rating>100")
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "title LIKE '%ta'")
		require.NoError(t, err)
	}
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 2)
	}
	{
		// Close without Flush: both immutable-segment and writing-segment deletes
		// must be reconstructed from the WAL.
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	assertFetchedPresence(t, collection, []string{"a", "b", "c", "d", "e"}, []bool{true, false, false, false, true})
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 2)
	}
	{
		err := collection.DeleteByFilter(ctx, "title LIKE '%'")
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	collection, err = Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer collection.Close()
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 0)
	}
}

func TestDeleteByFilterUsesOnlyCurrentDocumentVersions(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "delete-current"), deleteFilterSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{deleteFilterDocument("versioned", "before", int32(1), StringArray{"old"}, 1)})
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Update(ctx, []Document{{PrimaryKey: "versioned", Fields: map[string]any{
			"title": "after", "rating": int32(3), "tags": StringArray{"new"},
		}}})
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating=1 OR tags CONTAIN_ANY ('old')")
		require.NoError(t, err)
	}

	assertFetchedPresence(t, collection, []string{"versioned"}, []bool{true})
	{
		err := collection.DeleteByFilter(ctx, "rating=3 AND tags CONTAIN_ANY ('new')")
		require.NoError(t, err)
	}

	assertFetchedPresence(t, collection, []string{"versioned"}, []bool{false})
}

func TestDeleteByFilterValidationCancellationAndLifecycle(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.DeleteByFilter(ctx, "rating=1")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "delete-filter-errors")
	collection, err := CreateAndOpen(ctx, path, deleteFilterSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.DeleteByFilter(nil, "rating=1")
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	for _, filter := range []string{"", "   ", "missing=1", "embedding=1"} {
		{
			err := collection.DeleteByFilter(ctx, filter)
			assert.ErrorIs(t, err, ErrInvalidArgument)
		}
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.DeleteByFilter(canceled, "rating=1")
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating=1")
		require.NoError(t, err)
	}
	{
		_, err := collection.Insert(ctx, []Document{deleteFilterDocument("a", "alpha", int32(1), StringArray{}, 1)})
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.DeleteByFilter(ctx, "rating=1")
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		err := readOnly.DeleteByFilter(ctx, "rating=1")
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
	{
		got := readOnly.Stats().DocumentCount
		require.True(t, got == 1)
	}
}

func deleteFilterSchema() CollectionSchema {
	extended := NewInvertIndexParams()
	extended.EnableExtendedWildcard = true
	schema := NewCollectionSchema("delete_filter",
		FieldSchema{Name: "title", DataType: DataTypeString, Index: extended},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "tags", DataType: DataTypeArrayString, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func deleteFilterDocument(primaryKey, title string, rating, tags any, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "tags": tags, "embedding": VectorFP32{score, 0},
	}}
}

func assertFetchedPresence(t *testing.T, collection *Collection, keys []string, present []bool) {
	t.Helper()
	fetched, err := collection.Fetch(context.Background(), keys, Projection{})
	require.NoError(t, err)
	require.Len(t, fetched, len(present))

	for index := range fetched {
		assert.Equal(t, present[index], fetched[index] != nil)
	}
}
