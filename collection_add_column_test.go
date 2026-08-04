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

func TestAddColumnBackfillsAtomicallyAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column")
	schema := addColumnSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		addColumnDocument("a", 1, []float32{3, 0}),
		addColumnDocument("b", 2, []float32{2, 0}),
		addColumnDocument("c", 3, []float32{1, 0}),
	}
	inserted, err := collection.Insert(ctx, documents)
	require.NoError(t, err)

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"count": int32(4)}}})
	require.NoError(t, err)

	beforeIDs := map[string]uint64{"a": inserted[0].DocID, "c": inserted[2].DocID, "b": updated[0].DocID}
	initialGeneration := collection.store.Manifest().Generation
	index := NewInvertIndexParams()
	index.EnableRangeOptimization = true
	field := FieldSchema{Name: "derived", DataType: DataTypeInt64, Index: index}
	{
		err := collection.AddColumn(ctx, field, "(count * 2) + 1", AddColumnOptions{Concurrency: 3})
		require.NoError(t, err)
	}
	require.True(t, collection.store.Manifest().Generation > initialGeneration,
		"AddColumn did not publish a new manifest generation")
	require.True(t, collection.Stats().DocumentCount == 3)

	fetched, err := collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)

	wantDerived := []int64{3, 9, 7}
	for index, document := range fetched {
		require.NotNil(t, document)
		require.Equal(t, beforeIDs[document.PrimaryKey], document.DocID)
		require.Equal(t, wantDerived[index], document.Fields["derived"])
	}
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "derived >= 7", Projection: Projection{OutputFields: []string{"derived"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(results))
	{
		err := collection.AddColumn(ctx, FieldSchema{Name: "optional", DataType: DataTypeFloat, Nullable: true}, "", AddColumnOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{OutputFields: []string{"optional"}})
	require.NoError(t, err)

	for _, document := range fetched {
		value, found := document.Fields["optional"]
		require.True(t, found)
		require.Nil(t, value)
		require.Equal(t, beforeIDs[document.PrimaryKey], document.DocID)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)

	for index, document := range fetched {
		require.NotNil(t, document)
		require.Equal(t, wantDerived[index], document.Fields["derived"])
		{
			value, found := document.Fields["optional"]
			require.True(t, found)
			require.Nil(t, value)
		}
	}
	missing := addColumnDocument("missing", 5, []float32{1, 0})
	{
		_, err := collection.Insert(ctx, []Document{missing})
		require.Error(t, err,
			"insert without added non-nullable field succeeded")
	}

	missing.Fields["derived"] = int64(11)
	missing.Fields["optional"] = nil
	{
		_, err := collection.Insert(ctx, []Document{missing})
		require.NoError(t, err)
	}
}

func TestAddColumnValidationAndFailureRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.AddColumn(ctx, FieldSchema{}, "", AddColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "add-column-errors")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		err := collection.AddColumn(nil, FieldSchema{Name: "nil_ctx", DataType: DataTypeInt32}, "1", AddColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := collection.Insert(ctx, []Document{addColumnDocument("one", 2, []float32{1, 0})})
		require.NoError(t, err)
	}

	initialGeneration := collection.store.Manifest().Generation
	tests := []struct {
		name       string
		field      FieldSchema
		expression string
		options    AddColumnOptions
	}{
		{"unsupported type", FieldSchema{Name: "text", DataType: DataTypeString, Nullable: true}, "", AddColumnOptions{}},
		{"non-nullable without expression", FieldSchema{Name: "required", DataType: DataTypeInt32}, "", AddColumnOptions{}},
		{"duplicate", FieldSchema{Name: "count", DataType: DataTypeInt32}, "count", AddColumnOptions{}},
		{"missing reference", FieldSchema{Name: "missing_ref", DataType: DataTypeInt32}, "missing + 1", AddColumnOptions{}},
		{"syntax", FieldSchema{Name: "syntax", DataType: DataTypeInt32}, "count +", AddColumnOptions{}},
		{"evaluation", FieldSchema{Name: "divide", DataType: DataTypeInt32}, "count / 0", AddColumnOptions{}},
		{"negative concurrency", FieldSchema{Name: "workers", DataType: DataTypeInt32}, "1", AddColumnOptions{Concurrency: -1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			{
				err := collection.AddColumn(ctx, test.field, test.expression, test.options)
				require.Error(t, err,
					"AddColumn succeeded")
			}
			{
				_, found := collection.Schema().Field(test.field.Name)
				require.False(t, found && test.field.Name != "count")
			}
			{
				got := collection.store.Manifest().Generation
				require.Equal(t, initialGeneration, got)
			}
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.AddColumn(canceled, FieldSchema{Name: "canceled", DataType: DataTypeInt32}, "1", AddColumnOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}

	fetched, err := collection.Fetch(ctx, []string{"one"}, Projection{IncludeVectors: true})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, addColumnDocument("one", 2, []float32{1, 0}).Fields, fetched[0].Fields)
}

func TestAddColumnEmptyCollectionMatchesDeferredExpressionBehavior(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column-empty")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.AddColumn(ctx, FieldSchema{Name: "deferred", DataType: DataTypeInt32}, "CASE WHEN count > 0 THEN 1 END", AddColumnOptions{})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	document := addColumnDocument("one", 1, []float32{1, 0})
	document.Fields["deferred"] = int32(7)
	{
		_, err := collection.Insert(ctx, []Document{document})
		require.NoError(t, err)
	}
}

func TestAddColumnRejectsReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column-read-only")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
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
	err = collection.AddColumn(ctx, FieldSchema{Name: "new", DataType: DataTypeInt32}, "1", AddColumnOptions{})
	require.ErrorIs(t, err, ErrPermissionDenied)
}

func addColumnSchema() CollectionSchema {
	schema := NewCollectionSchema("add_columns",
		FieldSchema{Name: "count", DataType: DataTypeInt32},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func addColumnDocument(primaryKey string, count int32, embedding []float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"count": count, "embedding": VectorFP32(embedding),
	}}
}
