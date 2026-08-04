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
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestAlterColumnMigratesNamesTypesIndexesAndReopens(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "alter-column")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	inserted, err := collection.Insert(ctx, []Document{
		alterColumnDocument("a", 1, float32(1.75), true, math.MaxUint32, []float32{3, 0}),
		alterColumnDocument("b", 2, nil, true, 10, []float32{2, 0}),
		alterColumnDocument("c", 3, nil, false, 20, []float32{1, 0}),
	})
	require.NoError(t, err)

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"count": int32(4)}}})
	require.NoError(t, err)

	wantIDs := map[string]uint64{"a": inserted[0].DocID, "b": updated[0].DocID, "c": inserted[2].DocID}

	invert := NewInvertIndexParams()
	invert.EnableRangeOptimization = true
	total := FieldSchema{Name: "total", DataType: DataTypeInt64, Index: invert}
	{
		err := collection.AlterColumn(ctx, "count", "", &total, AlterColumnOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	invert.EnableRangeOptimization = false
	storedTotal, _ := collection.Schema().Field("total")
	require.True(t, storedTotal.Index.(InvertIndexParams).EnableRangeOptimization,
		"AlterColumn retained caller-owned index parameters")
	{
		err := collection.AlterColumn(ctx, "maybe", "cost", nil, AlterColumnOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	price := FieldSchema{Name: "price", DataType: DataTypeDouble, Nullable: true}
	{
		err := collection.AlterColumn(ctx, "cost", "", &price, AlterColumnOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	capField := FieldSchema{Name: "cap", DataType: DataTypeInt32}
	{
		err := collection.AlterColumn(ctx, "cap", "", &capField, AlterColumnOptions{Concurrency: 1})
		require.NoError(t, err)
	}

	schema := collection.Schema()
	for _, old := range []string{"count", "maybe", "cost"} {
		{
			_, found := schema.Field(old)
			require.False(t, found)
		}
	}
	for _, current := range []string{"total", "price", "cap"} {
		{
			_, found := schema.Field(current)
			require.True(t, found)
		}
	}
	fetched, err := collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)

	wantTotals := []int64{1, 4, 3}
	wantCaps := []int32{-1, 10, 20}
	for index, document := range fetched {
		require.NotNil(t, document)
		require.Equal(t, wantIDs[document.PrimaryKey], document.DocID)
		require.Equal(t, wantTotals[index], document.Fields["total"])
		require.Equal(t, wantCaps[index], document.Fields["cap"])
	}
	require.Equal(t, float64(1.75), fetched[0].Fields["price"])
	{
		value, found := fetched[1].Fields["price"]
		require.True(t, found)
		require.Nil(t, value)
	}
	{
		_, found := fetched[2].Fields["price"]
		require.False(t, found)
	}

	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "total >= 3", Projection: Projection{OutputFields: []string{"total"}},
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, documentKeys(results))
	require.True(t, collection.Stats().DocumentCount == 3)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	require.NoError(t, err)
	require.Equal(t, int64(1), fetched[0].Fields["total"])
	require.Equal(t, float64(1.75), fetched[0].Fields["price"])

	document := Document{PrimaryKey: "d", Fields: map[string]any{
		"total": int64(5), "price": float64(2.5), "cap": int32(30), "text": "d",
		"embedding": VectorFP32{0.5, 0},
	}}
	{
		_, err := collection.Insert(ctx, []Document{document})
		require.NoError(t, err)
	}
}

func TestAlterColumnValidationAndPublicationRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.AlterColumn(ctx, "count", "renamed", nil, AlterColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "alter-column-errors")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{alterColumnDocument("one", 2, nil, true, 3, []float32{1, 0})})
		require.NoError(t, err)
	}
	{
		err := collection.AlterColumn(nil, "count", "renamed", nil, AlterColumnOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	initialSchema := collection.Schema()
	initialGeneration := collection.store.Manifest().Generation
	tests := []struct {
		name    string
		column  string
		rename  string
		field   *FieldSchema
		options AlterColumnOptions
	}{
		{"empty column", "", "renamed", nil, AlterColumnOptions{}},
		{"missing column", "missing", "renamed", nil, AlterColumnOptions{}},
		{"both forms", "count", "renamed", &FieldSchema{Name: "count", DataType: DataTypeInt64}, AlterColumnOptions{}},
		{"neither form", "count", "", nil, AlterColumnOptions{}},
		{"rename same", "count", "count", nil, AlterColumnOptions{}},
		{"rename duplicate", "count", "cap", nil, AlterColumnOptions{}},
		{"invalid rename", "count", "bad name", nil, AlterColumnOptions{}},
		{"old type unsupported", "text", "renamed_text", nil, AlterColumnOptions{}},
		{"new type unsupported", "count", "", &FieldSchema{Name: "count", DataType: DataTypeString}, AlterColumnOptions{}},
		{"nullable to required", "maybe", "", &FieldSchema{Name: "maybe", DataType: DataTypeFloat}, AlterColumnOptions{}},
		{"replacement duplicate", "count", "", &FieldSchema{Name: "cap", DataType: DataTypeInt64}, AlterColumnOptions{}},
		{"negative concurrency", "count", "renamed", nil, AlterColumnOptions{Concurrency: -1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			{
				err := collection.AlterColumn(ctx, test.column, test.rename, test.field, test.options)
				require.Error(t, err,
					"AlterColumn succeeded")
			}
			require.Equal(t, initialSchema, collection.Schema())
			{
				got := collection.store.Manifest().Generation
				require.Equal(t, initialGeneration, got)
			}
		})
	}
	equal := initialSchema.Fields[0].Clone()
	{
		err := collection.AlterColumn(ctx, "count", "", &equal, AlterColumnOptions{})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, initialGeneration, got)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.AlterColumn(canceled, "count", "renamed", nil, AlterColumnOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}

	versionLock, err := ailego.AcquireFileLock(ctx, filepath.Join(path, ".version.lock"), ailego.LockExclusive)
	require.NoError(t, err)

	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.AlterColumn(deadline, "count", "renamed", nil, AlterColumnOptions{Concurrency: 2})
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
	require.Equal(t, int32(2), fetched[0].Fields["count"])
}

func TestAlterColumnRejectsReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "alter-column-read-only")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
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
	err = collection.AlterColumn(ctx, "count", "renamed", nil, AlterColumnOptions{})
	require.ErrorIs(t, err, ErrPermissionDenied)
}

func TestAlterColumnEmptyCollectionPublishesSchemaOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "alter-column-empty")
	collection, err := CreateAndOpen(ctx, path, alterColumnSchema(), NewCollectionOptions())
	require.NoError(t, err)

	initialGeneration := collection.store.Manifest().Generation
	replacement := FieldSchema{Name: "amount", DataType: DataTypeInt64}
	{
		err := collection.AlterColumn(ctx, "count", "", &replacement, AlterColumnOptions{Concurrency: 4})
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
		_, found := collection.Schema().Field("amount")
		require.True(t, found,
			"reopened schema does not contain replacement field")
	}
}

func alterColumnSchema() CollectionSchema {
	schema := NewCollectionSchema("alter_columns",
		FieldSchema{Name: "count", DataType: DataTypeInt32},
		FieldSchema{Name: "maybe", DataType: DataTypeFloat, Nullable: true},
		FieldSchema{Name: "cap", DataType: DataTypeUint32},
		FieldSchema{Name: "text", DataType: DataTypeString},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func alterColumnDocument(primaryKey string, count int32, maybe any, includeMaybe bool, cap uint32, embedding []float32) Document {
	fields := map[string]any{
		"count": count, "cap": cap, "text": primaryKey, "embedding": VectorFP32(embedding),
	}
	if includeMaybe {
		fields["maybe"] = maybe
	}
	return Document{PrimaryKey: primaryKey, Fields: fields}
}
