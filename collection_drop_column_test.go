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
	"reflect"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestDropColumnRemovesPayloadsAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-column")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	inserted, err := collection.Insert(ctx, []Document{
		dropColumnDocument("a", 5, float64(1.25), true, []float32{3, 0}),
		dropColumnDocument("b", 3, nil, true, []float32{2, 0}),
		dropColumnDocument("c", 1, nil, false, []float32{1, 0}),
	})
	if err != nil {
		t.Fatal(err)
	}
	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"rating": int32(4)}}})
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := map[string]uint64{"a": inserted[0].DocID, "b": updated[0].DocID, "c": inserted[2].DocID}
	if err := collection.DropColumn(ctx, "rating"); err != nil {
		t.Fatal(err)
	}
	if _, found := collection.Schema().Field("rating"); found {
		t.Fatal("dropped indexed field remains in schema")
	}
	assertStoredFieldAbsent(t, ctx, collection, "rating", wantIDs)
	if _, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 3, Filter: "rating >= 2",
	}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("query using dropped field = %v", err)
	}
	if err := collection.DropColumn(ctx, "optional"); err != nil {
		t.Fatal(err)
	}
	assertStoredFieldAbsent(t, ctx, collection, "optional", wantIDs)
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 3,
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"a", "b", "c"}) {
		t.Fatalf("query after DropColumn = %v, %v", documentKeys(results), err)
	}
	for index := range results {
		if _, found := results[index].Fields["rating"]; found {
			t.Fatalf("query result %d contains rating: %#v", index, results[index].Fields)
		}
		if _, found := results[index].Fields["optional"]; found {
			t.Fatalf("query result %d contains optional: %#v", index, results[index].Fields)
		}
	}
	if collection.Stats().DocumentCount != 3 {
		t.Fatalf("document count = %d", collection.Stats().DocumentCount)
	}

	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	assertStoredFieldAbsent(t, ctx, collection, "rating", wantIDs)
	assertStoredFieldAbsent(t, ctx, collection, "optional", wantIDs)
	withDropped := dropColumnDocument("bad", 2, nil, false, []float32{1, 0})
	if _, err := collection.Insert(ctx, []Document{withDropped}); err == nil {
		t.Fatal("insert containing dropped fields succeeded")
	}
	valid := Document{PrimaryKey: "d", Fields: map[string]any{
		"text": "d", "embedding": VectorFP32{0.5, 0},
	}}
	if _, err := collection.Insert(ctx, []Document{valid}); err != nil {
		t.Fatal(err)
	}
}

func TestDropColumnValidationAndPublicationRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	if err := nilCollection.DropColumn(ctx, "rating"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection DropColumn = %v", err)
	}
	path := filepath.Join(t.TempDir(), "drop-column-errors")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, []Document{dropColumnDocument("one", 2, nil, false, []float32{1, 0})}); err != nil {
		t.Fatal(err)
	}
	if err := collection.DropColumn(nil, "rating"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context DropColumn = %v", err)
	}
	initialSchema := collection.Schema()
	initialGeneration := collection.store.Manifest().Generation
	for _, column := range []string{"", "missing", "text", "embedding"} {
		t.Run(column, func(t *testing.T) {
			if err := collection.DropColumn(ctx, column); err == nil {
				t.Fatal("DropColumn succeeded")
			}
			if !reflect.DeepEqual(collection.Schema(), initialSchema) || collection.store.Manifest().Generation != initialGeneration {
				t.Fatalf("failed DropColumn changed state: schema %#v generation %d", collection.Schema(), collection.store.Manifest().Generation)
			}
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := collection.DropColumn(canceled, "rating"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled DropColumn = %v", err)
	}

	versionLock, err := ailego.AcquireFileLock(ctx, filepath.Join(path, ".version.lock"), ailego.LockExclusive)
	if err != nil {
		t.Fatal(err)
	}
	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.DropColumn(deadline, "rating")
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		_ = versionLock.Close()
		t.Fatalf("blocked DropColumn = %v", err)
	}
	if err := versionLock.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(collection.Schema(), initialSchema) || collection.store.Manifest().Generation != initialGeneration {
		t.Fatalf("failed publication changed state: schema %#v generation %d", collection.Schema(), collection.store.Manifest().Generation)
	}
	fetched, err := collection.Fetch(ctx, []string{"one"}, Projection{})
	if err != nil || fetched[0] == nil || fetched[0].Fields["rating"] != int32(2) {
		t.Fatalf("document after rollback = %#v, %v", fetched, err)
	}
}

func TestDropColumnRejectsLastFieldAndReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	t.Run("last field", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "drop-last-column")
		schema := NewCollectionSchema("drop_last", FieldSchema{Name: "only", DataType: DataTypeInt32})
		schema.MaxDocsPerSegment = MinMaxDocsPerSegment
		collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
		if err != nil {
			t.Fatal(err)
		}
		defer collection.Close()
		if err := collection.DropColumn(ctx, "only"); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("drop last field = %v", err)
		}
	})
	t.Run("read only", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "drop-column-read-only")
		collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
		if err != nil {
			t.Fatal(err)
		}
		if err := collection.Close(); err != nil {
			t.Fatal(err)
		}
		options := NewCollectionOptions()
		options.ReadOnly = true
		collection, err = Open(ctx, path, options)
		if err != nil {
			t.Fatal(err)
		}
		defer collection.Close()
		if err := collection.DropColumn(ctx, "rating"); !errors.Is(err, ErrPermissionDenied) {
			t.Fatalf("read-only DropColumn = %v", err)
		}
	})
}

func TestDropColumnEmptyCollectionPublishesSchemaOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-column-empty")
	collection, err := CreateAndOpen(ctx, path, dropColumnSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	initialGeneration := collection.store.Manifest().Generation
	if err := collection.DropColumn(ctx, "rating"); err != nil {
		t.Fatal(err)
	}
	if collection.store.Manifest().Generation <= initialGeneration || collection.store.Manifest().PersistedSegments != nil {
		t.Fatalf("empty drop manifest = %#v", collection.store.Manifest())
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, found := collection.Schema().Field("rating"); found {
		t.Fatal("reopened schema contains dropped field")
	}
}

func assertStoredFieldAbsent(t *testing.T, ctx context.Context, collection *Collection, field string, wantIDs map[string]uint64) {
	t.Helper()
	stored, err := collection.store.LiveDocuments(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range stored {
		if item.DocID != wantIDs[item.PrimaryKey] {
			t.Fatalf("document %q ID = %d, want %d", item.PrimaryKey, item.DocID, wantIDs[item.PrimaryKey])
		}
		fields, decodeErr := unmarshalDocumentPayload(item.Payload)
		if decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if _, found := fields[field]; found {
			t.Fatalf("stored document %q still contains %q", item.PrimaryKey, field)
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
