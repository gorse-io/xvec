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
)

func TestDeleteByFilterAcrossSegmentsAndWALRecovery(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "delete-filter")
	schema := deleteFilterSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	first := []Document{
		deleteFilterDocument("a", "alpha", int32(1), StringArray{"red"}, 5),
		deleteFilterDocument("b", "beta", int32(2), StringArray{"blue"}, 4),
		deleteFilterDocument("c", "gamma", nil, StringArray{"red", "blue"}, 3),
	}
	if _, err := collection.Insert(ctx, first); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	second := []Document{
		deleteFilterDocument("d", "delta", int32(4), StringArray{}, 2),
		deleteFilterDocument("e", "omega", int32(5), nil, 1),
	}
	if _, err := collection.Insert(ctx, second); err != nil {
		t.Fatal(err)
	}

	if err := collection.DeleteByFilter(ctx, "rating >"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid filter error = %v", err)
	}
	if got := collection.Stats().DocumentCount; got != 5 {
		t.Fatalf("invalid filter changed document count to %d", got)
	}
	if err := collection.DeleteByFilter(ctx, "(rating>=2 AND title LIKE 'd%') OR tags CONTAIN_ALL ('red', 'blue')"); err != nil {
		t.Fatal(err)
	}
	if got := collection.Stats().DocumentCount; got != 3 {
		t.Fatalf("document count after first filter = %d", got)
	}
	assertFetchedPresence(t, collection, []string{"a", "b", "c", "d", "e"}, []bool{true, true, false, false, true})
	if err := collection.DeleteByFilter(ctx, "rating>100"); err != nil {
		t.Fatalf("no-match delete = %v", err)
	}
	if err := collection.DeleteByFilter(ctx, "title LIKE '%ta'"); err != nil {
		t.Fatal(err)
	}
	if got := collection.Stats().DocumentCount; got != 2 {
		t.Fatalf("document count after wildcard filter = %d", got)
	}

	// Close without Flush: both immutable-segment and writing-segment deletes
	// must be reconstructed from the WAL.
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	assertFetchedPresence(t, collection, []string{"a", "b", "c", "d", "e"}, []bool{true, false, false, false, true})
	if got := collection.Stats().DocumentCount; got != 2 {
		t.Fatalf("reopened document count = %d", got)
	}
	if err := collection.DeleteByFilter(ctx, "title LIKE '%'"); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	collection, err = Open(ctx, path, readOnlyOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if got := collection.Stats().DocumentCount; got != 0 {
		t.Fatalf("fully deleted reopened count = %d", got)
	}
}

func TestDeleteByFilterUsesOnlyCurrentDocumentVersions(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "delete-current"), deleteFilterSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, []Document{deleteFilterDocument("versioned", "before", int32(1), StringArray{"old"}, 1)}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Update(ctx, []Document{{PrimaryKey: "versioned", Fields: map[string]any{
		"title": "after", "rating": int32(3), "tags": StringArray{"new"},
	}}}); err != nil {
		t.Fatal(err)
	}
	if err := collection.DeleteByFilter(ctx, "rating=1 OR tags CONTAIN_ANY ('old')"); err != nil {
		t.Fatal(err)
	}
	assertFetchedPresence(t, collection, []string{"versioned"}, []bool{true})
	if err := collection.DeleteByFilter(ctx, "rating=3 AND tags CONTAIN_ANY ('new')"); err != nil {
		t.Fatal(err)
	}
	assertFetchedPresence(t, collection, []string{"versioned"}, []bool{false})
}

func TestDeleteByFilterValidationCancellationAndLifecycle(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	if err := nilCollection.DeleteByFilter(ctx, "rating=1"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection error = %v", err)
	}
	path := filepath.Join(t.TempDir(), "delete-filter-errors")
	collection, err := CreateAndOpen(ctx, path, deleteFilterSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if err := collection.DeleteByFilter(nil, "rating=1"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context error = %v", err)
	}
	for _, filter := range []string{"", "   ", "missing=1", "embedding=1"} {
		if err := collection.DeleteByFilter(ctx, filter); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("filter %q error = %v", filter, err)
		}
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := collection.DeleteByFilter(canceled, "rating=1"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
	if err := collection.DeleteByFilter(ctx, "rating=1"); err != nil {
		t.Fatalf("empty collection delete = %v", err)
	}
	if _, err := collection.Insert(ctx, []Document{deleteFilterDocument("a", "alpha", int32(1), StringArray{}, 1)}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	if err := collection.DeleteByFilter(ctx, "rating=1"); !errors.Is(err, ErrFailedPrecondition) {
		t.Fatalf("closed collection error = %v", err)
	}
	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer readOnly.Close()
	if err := readOnly.DeleteByFilter(ctx, "rating=1"); !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("read-only error = %v", err)
	}
	if got := readOnly.Stats().DocumentCount; got != 1 {
		t.Fatalf("read-only mutation changed count to %d", got)
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
	if err != nil {
		t.Fatal(err)
	}
	if len(fetched) != len(present) {
		t.Fatalf("Fetch returned %d entries, want %d", len(fetched), len(present))
	}
	for index := range fetched {
		if (fetched[index] != nil) != present[index] {
			t.Errorf("Fetch(%q) present=%t, want %t", keys[index], fetched[index] != nil, present[index])
		}
	}
}
