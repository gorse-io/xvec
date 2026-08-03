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
)

func TestAddColumnBackfillsAtomicallyAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column")
	schema := addColumnSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	documents := []Document{
		addColumnDocument("a", 1, []float32{3, 0}),
		addColumnDocument("b", 2, []float32{2, 0}),
		addColumnDocument("c", 3, []float32{1, 0}),
	}
	inserted, err := collection.Insert(ctx, documents)
	if err != nil {
		t.Fatal(err)
	}
	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"count": int32(4)}}})
	if err != nil {
		t.Fatal(err)
	}
	beforeIDs := map[string]uint64{"a": inserted[0].DocID, "c": inserted[2].DocID, "b": updated[0].DocID}
	initialGeneration := collection.store.Manifest().Generation
	index := NewInvertIndexParams()
	index.EnableRangeOptimization = true
	field := FieldSchema{Name: "derived", DataType: DataTypeInt64, Index: index}
	if err := collection.AddColumn(ctx, field, "(count * 2) + 1", AddColumnOptions{Concurrency: 3}); err != nil {
		t.Fatal(err)
	}
	if collection.store.Manifest().Generation <= initialGeneration {
		t.Fatal("AddColumn did not publish a new manifest generation")
	}
	if collection.Stats().DocumentCount != 3 {
		t.Fatalf("document count = %d", collection.Stats().DocumentCount)
	}
	fetched, err := collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	if err != nil {
		t.Fatal(err)
	}
	wantDerived := []int64{3, 9, 7}
	for index, document := range fetched {
		if document == nil || document.DocID != beforeIDs[document.PrimaryKey] || document.Fields["derived"] != wantDerived[index] {
			t.Fatalf("backfilled document %d = %#v", index, document)
		}
	}
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Filter: "derived >= 7", Projection: Projection{OutputFields: []string{"derived"}},
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"b", "c"}) {
		t.Fatalf("query after backfill = %v, %v", documentKeys(results), err)
	}

	if err := collection.AddColumn(ctx, FieldSchema{Name: "optional", DataType: DataTypeFloat, Nullable: true}, "", AddColumnOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{OutputFields: []string{"optional"}})
	if err != nil {
		t.Fatal(err)
	}
	for index, document := range fetched {
		value, found := document.Fields["optional"]
		if !found || value != nil || document.DocID != beforeIDs[document.PrimaryKey] {
			t.Fatalf("nullable document %d = %#v", index, document)
		}
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	fetched, err = collection.Fetch(ctx, []string{"a", "b", "c"}, Projection{})
	if err != nil {
		t.Fatal(err)
	}
	for index, document := range fetched {
		if document == nil || document.Fields["derived"] != wantDerived[index] {
			t.Fatalf("reopened document %d = %#v", index, document)
		}
		if value, found := document.Fields["optional"]; !found || value != nil {
			t.Fatalf("reopened nullable field %d = %#v", index, document.Fields)
		}
	}
	missing := addColumnDocument("missing", 5, []float32{1, 0})
	if _, err := collection.Insert(ctx, []Document{missing}); err == nil {
		t.Fatal("insert without added non-nullable field succeeded")
	}
	missing.Fields["derived"] = int64(11)
	missing.Fields["optional"] = nil
	if _, err := collection.Insert(ctx, []Document{missing}); err != nil {
		t.Fatal(err)
	}
}

func TestAddColumnValidationAndFailureRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	if err := nilCollection.AddColumn(ctx, FieldSchema{}, "", AddColumnOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection AddColumn = %v", err)
	}
	path := filepath.Join(t.TempDir(), "add-column-errors")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if err := collection.AddColumn(nil, FieldSchema{Name: "nil_ctx", DataType: DataTypeInt32}, "1", AddColumnOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context AddColumn = %v", err)
	}
	if _, err := collection.Insert(ctx, []Document{addColumnDocument("one", 2, []float32{1, 0})}); err != nil {
		t.Fatal(err)
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
			if err := collection.AddColumn(ctx, test.field, test.expression, test.options); err == nil {
				t.Fatal("AddColumn succeeded")
			}
			if _, found := collection.Schema().Field(test.field.Name); found && test.field.Name != "count" {
				t.Fatalf("failed field %q is visible", test.field.Name)
			}
			if got := collection.store.Manifest().Generation; got != initialGeneration {
				t.Fatalf("failed AddColumn published generation %d", got)
			}
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := collection.AddColumn(canceled, FieldSchema{Name: "canceled", DataType: DataTypeInt32}, "1", AddColumnOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled AddColumn = %v", err)
	}
	fetched, err := collection.Fetch(ctx, []string{"one"}, Projection{IncludeVectors: true})
	if err != nil || fetched[0] == nil || !reflect.DeepEqual(fetched[0].Fields, addColumnDocument("one", 2, []float32{1, 0}).Fields) {
		t.Fatalf("document after rollback = %#v, %v", fetched, err)
	}
}

func TestAddColumnEmptyCollectionMatchesDeferredExpressionBehavior(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column-empty")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if err := collection.AddColumn(ctx, FieldSchema{Name: "deferred", DataType: DataTypeInt32}, "CASE WHEN count > 0 THEN 1 END", AddColumnOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	document := addColumnDocument("one", 1, []float32{1, 0})
	document.Fields["deferred"] = int32(7)
	if _, err := collection.Insert(ctx, []Document{document}); err != nil {
		t.Fatal(err)
	}
}

func TestAddColumnRejectsReadOnlyHandle(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "add-column-read-only")
	collection, err := CreateAndOpen(ctx, path, addColumnSchema(), NewCollectionOptions())
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
	err = collection.AddColumn(ctx, FieldSchema{Name: "new", DataType: DataTypeInt32}, "1", AddColumnOptions{})
	if !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("read-only AddColumn = %v", err)
	}
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
