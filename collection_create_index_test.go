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

	"github.com/gorse-io/zvec/internal/db"
)

func TestCreateScalarIndexPublishesSchemaAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-scalar-index")
	schema := createIndexSchema()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, []Document{
		createIndexDocument("a", "alpha", 1, 3),
		createIndexDocument("b", "alphabet", 2, 2),
		createIndexDocument("c", "beta", 3, 1),
	}); err != nil {
		t.Fatal(err)
	}
	initialGeneration := collection.store.Manifest().Generation
	params := NewInvertIndexParams()
	if err := collection.CreateIndex(ctx, "rating", &params, CreateIndexOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	createdGeneration := collection.store.Manifest().Generation
	if createdGeneration <= initialGeneration {
		t.Fatalf("manifest generation = %d, initial %d", createdGeneration, initialGeneration)
	}
	rating, _ := collection.Schema().Field("rating")
	if !equalIndexParams(rating.Index, NewInvertIndexParams()) {
		t.Fatalf("rating index = %#v", rating.Index)
	}
	params.EnableExtendedWildcard = true
	rating, _ = collection.Schema().Field("rating")
	stored := rating.Index.(InvertIndexParams)
	if stored.EnableExtendedWildcard {
		t.Fatal("CreateIndex retained caller-owned parameter pointer")
	}
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating>=2",
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"b", "c"}) {
		t.Fatalf("indexed query = %v, %v", documentKeys(results), err)
	}
	if err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{}); err != nil {
		t.Fatal(err)
	}
	if got := collection.store.Manifest().Generation; got != createdGeneration {
		t.Fatalf("idempotent CreateIndex advanced generation to %d", got)
	}
	changed := NewInvertIndexParams()
	changed.EnableRangeOptimization = false
	if err := collection.CreateIndex(ctx, "rating", changed, CreateIndexOptions{Concurrency: 1}); err != nil {
		t.Fatal(err)
	}
	if collection.store.Manifest().Generation <= createdGeneration {
		t.Fatal("changed index parameters did not publish a generation")
	}
	extended := NewInvertIndexParams()
	extended.EnableExtendedWildcard = true
	if err := collection.CreateIndex(ctx, "title", extended, CreateIndexOptions{Concurrency: 3}); err != nil {
		t.Fatal(err)
	}
	results, err = collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "title LIKE '%bet'",
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"b"}) {
		t.Fatalf("extended wildcard query = %v, %v", documentKeys(results), err)
	}

	// The schema manifest and existing write WAL are independently durable;
	// neither needs a Flush before reopening.
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	rating, _ = collection.Schema().Field("rating")
	title, _ := collection.Schema().Field("title")
	if rating.Index == nil || rating.Index.(InvertIndexParams).EnableRangeOptimization ||
		title.Index == nil || !title.Index.(InvertIndexParams).EnableExtendedWildcard {
		t.Fatalf("reopened indexes = rating %#v title %#v", rating.Index, title.Index)
	}
	if collection.Stats().DocumentCount != 3 {
		t.Fatalf("reopened document count = %d", collection.Stats().DocumentCount)
	}
}

func TestCreateFlatIndexChangesMetricAtomically(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "create-flat-index")
	schema := NewCollectionSchema("create_flat",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, []Document{
		{PrimaryKey: "far", Fields: map[string]any{"embedding": VectorFP32{10, 0}}},
		{PrimaryKey: "near", Fields: map[string]any{"embedding": VectorFP32{1, 0}}},
	}); err != nil {
		t.Fatal(err)
	}
	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 2}
	before, err := collection.Query(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(before), []string{"far", "near"}) {
		t.Fatalf("IP results = %v, %v", documentKeys(before), err)
	}
	if err := collection.CreateIndex(ctx, "embedding", NewFlatIndexParams(MetricTypeL2), CreateIndexOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	after, err := collection.Query(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(after), []string{"near", "far"}) {
		t.Fatalf("L2 results = %v, %v", documentKeys(after), err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	after, err = collection.Query(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(after), []string{"near", "far"}) {
		t.Fatalf("reopened L2 results = %v, %v", documentKeys(after), err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	var typedNil *InvertIndexParams
	invalidFlat := NewFlatIndexParams(MetricTypeUndefined)
	quantizedDiskANN := NewDiskANNIndexParams(MetricTypeIP)
	quantizedDiskANN.Quantize = QuantizeTypeFP16
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
		{"fts-unimplemented", "text", NewFTSIndexParams(), CreateIndexOptions{}, ErrNotSupported},
		{"diskann-scalar-quantization-unimplemented", "embedding", quantizedDiskANN, CreateIndexOptions{}, ErrNotSupported},
		{"binary-invert-unimplemented", "binary", NewInvertIndexParams(), CreateIndexOptions{}, ErrNotSupported},
		{"scalar-index-conflict", "already_fts", NewInvertIndexParams(), CreateIndexOptions{}, ErrNotSupported},
		{"invalid-index-params", "embedding", invalidFlat, CreateIndexOptions{}, ErrInvalidArgument},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			beforeSchema := collection.Schema()
			beforeGeneration := collection.store.Manifest().Generation
			err := collection.CreateIndex(ctx, testCase.column, testCase.index, testCase.options)
			if !errors.Is(err, testCase.want) {
				t.Fatalf("error = %v, want %v", err, testCase.want)
			}
			if !reflect.DeepEqual(collection.Schema(), beforeSchema) || collection.store.Manifest().Generation != beforeGeneration {
				t.Fatal("failed CreateIndex changed schema or manifest")
			}
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := collection.CreateIndex(canceled, "text", NewInvertIndexParams(), CreateIndexOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled CreateIndex = %v", err)
	}
	if err := collection.CreateIndex(nil, "text", NewInvertIndexParams(), CreateIndexOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context = %v", err)
	}
	var nilCollection *Collection
	if err := nilCollection.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection = %v", err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	if err := collection.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{}); !errors.Is(err, ErrFailedPrecondition) {
		t.Fatalf("closed CreateIndex = %v", err)
	}
	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer readOnly.Close()
	if err := readOnly.CreateIndex(ctx, "text", NewInvertIndexParams(), CreateIndexOptions{}); !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("read-only CreateIndex = %v", err)
	}
}

func TestCreateIndexBackfillFailureLeavesSchemaUnchanged(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "create-index-rollback"), createIndexSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.store.Insert(ctx, []db.WriteInput{{PrimaryKey: "corrupt", Payload: []byte("not-a-document")}}); err != nil {
		t.Fatal(err)
	}
	before := collection.Schema()
	generation := collection.store.Manifest().Generation
	if err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2}); err == nil {
		t.Fatal("CreateIndex unexpectedly accepted corrupt backfill data")
	}
	if !reflect.DeepEqual(collection.Schema(), before) || collection.store.Manifest().Generation != generation {
		t.Fatal("failed backfill changed published schema")
	}
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
