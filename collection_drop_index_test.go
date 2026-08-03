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

func TestDropScalarIndexPublishesAndPreservesForwardResults(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-scalar-index")
	schema := createIndexSchema()
	schema.Fields[1].Index = NewInvertIndexParams()
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, []Document{
		createIndexDocument("a", "alpha", 1, 3),
		createIndexDocument("b", "beta", 2, 2),
		createIndexDocument("c", "gamma", 3, 1),
	}); err != nil {
		t.Fatal(err)
	}
	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating>=2"}
	before, err := collection.Query(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(before), []string{"b", "c"}) {
		t.Fatalf("indexed results = %v, %v", documentKeys(before), err)
	}
	generation := collection.store.Manifest().Generation
	if err := collection.DropIndex(ctx, "rating"); err != nil {
		t.Fatal(err)
	}
	if collection.store.Manifest().Generation <= generation {
		t.Fatal("DropIndex did not publish a manifest generation")
	}
	rating, _ := collection.Schema().Field("rating")
	if rating.Index != nil || rating.IndexType() != IndexTypeUndefined {
		t.Fatalf("dropped scalar index = %#v", rating.Index)
	}
	after, err := collection.Query(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(after), []string{"b", "c"}) {
		t.Fatalf("forward results = %v, %v", documentKeys(after), err)
	}
	idempotentGeneration := collection.store.Manifest().Generation
	if err := collection.DropIndex(ctx, "rating"); err != nil {
		t.Fatal(err)
	}
	if collection.store.Manifest().Generation != idempotentGeneration {
		t.Fatal("idempotent scalar DropIndex advanced generation")
	}
	if err := collection.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	if err := collection.DropIndex(ctx, "rating"); err != nil {
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
	rating, _ = collection.Schema().Field("rating")
	if rating.Index != nil || collection.Stats().DocumentCount != 3 {
		t.Fatalf("reopened state = index %#v count %d", rating.Index, collection.Stats().DocumentCount)
	}
}

func TestDropVectorIndexRestoresDefaultFlatIP(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-vector-index")
	schema := NewCollectionSchema("drop_vector",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeL2)},
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
	if err != nil || !reflect.DeepEqual(documentKeys(before), []string{"near", "far"}) {
		t.Fatalf("L2 results = %v, %v", documentKeys(before), err)
	}
	if err := collection.DropIndex(ctx, "embedding"); err != nil {
		t.Fatal(err)
	}
	field, _ := collection.Schema().Field("embedding")
	flat, ok := field.Index.(FlatIndexParams)
	if !ok || flat.Metric != MetricTypeIP || flat.Quantize != QuantizeTypeUndefined {
		t.Fatalf("default vector index = %#v", field.Index)
	}
	after, err := collection.Query(ctx, query)
	if err != nil || !reflect.DeepEqual(documentKeys(after), []string{"far", "near"}) {
		t.Fatalf("default IP results = %v, %v", documentKeys(after), err)
	}
	generation := collection.store.Manifest().Generation
	if err := collection.DropIndex(ctx, "embedding"); err != nil {
		t.Fatal(err)
	}
	if collection.store.Manifest().Generation != generation {
		t.Fatal("idempotent vector DropIndex advanced generation")
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	field, _ = collection.Schema().Field("embedding")
	if flat, ok = field.Index.(FlatIndexParams); !ok || flat.Metric != MetricTypeIP {
		t.Fatalf("reopened default index = %#v", field.Index)
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
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{
		"text": "hello", "embedding": VectorFP32{1, 0},
	}}}); err != nil {
		t.Fatal(err)
	}
	if err := collection.DropIndex(ctx, "text"); err != nil {
		t.Fatal(err)
	}
	if err := collection.DropIndex(ctx, "embedding"); err != nil {
		t.Fatal(err)
	}
	text, _ := collection.Schema().Field("text")
	embedding, _ := collection.Schema().Field("embedding")
	if text.Index != nil || embedding.IndexType() != IndexTypeFlat {
		t.Fatalf("dropped later indexes = text %#v vector %#v", text.Index, embedding.Index)
	}
	results, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1})
	if err != nil || len(results) != 1 || results[0].PrimaryKey != "a" {
		t.Fatalf("query after dropping HNSW = %#v, %v", results, err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	collection, err = Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	text, _ = collection.Schema().Field("text")
	embedding, _ = collection.Schema().Field("embedding")
	if text.Index != nil || embedding.IndexType() != IndexTypeFlat {
		t.Fatalf("reopened dropped indexes = text %#v vector %#v", text.Index, embedding.Index)
	}
}

func TestDropIndexValidationLifecycleAndRollback(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "drop-index-errors")
	collection, err := CreateAndOpen(ctx, path, createIndexSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	generation := collection.store.Manifest().Generation
	if err := collection.DropIndex(ctx, "title"); err != nil {
		t.Fatalf("unindexed scalar no-op = %v", err)
	}
	if collection.store.Manifest().Generation != generation {
		t.Fatal("unindexed scalar no-op advanced generation")
	}
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
			if err := collection.DropIndex(ctx, testCase.column); !errors.Is(err, testCase.want) {
				t.Fatalf("error = %v, want %v", err, testCase.want)
			}
			if !reflect.DeepEqual(collection.Schema(), before) || collection.store.Manifest().Generation != beforeGeneration {
				t.Fatal("failed DropIndex changed schema")
			}
		})
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := collection.DropIndex(canceled, "embedding"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled DropIndex = %v", err)
	}
	if err := collection.DropIndex(nil, "embedding"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context = %v", err)
	}
	var nilCollection *Collection
	if err := nilCollection.DropIndex(ctx, "embedding"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection = %v", err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	if err := collection.DropIndex(ctx, "embedding"); !errors.Is(err, ErrFailedPrecondition) {
		t.Fatalf("closed DropIndex = %v", err)
	}
	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer readOnly.Close()
	if err := readOnly.DropIndex(ctx, "embedding"); !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("read-only DropIndex = %v", err)
	}

	corrupt, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "drop-index-rollback"), NewCollectionSchema("drop_rollback",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeL2)},
	), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer corrupt.Close()
	if _, err := corrupt.store.Insert(ctx, []db.WriteInput{{PrimaryKey: "corrupt", Payload: []byte("bad")}}); err != nil {
		t.Fatal(err)
	}
	before := corrupt.Schema()
	beforeGeneration := corrupt.store.Manifest().Generation
	if err := corrupt.DropIndex(ctx, "embedding"); err == nil {
		t.Fatal("DropIndex accepted corrupt vector backfill")
	}
	if !reflect.DeepEqual(corrupt.Schema(), before) || corrupt.store.Manifest().Generation != beforeGeneration {
		t.Fatal("failed vector DropIndex changed schema")
	}
}
