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
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCollectionCRUDFlushReopenAndReadOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "books")
	schema := testPublicCollectionSchema()
	options := NewCollectionOptions()
	options.WALSyncEvery = 1
	collection, err := CreateAndOpen(ctx, path, schema, options)
	if err != nil {
		t.Fatal(err)
	}
	if collection.Path() != path || !reflect.DeepEqual(collection.Schema(), schema) {
		t.Fatalf("collection metadata = %q, %#v", collection.Path(), collection.Schema())
	}
	if got := collection.Options(); !got.EnableMmap || got.MaxBufferSize != DefaultMaxBufferSize {
		t.Fatalf("effective options = %#v", got)
	}

	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{5, 0}),
	}
	inserted, err := collection.Insert(ctx, documents)
	if err != nil || inserted[0].DocID != 0 || inserted[1].DocID != 1 {
		t.Fatalf("insert = %#v, %v", inserted, err)
	}
	if stats := collection.Stats(); stats.DocumentCount != 2 || stats.IndexCompleteness["embedding"] != 1 || stats.IndexCompleteness["sparse"] != 1 {
		t.Fatalf("stats after insert = %#v", stats)
	}

	mixed, err := collection.Insert(ctx, []Document{
		documents[0],
		testPublicDocument("c", "charlie", "low", 3, 3, []float32{4, 0}),
	})
	var batchError *BatchWriteError
	if !errors.As(err, &batchError) || batchError.Failed != 1 || !errors.Is(err, ErrAlreadyExists) {
		t.Fatalf("mixed insert error = %#v, %v", mixed, err)
	}
	if mixed[0].Err == nil || mixed[1].Err != nil || mixed[1].DocID != 2 {
		t.Fatalf("mixed insert results = %#v", mixed)
	}

	updated, err := collection.Update(ctx, []Document{{
		PrimaryKey: "a", Fields: map[string]any{"title": "alpha-updated", "category": nil},
	}})
	if err != nil || updated[0].DocID != 3 {
		t.Fatalf("partial update = %#v, %v", updated, err)
	}
	upserted, err := collection.Upsert(ctx, []Document{{
		PrimaryKey: "b", Fields: map[string]any{"rating": int32(20)},
	}})
	if err != nil || upserted[0].DocID != 4 {
		t.Fatalf("partial upsert = %#v, %v", upserted, err)
	}
	invalidUpsert, err := collection.Upsert(ctx, []Document{{PrimaryKey: "new", Fields: map[string]any{"title": "new"}}})
	if !errors.Is(err, ErrInvalidArgument) || invalidUpsert[0].Err == nil {
		t.Fatalf("invalid new upsert = %#v, %v", invalidUpsert, err)
	}

	fetched, err := collection.Fetch(ctx, []string{"a", "b", "missing"}, Projection{IncludeVectors: true})
	if err != nil || fetched[2] != nil {
		t.Fatalf("fetch = %#v, %v", fetched, err)
	}
	if fetched[0].Fields["title"] != "alpha-updated" || fetched[0].Fields["category"] != nil || fetched[0].Fields["rating"] != int32(1) {
		t.Fatalf("updated document = %#v", fetched[0])
	}
	if _, found := fetched[0].Fields["embedding"]; !found {
		t.Fatalf("Fetch IncludeVectors omitted embedding: %#v", fetched[0])
	}
	if fetched[1].Fields["rating"] != int32(20) || fetched[1].Fields["title"] != "bravo" {
		t.Fatalf("upserted document = %#v", fetched[1])
	}

	deleted, err := collection.Delete(ctx, []string{"c", "missing"})
	if !errors.Is(err, ErrNotFound) || deleted[0].Err != nil || deleted[1].Err == nil {
		t.Fatalf("mixed delete = %#v, %v", deleted, err)
	}
	if got := collection.Stats().DocumentCount; got != 2 {
		t.Fatalf("document count after delete = %d", got)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Fetch(ctx, []string{"a"}, Projection{}); !errors.Is(err, ErrFailedPrecondition) {
		t.Fatalf("fetch after close = %v", err)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer readOnly.Close()
	if got := readOnly.Stats().DocumentCount; got != 2 {
		t.Fatalf("reopened count = %d", got)
	}
	fetched, err = readOnly.Fetch(ctx, []string{"a"}, Projection{})
	if err != nil || fetched[0].DocID != 3 {
		t.Fatalf("reopened fetch = %#v, %v", fetched, err)
	}
	if _, found := fetched[0].Fields["embedding"]; found {
		t.Fatalf("default projection included vector: %#v", fetched[0])
	}
	if _, err := readOnly.Insert(ctx, []Document{testPublicDocument("x", "xray", "low", 1, 1, []float32{1, 0})}); !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("read-only insert = %v", err)
	}
}

func TestCollectionDenseSparseRadiusProjectionAndGroupBy(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "query"), testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 5, []float32{5, 0}),
		testPublicDocument("c", "charlie", "low", 3, 4, []float32{4, 0}),
		testPublicDocument("d", "delta", "high", 4, 3, []float32{3, 0}),
		testPublicDocument("e", "echo", "", 5, 6, []float32{6, 0}),
	}
	documents[4].Fields["category"] = nil
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
	}

	params := NewFlatQueryParams()
	params.Radius = 4
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Params: params, Projection: Projection{OutputFields: []string{"title"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := documentKeys(results); !reflect.DeepEqual(got, []string{"e", "b", "c"}) {
		t.Fatalf("dense radius keys = %#v", got)
	}
	if !reflect.DeepEqual(results[0].Fields, map[string]any{"title": "echo"}) || results[0].Score != 6 {
		t.Fatalf("dense projection result = %#v", results[0])
	}

	sparseResults, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}},
		TopK: 3, Projection: Projection{OutputFields: []string{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := documentKeys(sparseResults); !reflect.DeepEqual(got, []string{"e", "b", "c"}) {
		t.Fatalf("sparse keys = %#v", got)
	}
	if sparseResults[0].Fields == nil || len(sparseResults[0].Fields) != 0 {
		t.Fatalf("empty projection = %#v", sparseResults[0])
	}

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0},
		GroupByField: "category", GroupCount: 3, TopKPerGroup: 2,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 3 || groups[0].Value != "" || groups[1].Value != "high" || groups[2].Value != "low" {
		t.Fatalf("groups = %#v", groups)
	}
	if got := documentKeys(groups[1].Documents); !reflect.DeepEqual(got, []string{"b", "d"}) {
		t.Fatalf("high group = %#v", got)
	}
	if got := documentKeys(groups[2].Documents); !reflect.DeepEqual(got, []string{"c", "a"}) {
		t.Fatalf("low group = %#v", got)
	}

	filtered, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: "rating > 1"})
	if err != nil || !reflect.DeepEqual(documentKeys(filtered), []string{"e"}) {
		t.Fatalf("filtered query = %#v, %v", filtered, err)
	}
	if _, err := collection.Query(ctx, VectorQuery{Field: "embedding", SparseVector: SparseVectorFP32{}, TopK: 1}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("wrong query vector kind = %v", err)
	}
	if _, err := collection.GroupByQuery(ctx, GroupByVectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, GroupByField: "sparse", GroupCount: 1, TopKPerGroup: 1}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("vector group field = %v", err)
	}
}

func TestCollectionUnsupportedIndexesReturnNotSupported(t *testing.T) {
	ctx := context.Background()
	tests := []struct {
		name  string
		index IndexParams
	}{
		{name: "HNSW", index: NewHNSWIndexParams(MetricTypeIP)},
		{name: "quantized Flat", index: FlatIndexParams{Metric: MetricTypeIP, Quantize: QuantizeTypeFP16}},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			schema := NewCollectionSchema("later", FieldSchema{
				Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: testCase.index,
			})
			collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "later"), schema, NewCollectionOptions())
			if err != nil {
				t.Fatal(err)
			}
			defer collection.Close()
			if _, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{1, 0}}}}); err != nil {
				t.Fatal(err)
			}
			if _, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1}); !errors.Is(err, ErrNotSupported) {
				t.Fatalf("query error = %v", err)
			}
			if got := collection.Stats().IndexCompleteness["embedding"]; got != 0 {
				t.Fatalf("unsupported completeness = %v", got)
			}
		})
	}
}

func TestCollectionFutureMutationsReturnNotSupported(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "later-api"), testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	operations := []func() error{
		func() error { return collection.AlterColumn(ctx, "title", "name", nil, AlterColumnOptions{}) },
		func() error { return collection.DropColumn(ctx, "rating") },
		func() error { return collection.Optimize(ctx, OptimizeOptions{}) },
	}
	for index, operation := range operations {
		if err := operation(); !errors.Is(err, ErrNotSupported) {
			t.Fatalf("future operation %d = %v", index, err)
		}
	}
}

func TestCollectionReplaysPublicDocumentPayloadWithoutFlush(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "recovery")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 2, []float32{2, 0})}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(ctx, path, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	results, err := reopened.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1,
		Projection: Projection{IncludeVectors: true},
	})
	if err != nil || len(results) != 1 || results[0].PrimaryKey != "a" || results[0].Score != 2 {
		t.Fatalf("recovered public query = %#v, %v", results, err)
	}
}

func TestCollectionDestroyAndArgumentErrors(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "destroy")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if err := collection.Destroy(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("destroyed path stat = %v", err)
	}
	if _, err := Open(ctx, path, NewCollectionOptions()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("open destroyed collection = %v", err)
	}

	if _, err := CreateAndOpen(nil, path, testPublicCollectionSchema(), NewCollectionOptions()); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil create context = %v", err)
	}
	var nilCollection *Collection
	if _, err := nilCollection.Insert(ctx, []Document{{PrimaryKey: "a"}}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection insert = %v", err)
	}
	if _, err := nilCollection.Query(ctx, VectorQuery{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil collection query = %v", err)
	}
}

func TestEncodeGroupValueMatchesPinnedFormatting(t *testing.T) {
	tests := []struct {
		value    any
		dataType DataType
		want     string
	}{
		{value: nil, dataType: DataTypeString, want: ""},
		{value: true, dataType: DataTypeBool, want: "true"},
		{value: int32(-2), dataType: DataTypeInt32, want: "-2"},
		{value: uint64(9), dataType: DataTypeUint64, want: "9"},
		{value: float32(1.25), dataType: DataTypeFloat, want: "1.250000"},
		{value: 2.5, dataType: DataTypeDouble, want: "2.500000"},
	}
	for _, testCase := range tests {
		got, err := encodeGroupValue(testCase.value, testCase.dataType)
		if err != nil || got != testCase.want {
			t.Fatalf("encodeGroupValue(%T) = %q, %v; want %q", testCase.value, got, err, testCase.want)
		}
	}
}

func testPublicCollectionSchema() CollectionSchema {
	schema := NewCollectionSchema("books",
		FieldSchema{Name: "title", DataType: DataTypeString},
		FieldSchema{Name: "category", DataType: DataTypeString, Nullable: true},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	return schema
}

func testPublicDocument(primaryKey, title, category string, rating int32, score float32, dense []float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "category": category, "rating": rating,
		"embedding": VectorFP32(dense),
		"sparse":    SparseVectorFP32{Indices: []uint32{2}, Values: []float32{score}},
	}}
}

func documentKeys(documents []Document) []string {
	keys := make([]string, len(documents))
	for index := range documents {
		keys[index] = documents[index].PrimaryKey
	}
	return keys
}
