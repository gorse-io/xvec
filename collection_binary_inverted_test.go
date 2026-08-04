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
	"reflect"
	"testing"
)

func TestCollectionBinaryInvertedDDLQueryOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "binary-invert")
	schema := NewCollectionSchema("binary_invert",
		FieldSchema{Name: "payload", DataType: DataTypeBinary, Nullable: true},
		FieldSchema{Name: "blobs", DataType: DataTypeArrayBinary, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{
			Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
			Index: NewFlatIndexParams(MetricTypeIP),
		},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, []Document{
		{PrimaryKey: "a", Fields: map[string]any{"payload": Binary("x"), "blobs": BinaryArray{Binary("x"), Binary("y")}, "embedding": VectorFP32{3, 0}}},
		{PrimaryKey: "b", Fields: map[string]any{"payload": Binary("y"), "blobs": BinaryArray{Binary("z")}, "embedding": VectorFP32{2, 0}}},
		{PrimaryKey: "c", Fields: map[string]any{"payload": nil, "blobs": nil, "embedding": VectorFP32{1, 0}}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := collection.CreateIndex(ctx, "payload", NewInvertIndexParams(), CreateIndexOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	plan, err := buildFilterPlan("payload = 'x'", collection.Schema())
	if err != nil {
		t.Fatal(err)
	}
	documents, err := collection.store.LiveDocuments(ctx)
	if err != nil {
		t.Fatal(err)
	}
	decoded := make([]Document, len(documents))
	for index := range documents {
		decoded[index], err = decodeStoredDocument(documents[index])
		if err != nil {
			t.Fatal(err)
		}
	}
	evaluated, err := evaluateFilterDocuments(ctx, plan, decoded, 1)
	if err != nil || !evaluated.usedIndex {
		t.Fatalf("binary candidate route = %#v, %v", evaluated, err)
	}
	assert := func(handle *Collection, filter string, want []string) {
		t.Helper()
		results, queryErr := handle.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: filter,
		})
		if queryErr != nil || !reflect.DeepEqual(documentKeys(results), want) {
			t.Fatalf("Query(%q) = %v, %v; want %v", filter, documentKeys(results), queryErr, want)
		}
	}
	assert(collection, "payload IN ('x', 'z')", []string{"a"})
	assert(collection, "payload >= 'y'", []string{"b"})
	assert(collection, "payload IS NULL", []string{"c"})
	assert(collection, "blobs CONTAIN_ANY ('y')", []string{"a"})
	assert(collection, "array_length(blobs) = 1", []string{"b"})
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}

	options := NewCollectionOptions()
	options.ReadOnly = true
	reopened, err := Open(ctx, path, options)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	assert(reopened, "payload IN ('x', 'z') AND blobs CONTAIN_ALL ('x', 'y')", []string{"a"})
}
