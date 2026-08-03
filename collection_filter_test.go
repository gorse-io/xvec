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

	dbsql "github.com/gorse-io/zvec/internal/db/sql"
)

func TestCollectionSQLFilterQueryGroupByAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "filters")
	schema := NewCollectionSchema("filters",
		FieldSchema{Name: "title", DataType: DataTypeString, Nullable: true},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true},
		FieldSchema{Name: "tags", DataType: DataTypeArrayString, Nullable: true},
		FieldSchema{Name: "numbers", DataType: DataTypeArrayInt32},
		FieldSchema{Name: "active", DataType: DataTypeBool},
		FieldSchema{Name: "payload", DataType: DataTypeBinary},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	documents := []Document{
		filterDocument("a", "user-22", int32(1), StringArray{"red", "blue"}, Int32Array{1, 2}, true, "x", 1),
		filterDocument("b", "user-%22", int32(2), StringArray{"blue"}, Int32Array{}, false, "y", 4),
		filterDocument("c", "user-_22", nil, nil, Int32Array{2, 3}, true, "x", 3),
		filterDocument("d", "other", int32(4), StringArray{}, Int32Array{4}, false, "z", 2),
	}
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		filter string
		want   []string
	}{
		{"rating=2 OR rating=4", []string{"b", "d"}},
		{`title LIKE 'user-\%%'`, []string{"b"}},
		{`title LIKE 'user-\_%'`, []string{"c"}},
		{"tags CONTAIN_ALL ('red', 'blue')", []string{"a"}},
		{"tags CONTAIN_ANY ('blue')", []string{"b", "a"}},
		{"tags CONTAIN_ALL ()", []string{"b", "d", "a"}},
		{"tags NOT CONTAIN_ANY ()", []string{"b", "d", "a"}},
		{"tags CONTAIN_ANY ()", []string{}},
		{"array_length(tags) = 0", []string{"d"}},
		{"rating IS NULL", []string{"c"}},
		{"rating IS NOT NULL AND active = false", []string{"b", "d"}},
		{"payload = 'x'", []string{"c", "a"}},
	}
	for _, testCase := range tests {
		t.Run(testCase.filter, func(t *testing.T) {
			results, queryErr := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
				Filter: testCase.filter, Projection: Projection{OutputFields: []string{}},
			})
			if queryErr != nil {
				t.Fatal(queryErr)
			}
			if got := documentKeys(results); !reflect.DeepEqual(got, testCase.want) {
				t.Fatalf("keys = %#v, want %#v", got, testCase.want)
			}
		})
	}

	sparse, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}},
		TopK: 10, Filter: "numbers CONTAIN_ANY (2)",
	})
	if err != nil || !reflect.DeepEqual(documentKeys(sparse), []string{"c", "a"}) {
		t.Fatalf("sparse filtered query = %#v, %v", documentKeys(sparse), err)
	}

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, Filter: "tags CONTAIN_ANY ('blue')",
		GroupByField: "active", GroupCount: 2, TopKPerGroup: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || groups[0].Value != "false" || groups[1].Value != "true" ||
		!reflect.DeepEqual(documentKeys(groups[0].Documents), []string{"b"}) ||
		!reflect.DeepEqual(documentKeys(groups[1].Documents), []string{"a"}) {
		t.Fatalf("filtered groups = %#v", groups)
	}

	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if err := collection.Close(); err != nil {
		t.Fatal(err)
	}
	readOnly := NewCollectionOptions()
	readOnly.ReadOnly = true
	collection, err = Open(ctx, path, readOnly)
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating >= 2",
	})
	if err != nil || !reflect.DeepEqual(documentKeys(results), []string{"b", "d"}) {
		t.Fatalf("reopened filtered query = %#v, %v", documentKeys(results), err)
	}
}

func TestCollectionSQLFilterValidationAndCancellation(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "filter-errors"), testPublicCollectionSchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0})}); err != nil {
		t.Fatal(err)
	}
	for _, filter := range []string{
		"rating >", "missing=1", "rating='bad'", "rating=2147483648",
		"embedding=1", "category CONTAIN_ANY ('low')", "rating LIKE '1%'",
	} {
		_, queryErr := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: filter,
		})
		if !errors.Is(queryErr, ErrInvalidArgument) {
			t.Errorf("filter %q error = %v", filter, queryErr)
		}
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = collection.Query(canceled, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: "rating=1",
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled filtered query = %v", err)
	}
}

func TestFilterSchemaRejectsFTSAndValueAdapterCoversEveryScalarArray(t *testing.T) {
	fts := NewFTSIndexParams()
	schema := NewCollectionSchema("fts_filter",
		FieldSchema{Name: "body", DataType: DataTypeString, Index: fts},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	if _, err := buildFilterPlan("body='text'", schema); err == nil {
		t.Fatal("FTS field scalar filter succeeded")
	}

	tests := []struct {
		kind  dbsql.ValueKind
		array bool
		raw   any
		len   int
	}{
		{dbsql.ValueBinary, false, Binary("x"), 0},
		{dbsql.ValueString, false, "x", 0},
		{dbsql.ValueBool, false, true, 0},
		{dbsql.ValueInt32, false, int32(-1), 0},
		{dbsql.ValueInt64, false, int64(-1), 0},
		{dbsql.ValueUint32, false, uint32(1), 0},
		{dbsql.ValueUint64, false, uint64(1), 0},
		{dbsql.ValueFloat32, false, float32(1.5), 0},
		{dbsql.ValueFloat64, false, float64(1.5), 0},
		{dbsql.ValueBinary, true, BinaryArray{Binary("x")}, 1},
		{dbsql.ValueString, true, StringArray{"x"}, 1},
		{dbsql.ValueBool, true, BoolArray{true}, 1},
		{dbsql.ValueInt32, true, Int32Array{-1}, 1},
		{dbsql.ValueInt64, true, Int64Array{-1}, 1},
		{dbsql.ValueUint32, true, Uint32Array{1}, 1},
		{dbsql.ValueUint64, true, Uint64Array{1}, 1},
		{dbsql.ValueFloat32, true, Float32Array{1.5}, 1},
		{dbsql.ValueFloat64, true, Float64Array{1.5}, 1},
	}
	for _, testCase := range tests {
		field := dbsql.Field{Name: "value", Kind: testCase.kind, Array: testCase.array, Filterable: true}
		value, err := toFilterValue(field, testCase.raw, true)
		if err != nil || value.Kind() != testCase.kind || value.IsArray() != testCase.array || value.IsNull() {
			t.Fatalf("toFilterValue(%T) = kind=%s array=%t null=%t error=%v", testCase.raw, value.Kind(), value.IsArray(), value.IsNull(), err)
		}
		if testCase.array {
			if length, ok := value.Len(); !ok || length != testCase.len {
				t.Fatalf("array %T length = %d, %t", testCase.raw, length, ok)
			}
		}
	}
	null, err := toFilterValue(dbsql.Field{Name: "missing", Kind: dbsql.ValueString, Filterable: true}, nil, false)
	if err != nil || !null.IsNull() {
		t.Fatalf("missing field = %#v, %v", null, err)
	}
	if _, err := toFilterValue(dbsql.Field{Name: "bad", Kind: dbsql.ValueInt32, Filterable: true}, int64(1), true); err == nil {
		t.Fatal("mismatched adapter value succeeded")
	}
}

func filterDocument(primaryKey, title string, rating, tags any, numbers Int32Array, active bool, payload string, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "rating": rating, "tags": tags, "numbers": numbers,
		"active": active, "payload": Binary(payload),
		"embedding": VectorFP32{score, 0},
		"sparse":    SparseVectorFP32{Indices: []uint32{1}, Values: []float32{score}},
	}}
}
