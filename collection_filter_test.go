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

	dbsql "github.com/gorse-io/zvec/internal/db/sql"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.NoError(t, err)

	documents := []Document{
		filterDocument("a", "user-22", int32(1), StringArray{"red", "blue"}, Int32Array{1, 2}, true, "x", 1),
		filterDocument("b", "user-%22", int32(2), StringArray{"blue"}, Int32Array{}, false, "y", 4),
		filterDocument("c", "user-_22", nil, nil, Int32Array{2, 3}, true, "x", 3),
		filterDocument("d", "other", int32(4), StringArray{}, Int32Array{4}, false, "z", 2),
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
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
			require.NoError(t, queryErr)
			{
				got := documentKeys(results)
				require.Equal(t, testCase.want, got)
			}
		})
	}

	sparse, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}},
		TopK: 10, Filter: "numbers CONTAIN_ANY (2)",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"c", "a"}, documentKeys(sparse))

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, Filter: "tags CONTAIN_ANY ('blue')",
		GroupByField: "active", GroupCount: 2, TopKPerGroup: 2,
	})
	require.NoError(t, err)
	require.Len(t, groups, 2)
	require.True(t, groups[0].Value == "false")
	require.True(t, groups[1].Value == "true")
	require.Equal(t, []string{"b"}, documentKeys(groups[0].Documents))
	require.Equal(t, []string{"a"}, documentKeys(groups[1].Documents))
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	readOnly := NewCollectionOptions()
	readOnly.ReadOnly = true
	collection, err = Open(ctx, path, readOnly)
	require.NoError(t, err)

	defer collection.Close()
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: "rating >= 2",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "d"}, documentKeys(results))
}

func TestCollectionSQLFilterValidationAndCancellation(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "filter-errors"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0})})
		require.NoError(t, err)
	}

	for _, filter := range []string{
		"rating >", "missing=1", "rating='bad'", "rating=2147483648",
		"embedding=1", "category CONTAIN_ANY ('low')", "rating LIKE '1%'",
	} {
		_, queryErr := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: filter,
		})
		assert.ErrorIs(t, queryErr, ErrInvalidArgument)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = collection.Query(canceled, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: "rating=1",
	})
	require.ErrorIs(t, err, context.Canceled)
}

func TestCollectionScalarInvertedCandidatesMatchForwardSemantics(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "inverted-filters")
	extended := NewInvertIndexParams()
	extended.EnableExtendedWildcard = true
	schema := NewCollectionSchema("inverted_filters",
		FieldSchema{Name: "title", DataType: DataTypeString, Nullable: true, Index: extended},
		FieldSchema{Name: "code", DataType: DataTypeString, Index: NewInvertIndexParams()},
		FieldSchema{Name: "rating", DataType: DataTypeInt32, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "tags", DataType: DataTypeArrayString, Nullable: true, Index: NewInvertIndexParams()},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	plan, err := buildFilterPlan("title LIKE '%alpha' AND rating>=2", schema)
	require.NoError(t, err)

	configured := make(map[string]dbsql.Field)
	for _, field := range plan.Fields() {
		configured[field.Name] = field
	}
	require.True(t, configured["title"].Indexed)
	require.True(t, configured["title"].ExtendedWildcard)
	require.True(t, configured["title"].RangeOptimized)
	require.True(t, configured["rating"].Indexed)
	require.True(t, configured["rating"].RangeOptimized)

	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		invertedFilterDocument("a", "alpha", "alpha", int32(1), StringArray{"red", "blue"}, 5),
		invertedFilterDocument("b", "alphabet", "alphabet", int32(2), StringArray{"blue"}, 4),
		invertedFilterDocument("c", "beta", "beta", int32(3), nil, 3),
		invertedFilterDocument("d", "gamma-alpha", "gamma-alpha", int32(4), StringArray{}, 2),
		invertedFilterDocument("e", nil, "omega", nil, StringArray{"red"}, 1),
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	assertQuery := func(filter string, want []string) {
		t.Helper()
		results, queryErr := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10, Filter: filter,
		})
		require.NoError(t, queryErr)
		{
			got := documentKeys(results)
			require.Equal(t, want, got)
		}
	}
	assertQuery("title LIKE '%alpha'", []string{"a", "d"})
	assertQuery("code LIKE '%bet'", []string{"b"})
	assertQuery("rating>=2 AND code LIKE 'a%'", []string{"b"})
	assertQuery("rating=1 OR code LIKE '%pha%'", []string{"a", "b", "d"})
	assertQuery("tags CONTAIN_ANY ('red')", []string{"a", "e"})
	assertQuery("array_length(tags)=0", []string{"d"})
	assertQuery("title IS NULL", []string{"e"})
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	readOnly := NewCollectionOptions()
	readOnly.ReadOnly = true
	collection, err = Open(ctx, path, readOnly)
	require.NoError(t, err)

	defer collection.Close()
	assertQuery("rating>=2 AND tags NOT CONTAIN_ANY ('blue')", []string{"d"})
}

func TestFilterSchemaRejectsFTSAndValueAdapterCoversEveryScalarArray(t *testing.T) {
	fts := NewFTSIndexParams()
	schema := NewCollectionSchema("fts_filter",
		FieldSchema{Name: "body", DataType: DataTypeString, Index: fts},
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	{
		_, err := buildFilterPlan("body='text'", schema)
		require.Error(t, err,
			"FTS field scalar filter succeeded")
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
		require.NoError(t, err)
		require.Equal(t, testCase.kind, value.Kind())
		require.Equal(t, testCase.array, value.IsArray())
		require.False(t, value.IsNull())

		if testCase.array {
			{
				length, ok := value.Len()
				require.True(t, ok)
				require.Equal(t, testCase.len, length)
			}
		}
	}
	null, err := toFilterValue(dbsql.Field{Name: "missing", Kind: dbsql.ValueString, Filterable: true}, nil, false)
	require.NoError(t, err)
	require.True(t, null.IsNull())
	{
		_, err := toFilterValue(dbsql.Field{Name: "bad", Kind: dbsql.ValueInt32, Filterable: true}, int64(1), true)
		require.Error(t, err,
			"mismatched adapter value succeeded")
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

func invertedFilterDocument(primaryKey string, title any, code string, rating any, tags any, score float32) Document {
	return Document{PrimaryKey: primaryKey, Fields: map[string]any{
		"title": title, "code": code, "rating": rating, "tags": tags,
		"embedding": VectorFP32{score, 0},
	}}
}
