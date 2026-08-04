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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCollectionCRUDFlushReopenAndReadOnly(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "books")
	schema := testPublicCollectionSchema()
	options := NewCollectionOptions()
	options.WALSyncEvery = 1
	collection, err := CreateAndOpen(ctx, path, schema, options)
	require.NoError(t, err)
	require.Equal(t, path, collection.Path())
	require.Equal(t, schema, collection.Schema())
	{
		got := collection.Options()
		require.True(t, got.EnableMmap)
		require.Equal(t, DefaultMaxBufferSize, got.MaxBufferSize)
	}

	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{5, 0}),
	}
	inserted, err := collection.Insert(ctx, documents)
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 0)
	require.True(t, inserted[1].DocID == 1)
	{
		stats := collection.Stats()
		require.True(t, stats.DocumentCount == 2)
		require.True(t, stats.IndexCompleteness["embedding"] == 1)
		require.True(t, stats.IndexCompleteness["sparse"] == 1)
	}

	mixed, err := collection.Insert(ctx, []Document{
		documents[0],
		testPublicDocument("c", "charlie", "low", 3, 3, []float32{4, 0}),
	})
	var batchError *BatchWriteError
	require.ErrorAs(t, err, &batchError)
	require.True(t, batchError.Failed == 1)
	require.ErrorIs(t, err, ErrAlreadyExists)
	require.Error(t, mixed[0].Err)
	require.NoError(t, mixed[1].Err)
	require.True(t, mixed[1].DocID == 2)

	updated, err := collection.Update(ctx, []Document{{
		PrimaryKey: "a", Fields: map[string]any{"title": "alpha-updated", "category": nil},
	}})
	require.NoError(t, err)
	require.True(t, updated[0].DocID == 3)

	upserted, err := collection.Upsert(ctx, []Document{{
		PrimaryKey: "b", Fields: map[string]any{"rating": int32(20)},
	}})
	require.NoError(t, err)
	require.True(t, upserted[0].DocID == 4)

	invalidUpsert, err := collection.Upsert(ctx, []Document{{PrimaryKey: "new", Fields: map[string]any{"title": "new"}}})
	require.ErrorIs(t, err, ErrInvalidArgument)
	require.Error(t, invalidUpsert[0].Err)

	fetched, err := collection.Fetch(ctx, []string{"a", "b", "missing"}, Projection{IncludeVectors: true})
	require.NoError(t, err)
	require.Nil(t, fetched[2])
	require.True(t, fetched[0].Fields["title"] == "alpha-updated")
	require.Nil(t, fetched[0].Fields["category"])
	require.Equal(t, int32(1), fetched[0].Fields["rating"])
	{
		_, found := fetched[0].Fields["embedding"]
		require.True(t, found)
	}
	require.Equal(t, int32(20), fetched[1].Fields["rating"])
	require.True(t, fetched[1].Fields["title"] == "bravo")

	deleted, err := collection.Delete(ctx, []string{"c", "missing"})
	require.ErrorIs(t, err, ErrNotFound)
	require.NoError(t, deleted[0].Err)
	require.Error(t, deleted[1].Err)
	{
		got := collection.Stats().DocumentCount
		require.True(t, got == 2)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		_, err := collection.Fetch(ctx, []string{"a"}, Projection{})
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	readOnly, err := Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)

	defer readOnly.Close()
	{
		got := readOnly.Stats().DocumentCount
		require.True(t, got == 2)
	}

	fetched, err = readOnly.Fetch(ctx, []string{"a"}, Projection{})
	require.NoError(t, err)
	require.True(t, fetched[0].DocID == 3)
	{
		_, found := fetched[0].Fields["embedding"]
		require.False(t, found)
	}
	{
		_, err := readOnly.Insert(ctx, []Document{testPublicDocument("x", "xray", "low", 1, 1, []float32{1, 0})})
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
}

func TestCollectionDenseSparseRadiusProjectionAndGroupBy(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "query"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 5, []float32{5, 0}),
		testPublicDocument("c", "charlie", "low", 3, 4, []float32{4, 0}),
		testPublicDocument("d", "delta", "high", 4, 3, []float32{3, 0}),
		testPublicDocument("e", "echo", "", 5, 6, []float32{6, 0}),
	}
	documents[4].Fields["category"] = nil
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	params := NewFlatQueryParams()
	params.Radius = 4
	results, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10,
		Params: params, Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	{
		got := documentKeys(results)
		require.Equal(t, []string{"e", "b", "c"}, got)
	}
	require.Equal(t, map[string]any{"title": "echo"}, results[0].Fields)
	require.True(t, results[0].Score == 6)

	sparseResults, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}},
		TopK: 3, Projection: Projection{OutputFields: []string{}},
	})
	require.NoError(t, err)
	{
		got := documentKeys(sparseResults)
		require.Equal(t, []string{"e", "b", "c"}, got)
	}
	require.NotNil(t, sparseResults[0].Fields)
	require.Len(t, sparseResults[0].Fields, 0)

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0},
		GroupByField: "category", GroupCount: 3, TopKPerGroup: 2,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	require.Len(t, groups, 3)
	require.True(t, groups[0].Value == "")
	require.True(t, groups[1].Value == "high")
	require.True(t, groups[2].Value == "low")
	{
		got := documentKeys(groups[1].Documents)
		require.Equal(t, []string{"b", "d"}, got)
	}
	{
		got := documentKeys(groups[2].Documents)
		require.Equal(t, []string{"c", "a"}, got)
	}

	filtered, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: "rating > 1"})
	require.NoError(t, err)
	require.Equal(t, []string{"e"}, documentKeys(filtered))
	{
		_, err := collection.Query(ctx, VectorQuery{Field: "embedding", SparseVector: SparseVectorFP32{}, TopK: 1})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := collection.GroupByQuery(ctx, GroupByVectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, GroupByField: "sparse", GroupCount: 1, TopKPerGroup: 1})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
}

func TestCollectionScalarQuantizedDiskANNDirectSchema(t *testing.T) {
	ctx := context.Background()
	for _, quantize := range []QuantizeType{QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		t.Run(quantize.String(), func(t *testing.T) {
			diskANN := NewDiskANNIndexParams(MetricTypeL2)
			diskANN.MaxDegree, diskANN.ListSize, diskANN.PQChunks = 4, 8, 2
			diskANN.Quantize = quantize
			diskANN.Quantizer.EnableRotate = quantize != QuantizeTypeFP16
			schema := NewCollectionSchema("later", FieldSchema{
				Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: diskANN,
			})
			collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "later"), schema, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			{
				_, err := collection.Insert(ctx, []Document{
					{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{0, 0, 0, 0}}},
					{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{1.013, .031, -.077, .125}}},
					{PrimaryKey: "c", Fields: map[string]any{"embedding": VectorFP32{3.117, -.271, .051, -.2}}},
				})
				require.NoError(t, err)
			}

			results, err := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: VectorFP32{.9, .02, -.04, .1}, TopK: 3,
			})
			require.NoError(t, err)
			require.Equal(t, []string{"b", "a", "c"}, documentKeys(results))
			{
				got := collection.Stats().IndexCompleteness["embedding"]
				require.True(t, got == 1)
			}
		})
	}
}

func TestCollectionReplaysPublicDocumentPayloadWithoutFlush(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "recovery")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		_, err := collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 2, []float32{2, 0})})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	reopened, err := Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer reopened.Close()
	results, err := reopened.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1,
		Projection: Projection{IncludeVectors: true},
	})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.True(t, results[0].PrimaryKey == "a")
	require.True(t, results[0].Score == 2)
}

func TestCollectionDestroyAndArgumentErrors(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "destroy")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.Destroy(ctx)
		require.NoError(t, err)
	}
	{
		_, err := os.Stat(path)
		require.ErrorIs(t, err, os.ErrNotExist)
	}
	{
		_, err := Open(ctx, path, NewCollectionOptions())
		require.ErrorIs(t, err, ErrNotFound)
	}
	{
		_, err := CreateAndOpen(nil, path, testPublicCollectionSchema(), NewCollectionOptions())
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	var nilCollection *Collection
	{
		_, err := nilCollection.Insert(ctx, []Document{{PrimaryKey: "a"}})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := nilCollection.Query(ctx, VectorQuery{})
		require.ErrorIs(t, err, ErrInvalidArgument)
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
		require.NoError(t, err)
		require.Equal(t, testCase.want, got)
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
