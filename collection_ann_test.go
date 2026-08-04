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
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/core"
	"github.com/stretchr/testify/require"
)

func TestCollectionDenseHNSWQueryControlsAndRecall(t *testing.T) {
	ctx := context.Background()
	params := NewHNSWIndexParams(MetricTypeL2)
	params.M = 12
	params.EFConstruction = 80
	schema := NewCollectionSchema("dense_hnsw",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "hnsw"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := annDenseDocuments(core.DefaultHNSWBruteForceThreshold + 200)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[713].Fields["embedding"].(VectorFP32)
	queryParams := NewHNSWQueryParams()
	queryParams.EF = 120
	queryParams.PrefetchOffset = math.MaxUint32
	queryParams.PrefetchLines = math.MaxUint32
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 20,
		Filter: "rating >= 1", Params: queryParams,
	}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)

	queryParams.Linear = true
	query.Params = queryParams
	exact, err := collection.Query(ctx, query)
	require.NoError(t, err)
	{
		recall := documentRecall(approximate, exact)
		require.True(t, recall >= .75)
	}

	for _, document := range approximate {
		require.True(t, document.Fields["rating"].(int32) >= 1)
	}
	queryParams.Linear = false
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(approximate), documentKeys(refined))
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
}

func TestCollectionHNSWRaBitQQueryCreateIndexOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "rabitq")
	schema := NewCollectionSchema("rabitq_collection",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 64, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annRaBitQDocuments(180)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[73].Fields["embedding"].(VectorFP32)
	exact := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, 15)

	indexParams := NewHNSWRaBitQIndexParams(MetricTypeL2)
	indexParams.TotalBits = 7
	indexParams.NumClusters = 8
	indexParams.M = 8
	indexParams.EFConstruction = 40
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	defaulted, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 5})
	require.NoError(t, err)
	require.Len(t, defaulted, 5)

	queryParams := NewHNSWRaBitQQueryParams()
	queryParams.EF = 100
	query := VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 15, Params: queryParams}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)
	{
		recall := documentRecall(approximate, exact)
		require.True(t, recall >= .85)
	}

	queryParams.Linear = true
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(exact), documentKeys(refined))
	require.Equal(t, documentScores(exact), documentScores(refined))

	queryParams.Linear = false
	queryParams.UseRefiner = false
	queryParams.Radius = approximate[len(approximate)-1].Score
	query.Params = queryParams
	query.Filter = "rating >= 1"
	filtered, err := collection.Query(ctx, query)
	require.NoError(t, err)

	for _, document := range filtered {
		require.True(t, document.Fields["rating"].(int32) >= 1)
		require.True(t, document.Score <= queryParams.Radius)
	}
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, filtered, reopened,
		"reopened HNSW-RaBitQ query differs")

	field, _ := collection.Schema().Field("embedding")
	require.Equal(t, IndexTypeHNSWRaBitQ, field.IndexType())
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionVamanaQueryCreateIndexQuantizeOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "vamana")
	schema := NewCollectionSchema("vamana_collection",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annDenseDocuments(320)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[217].Fields["embedding"].(VectorFP32)

	indexParams := NewVamanaIndexParams(MetricTypeL2)
	indexParams.MaxDegree = 12
	indexParams.SearchListSize = 60
	indexParams.MaxOcclusionSize = 120
	indexParams.SaturateGraph = true
	indexParams.UseContiguousMemory = true
	indexParams.UseIDMap = true
	indexParams.Quantize = QuantizeTypeInt8
	indexParams.Quantizer.EnableRotate = true
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	defaulted, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 5})
	require.NoError(t, err)
	require.Len(t, defaulted, 5)

	queryParams := NewVamanaQueryParams()
	queryParams.EFSearch = 100
	queryParams.PrefetchOffset = math.MaxUint32
	queryParams.PrefetchLines = math.MaxUint32
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: queryParams,
	}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, approximate, 15)

	for _, document := range approximate {
		require.True(t, document.Fields["rating"].(int32) >= 1)
	}
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)

	byKey := make(map[string]Document, len(documents))
	for _, document := range documents {
		byKey[document.PrimaryKey] = document
	}
	for _, document := range refined {
		original := byKey[document.PrimaryKey].Fields["embedding"].(VectorFP32)
		want, err := core.MetricL2.Compute(queryVector, original)
		require.NoError(t, err)
		require.Equal(t, want, document.Score)
	}
	queryParams.UseRefiner = false
	queryParams.Radius = approximate[len(approximate)-1].Score
	query.Params = queryParams
	bounded, err := collection.Query(ctx, query)
	require.NoError(t, err)

	for _, document := range bounded {
		require.True(t, document.Score <= queryParams.Radius)
	}
	queryParams.Linear = true
	queryParams.Radius = 0
	query.Params = queryParams
	linear, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, linear, 15)
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, linear, reopened,
		"reopened Vamana query differs")

	field, _ := collection.Schema().Field("embedding")
	require.Equal(t, IndexTypeVamana, field.IndexType())
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionDiskANNQueryCreateIndexRefineOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "diskann")
	schema := NewCollectionSchema("diskann_collection",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annDenseDocuments(320)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[217].Fields["embedding"].(VectorFP32)

	indexParams := NewDiskANNIndexParams(MetricTypeL2)
	indexParams.MaxDegree = 12
	indexParams.ListSize = 60
	indexParams.PQChunks = 2
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	defaulted, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 5})
	require.NoError(t, err)
	require.Len(t, defaulted, 5)

	params := NewDiskANNQueryParams()
	params.ListSize = 100
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: params,
	}
	approximate, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, approximate, 15)

	for _, document := range approximate {
		require.True(t, document.Fields["rating"].(int32) >= 1)
	}

	params.UseRefiner = true
	query.Params = params
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)

	byKey := make(map[string]Document, len(documents))
	for _, document := range documents {
		byKey[document.PrimaryKey] = document
	}
	for _, document := range refined {
		original := byKey[document.PrimaryKey].Fields["embedding"].(VectorFP32)
		want, err := core.MetricL2.Compute(queryVector, original)
		require.NoError(t, err)
		require.Equal(t, want, document.Score)
	}

	params.UseRefiner = false
	params.Radius = approximate[len(approximate)-1].Score
	query.Params = params
	bounded, err := collection.Query(ctx, query)
	require.NoError(t, err)

	for _, document := range bounded {
		require.True(t, document.Score <= params.Radius)
	}
	params.Linear = true
	params.Radius = 0
	query.Params = params
	linear, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, linear, 15)
	{
		recall := documentRecall(approximate, linear)
		require.True(t, recall >= .80)
	}
	{
		got := collection.Stats().IndexCompleteness["embedding"]
		require.True(t, got == 1)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, linear, reopened,
		"reopened DiskANN query differs")

	field, _ := collection.Schema().Field("embedding")
	require.Equal(t, IndexTypeDiskANN, field.IndexType())
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionDiskANNDirectFP16SchemaDefaults(t *testing.T) {
	ctx := context.Background()
	params := NewDiskANNIndexParams(MetricTypeL2)
	params.MaxDegree, params.ListSize, params.PQChunks = 4, 8, 1
	schema := NewCollectionSchema("diskann_direct",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP16, Dimension: 2, Index: params},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "direct"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{
			{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP16{Float16FromFloat32(0), Float16FromFloat32(0)}}},
			{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP16{Float16FromFloat32(1), Float16FromFloat32(0)}}},
			{PrimaryKey: "c", Fields: map[string]any{"embedding": VectorFP16{Float16FromFloat32(3), Float16FromFloat32(0)}}},
		})
		require.NoError(t, err)
	}

	results, err := collection.Query(ctx, VectorQuery{
		Field:       "embedding",
		DenseVector: VectorFP16{Float16FromFloat32(0.9), Float16FromFloat32(0)},
		TopK:        2,
	})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(results))
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionScalarQuantizedDiskANNBackfillRefineOptimizeAndReopen(t *testing.T) {
	ctx := context.Background()
	for _, quantize := range []QuantizeType{QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		t.Run(quantize.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "diskann")
			schema := NewCollectionSchema("quantized_diskann",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
				FieldSchema{Name: "rating", DataType: DataTypeInt32},
			)
			schema.MaxDocsPerSegment = MinMaxDocsPerSegment
			collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			require.NoError(t, err)

			documents := annDenseDocuments(48)
			{
				_, err := collection.Insert(ctx, documents)
				require.NoError(t, err)
			}

			indexParams := NewDiskANNIndexParams(MetricTypeL2)
			indexParams.MaxDegree, indexParams.ListSize, indexParams.PQChunks = 8, len(documents), 2
			indexParams.Quantize = quantize
			indexParams.Quantizer.EnableRotate = quantize != QuantizeTypeFP16
			{
				err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 2})
				require.NoError(t, err)
			}

			queryVector := append(VectorFP32(nil), documents[17].Fields["embedding"].(VectorFP32)...)
			queryVector[1] += .137
			queryParams := NewDiskANNQueryParams()
			queryParams.ListSize = len(documents)
			query := VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 12, Params: queryParams}
			graph, err := collection.Query(ctx, query)
			require.NoError(t, err)

			queryParams.Linear = true
			query.Params = queryParams
			linear, err := collection.Query(ctx, query)
			require.NoError(t, err)
			require.Equal(t, linear, graph)

			byKey := make(map[string]Document, len(documents))
			for _, document := range documents {
				byKey[document.PrimaryKey] = document
			}
			quantizedDifference := false
			for _, document := range linear {
				original := byKey[document.PrimaryKey].Fields["embedding"].(VectorFP32)
				score, err := core.MetricL2.Compute(queryVector, original)
				require.NoError(t, err)

				if document.Score != score {
					quantizedDifference = true
					break
				}
			}
			require.True(t, quantizedDifference,
				"DiskANN scalar quantization did not affect any first-stage score")

			queryParams.Linear = false
			queryParams.UseRefiner = true
			query.Params = queryParams
			refined, err := collection.Query(ctx, query)
			require.NoError(t, err)

			exact := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, query.TopK)
			require.Equal(t, documentKeys(exact), documentKeys(refined))
			require.Equal(t, documentScores(exact), documentScores(refined))

			queryParams.UseRefiner = false
			queryParams.Radius = graph[7].Score
			query.Filter = "rating >= 1"
			query.Params = queryParams
			bounded, err := collection.Query(ctx, query)
			require.NoError(t, err)

			for _, document := range bounded {
				require.True(t, document.Fields["rating"].(int32) >= 1)
				require.True(t, document.Score <= queryParams.Radius)
			}
			require.False(t, len(bounded) == 0,
				"filter/radius removed every document")

			if quantize == QuantizeTypeFP16 {
				results, err := collection.Insert(ctx, []Document{{
					PrimaryKey: "overflow", Fields: map[string]any{"embedding": VectorFP32{70000, 0, 0, 0}, "rating": int32(1)},
				}})
				require.Error(t, err)
				require.Len(t, results, 1)
				require.ErrorIs(t, results[0].Err, ErrInvalidArgument)
			}
			{
				_, err := collection.Delete(ctx, []string{"d0003"})
				require.NoError(t, err)
			}
			{
				err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
				require.NoError(t, err)
			}
			{
				got := collection.Stats().IndexCompleteness["embedding"]
				require.True(t, got == 1)
			}

			beforeReopen, err := collection.Query(ctx, query)
			require.NoError(t, err)
			{
				err := collection.Close()
				require.NoError(t, err)
			}

			collection, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			reopened, err := collection.Query(ctx, query)
			require.NoError(t, err)
			require.Equal(t, beforeReopen, reopened)

			field, _ := collection.Schema().Field("embedding")
			persisted := field.Index.(DiskANNIndexParams)
			require.Equal(t, quantize, persisted.Quantize)
			require.Equal(t, quantize != QuantizeTypeFP16, persisted.Quantizer.EnableRotate)
		})
	}
}

func TestCollectionQuantizedIVFSOARRefinementCreateIndexAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "ivf")
	schema := NewCollectionSchema("quantized_ivf",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := annDenseDocuments(320)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := documents[217].Fields["embedding"].(VectorFP32)
	exact, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15, Filter: "rating >= 1",
	})
	require.NoError(t, err)

	indexParams := NewIVFIndexParams(MetricTypeL2)
	indexParams.NList = 16
	indexParams.NIterations = 12
	indexParams.UseSOAR = true
	indexParams.Quantize = QuantizeTypeInt8
	indexParams.Quantizer.EnableRotate = true
	standardParams := indexParams
	standardParams.UseSOAR = false
	{
		err := collection.CreateIndex(ctx, "embedding", standardParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	compatibilityParams := NewIVFQueryParams()
	compatibilityParams.NProbe = 4
	compatibilityQuery := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: compatibilityParams,
	}
	standardResults, err := collection.Query(ctx, compatibilityQuery)
	require.NoError(t, err)
	{
		err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	soarResults, err := collection.Query(ctx, compatibilityQuery)
	require.NoError(t, err)
	require.Equal(t, standardResults, soarResults)

	queryParams := NewIVFQueryParams()
	queryParams.NProbe = 16
	queryParams.UseRefiner = true
	queryParams.ScaleFactor = 100
	queryParams.Radius = exact[len(exact)-1].Score
	query := VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15,
		Filter: "rating >= 1", Params: queryParams,
	}
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(exact), documentKeys(refined))

	for position := range refined {
		require.Equal(t, exact[position].Score, refined[position].Score)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, refined, reopened,
		"reopened refined IVF differs")

	field, _ := collection.Schema().Field("embedding")
	persisted := field.Index.(IVFIndexParams)
	require.Equal(t, IndexTypeIVF, field.IndexType())
	require.True(t, persisted.UseSOAR)
	require.True(t, collection.Stats().IndexCompleteness["embedding"] == 1)
}

func TestCollectionQuantizedFlatRotationAndRefiner(t *testing.T) {
	ctx := context.Background()
	params := NewFlatIndexParams(MetricTypeL2)
	params.Quantize = QuantizeTypeInt4
	params.Quantizer.EnableRotate = true
	schema := NewCollectionSchema("quantized_flat",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "flat"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := annDenseDocuments(80)
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := VectorFP32{3.25, -1.5, 7, 2}
	queryParams := NewFlatQueryParams()
	queryParams.UseRefiner = true
	queryParams.ScaleFactor = 100
	refined, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 12, Params: queryParams,
	})
	require.NoError(t, err)

	want := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, 12)
	require.Equal(t, documentKeys(want), documentKeys(refined))

	for position := range refined {
		require.Equal(t, want[position].Score, refined[position].Score)
	}
}

func TestCollectionSparseHNSWFP16Controls(t *testing.T) {
	ctx := context.Background()
	params := NewHNSWIndexParams(MetricTypeIP)
	params.M = 8
	params.EFConstruction = 40
	params.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("sparse_hnsw",
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "sparse"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	documents := make([]Document, 240)
	for position := range documents {
		documents[position] = Document{PrimaryKey: fmt.Sprintf("s%03d", position), Fields: map[string]any{
			"sparse": SparseVectorFP32{
				Indices: []uint32{uint32(position % 31), uint32(100 + position%37), uint32(200 + position%43)},
				Values:  []float32{float32(position%7) + .12345, float32(position%11) + .33331, float32(position%13) + .77771},
			},
			"rating": int32(position % 3),
		}}
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryParams := NewHNSWQueryParams()
	queryParams.EF = 24
	queryParams.PrefetchOffset = 8
	queryParams.PrefetchLines = 0
	queryParams.Radius = 1
	query := VectorQuery{
		Field: "sparse", SparseVector: documents[117].Fields["sparse"].(SparseVectorFP32),
		TopK: 20, Filter: "rating >= 1", Params: queryParams,
	}
	got, err := collection.Query(ctx, query)
	require.NoError(t, err)

	queryParams.Linear = true
	query.Params = queryParams
	want, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)

	queryParams.Linear = false
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)

	approximateByKey := make(map[string]float32, len(got))
	for _, document := range got {
		approximateByKey[document.PrimaryKey] = document.Score
	}
	originalByKey := make(map[string]SparseVectorFP32, len(documents))
	for _, document := range documents {
		originalByKey[document.PrimaryKey] = document.Fields["sparse"].(SparseVectorFP32)
	}
	changedScore := false
	querySparse := query.SparseVector.(SparseVectorFP32)
	for _, document := range refined {
		require.True(t, document.Fields["rating"].(int32) >= 1)
		require.True(t, document.Score >= queryParams.Radius)

		original := originalByKey[document.PrimaryKey]
		exact, err := ailego.SparseInnerProduct(querySparse.Indices, querySparse.Values, original.Indices, original.Values)
		require.NoError(t, err)
		require.Equal(t, exact, document.Score)

		if approximate, found := approximateByKey[document.PrimaryKey]; found && approximate != exact {
			changedScore = true
		}
	}
	require.False(t, len(refined) == 0)
	require.True(t, changedScore)
}

func TestCollectionSparseFlatFP16RefinementMultiQueryAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "sparse-flat-refine")
	params := NewFlatIndexParams(MetricTypeIP)
	params.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("sparse_flat_refine",
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: params},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		{PrimaryKey: "a", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 3}, Values: []float32{1.0003, 4.0007}}, "rating": int32(1)}},
		{PrimaryKey: "b", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 2}, Values: []float32{2.0009, 5.0011}}, "rating": int32(2)}},
		{PrimaryKey: "c", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{1, 3}, Values: []float32{9.0021, 1.0004}}, "rating": int32(3)}},
		{PrimaryKey: "d", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 1, 3}, Values: []float32{1.5006, 2.0013, 2.5008}}, "rating": int32(4)}},
		{PrimaryKey: "e", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{2, 3}, Values: []float32{8.0005, 3.0009}}, "rating": int32(5)}},
		{PrimaryKey: "f", Fields: map[string]any{"sparse": SparseVectorFP32{Indices: []uint32{0, 2}, Values: []float32{.5003, 1.0007}}, "rating": int32(6)}},
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}

	queryVector := SparseVectorFP32{Indices: []uint32{0, 3}, Values: []float32{2.0007, 1.0003}}
	queryParams := NewFlatQueryParams()
	queryParams.UseRefiner = true
	queryParams.ScaleFactor = 100
	exact := exactSparseDocumentResults(t, documents, queryVector, len(documents))
	queryParams.Radius = exact[4].Score
	query := VectorQuery{Field: "sparse", SparseVector: queryVector, TopK: 5, Params: queryParams}
	refined, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, documentKeys(exact[:5]), documentKeys(refined))
	require.Equal(t, documentScores(exact[:5]), documentScores(refined))

	unrefinedParams := queryParams
	unrefinedParams.UseRefiner = false
	unrefinedParams.Radius = 0
	unrefined, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: queryVector, TopK: 5, Params: unrefinedParams,
	})
	require.NoError(t, err)
	require.NotEqual(t, documentScores(refined), documentScores(unrefined),
		"FP16 sparse first-stage scores unexpectedly equal original-vector scores")

	alternate := SparseVectorFP32{Indices: []uint32{1, 2}, Values: []float32{.7503, 1.2509}}
	alternateExact := exactSparseDocumentResults(t, documents, alternate, 4)
	alternateParams := queryParams
	alternateParams.Radius = 0
	multi := MultiQuery{
		Queries: []SubQuery{
			{Field: "sparse", SparseVector: queryVector, Params: queryParams, NumCandidates: 5},
			{Field: "sparse", SparseVector: alternate, Params: alternateParams, NumCandidates: 4},
		},
		TopK: 2,
		Reranker: testRerankerFunc(func(_ context.Context, batches []RerankBatch, _ int) ([]Document, error) {
			require.Equal(t, documentKeys(exact[:5]), documentKeys(batches[0].Documents))
			require.Equal(t, documentScores(exact[:5]), documentScores(batches[0].Documents))
			require.Equal(t, documentKeys(alternateExact), documentKeys(batches[1].Documents))
			require.Equal(t, documentScores(alternateExact), documentScores(batches[1].Documents))

			return []Document{batches[0].Documents[0], batches[1].Documents[0]}, nil
		}),
	}
	beforeReopen, err := collection.MultiQuery(ctx, multi)
	require.NoError(t, err)
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	reopened, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, refined, reopened)

	reopenedMulti, err := collection.MultiQuery(ctx, multi)
	require.NoError(t, err)
	require.Equal(t, beforeReopen, reopenedMulti)
}

func TestCollectionANNValidationAndBackfillRollback(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("ann_validation",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "group", DataType: DataTypeString},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "validation"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, []Document{{PrimaryKey: "huge", Fields: map[string]any{
			"embedding": VectorFP32{70000, 1, 2, 3}, "group": "g",
		}}})
		require.NoError(t, err)
	}

	before := collection.Schema()
	generation := collection.store.Manifest().Generation
	quantized := NewIVFIndexParams(MetricTypeL2)
	quantized.NList = 1
	quantized.Quantize = QuantizeTypeFP16
	{
		err := collection.CreateIndex(ctx, "embedding", quantized, CreateIndexOptions{})
		require.Error(t, err,
			"FP16 overflow backfill succeeded")
	}
	require.Equal(t, before, collection.Schema(),
		"failed ANN backfill changed schema generation")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed ANN backfill changed schema generation")

	vamana := NewVamanaIndexParams(MetricTypeL2)
	vamana.MaxDegree, vamana.SearchListSize, vamana.MaxOcclusionSize = 4, 8, 16
	vamana.Quantize = QuantizeTypeFP16
	{
		err := collection.CreateIndex(ctx, "embedding", vamana, CreateIndexOptions{})
		require.Error(t, err,
			"Vamana FP16 overflow backfill succeeded")
	}
	require.Equal(t, before, collection.Schema(),
		"failed Vamana backfill changed schema generation")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed Vamana backfill changed schema generation")

	diskANN := NewDiskANNIndexParams(MetricTypeL2)
	diskANN.MaxDegree, diskANN.ListSize, diskANN.PQChunks = 4, 8, 2
	diskANN.Quantize = QuantizeTypeFP16
	{
		err := collection.CreateIndex(ctx, "embedding", diskANN, CreateIndexOptions{})
		require.Error(t, err,
			"DiskANN FP16 overflow backfill succeeded")
	}
	require.Equal(t, before, collection.Schema(),
		"failed DiskANN backfill changed schema generation")
	require.Equal(t, generation, collection.store.Manifest().Generation,
		"failed DiskANN backfill changed schema generation")

	soar := NewIVFIndexParams(MetricTypeL2)
	soar.UseSOAR = true
	{
		err := collection.CreateIndex(ctx, "embedding", soar, CreateIndexOptions{})
		require.NoError(t, err)
	}

	soarField, _ := collection.Schema().Field("embedding")
	{
		persisted := soarField.Index.(IVFIndexParams)
		require.True(t, persisted.UseSOAR)
		require.Equal(t, generation+1, collection.store.Manifest().Generation)
	}

	hnswParams := NewHNSWQueryParams()
	hnswParams.EF = MaxGraphEFSearch + 1
	{
		_, err := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, TopK: 1, Params: hnswParams,
		})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	index := NewHNSWIndexParams(MetricTypeL2)
	{
		err := collection.CreateIndex(ctx, "embedding", index, CreateIndexOptions{})
		require.NoError(t, err)
	}
	{
		_, err := collection.Query(ctx, VectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, TopK: 1, Params: hnswParams,
		})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	groups, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4},
		GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
	})
	require.NoError(t, err)
	require.Len(t, groups, 1)
	require.True(t, groups[0].Value == "g")

	refined := NewHNSWQueryParams()
	refined.UseRefiner = true
	{
		_, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, Params: refined,
			GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
		})
		require.ErrorIs(t, err, ErrNotSupported)
	}

	linear := NewHNSWQueryParams()
	linear.Linear = true
	{
		_, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
			Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, Params: linear,
			GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
		})
		require.NoError(t, err)
	}
}

func TestCollectionGroupByPreservesUnsupportedANNBoundary(t *testing.T) {
	tests := []struct {
		name   string
		index  IndexParams
		params func(linear bool) QueryParams
	}{
		{
			name: "IVF", index: NewIVFIndexParams(MetricTypeL2),
			params: func(linear bool) QueryParams {
				params := NewIVFQueryParams()
				params.Linear = linear
				return params
			},
		},
		{
			name: "Vamana", index: NewVamanaIndexParams(MetricTypeL2),
			params: func(linear bool) QueryParams {
				params := NewVamanaQueryParams()
				params.Linear = linear
				return params
			},
		},
		{
			name: "DiskANN", index: NewDiskANNIndexParams(MetricTypeL2),
			params: func(linear bool) QueryParams {
				params := NewDiskANNQueryParams()
				params.Linear = linear
				return params
			},
		},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := context.Background()
			schema := NewCollectionSchema("unsupported_native_group",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: testCase.index},
				FieldSchema{Name: "group", DataType: DataTypeString},
			)
			collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "collection"), schema, NewCollectionOptions())
			require.NoError(t, err)

			defer collection.Close()
			{
				_, err := collection.Insert(ctx, []Document{
					{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{0, 0, 0, 0}, "group": "a"}},
					{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{1, 1, 1, 1}, "group": "b"}},
				})
				require.NoError(t, err)
			}

			query := GroupByVectorQuery{
				Field: "embedding", DenseVector: VectorFP32{0, 0, 0, 0},
				GroupByField: "group", GroupCount: 2, TopKPerGroup: 1,
				Params: testCase.params(false),
			}
			{
				_, err := collection.GroupByQuery(ctx, query)
				require.ErrorIs(t, err, ErrNotSupported)
			}

			query.Params = testCase.params(true)
			groups, err := collection.GroupByQuery(ctx, query)
			require.NoError(t, err)
			require.Len(t, groups, 2)
		})
	}
}

func TestCollectionQuantizedWriteRejectsUnrepresentableVector(t *testing.T) {
	ctx := context.Background()
	params := NewFlatIndexParams(MetricTypeL2)
	params.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("quantized_write", FieldSchema{
		Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: params,
	})
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "write"), schema, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	results, err := collection.Insert(ctx, []Document{{
		PrimaryKey: "overflow", Fields: map[string]any{"embedding": VectorFP32{70000, 1}},
	}})
	require.Error(t, err)
	require.Len(t, results, 1)
	require.ErrorIs(t, results[0].Err, ErrInvalidArgument)
	require.True(t, collection.Stats().DocumentCount == 0,
		"failed quantized write changed document count")
}

func annDenseDocuments(count int) []Document {
	documents := make([]Document, count)
	for position := range documents {
		documents[position] = Document{PrimaryKey: fmt.Sprintf("d%04d", position), Fields: map[string]any{
			"embedding": VectorFP32{
				float32(position%23) - 7.25,
				float32((position*7)%31) - 11.5,
				float32((position*13)%37) + .125,
				float32((position*19)%41) - 3.75,
			},
			"rating": int32(position % 3),
		}}
	}
	return documents
}

func annRaBitQDocuments(count int) []Document {
	documents := make([]Document, count)
	for position := range documents {
		vector := make(VectorFP32, 64)
		for dimension := range vector {
			vector[dimension] = float32(int((position*37+dimension*17+position*dimension*3)%211)-105) / 19
		}
		documents[position] = Document{PrimaryKey: fmt.Sprintf("r%04d", position), Fields: map[string]any{
			"embedding": vector,
			"rating":    int32(position % 3),
		}}
	}
	return documents
}

func documentRecall(got, want []Document) float64 {
	keys := make(map[string]struct{}, len(want))
	for _, document := range want {
		keys[document.PrimaryKey] = struct{}{}
	}
	matched := 0
	for _, document := range got {
		if _, found := keys[document.PrimaryKey]; found {
			matched++
		}
	}
	if len(want) == 0 {
		return 1
	}
	return float64(matched) / float64(len(want))
}

func documentScores(documents []Document) []float32 {
	scores := make([]float32, len(documents))
	for index := range documents {
		scores[index] = documents[index].Score
	}
	return scores
}

func exactDenseDocumentResults(t testing.TB, documents []Document, query VectorFP32, metric core.Metric, topK int) []Document {
	t.Helper()
	candidates := make([]core.Candidate, len(documents))
	byID := make(map[uint64]Document, len(documents))
	for position, document := range documents {
		document.DocID = uint64(position)
		candidates[position] = core.Candidate{Key: uint64(position), Vector: []float32(document.Fields["embedding"].(VectorFP32))}
		byID[uint64(position)] = document
	}
	results, err := core.TopK(context.Background(), metric, []float32(query), candidates, topK)
	require.NoError(t, err)

	output := make([]Document, len(results))
	for position, result := range results {
		output[position] = byID[result.Key]
		output[position].Score = result.Score
	}
	return output
}

func exactSparseDocumentResults(t testing.TB, documents []Document, query SparseVectorFP32, topK int) []Document {
	t.Helper()
	index, err := core.NewSparseFlatIndex(core.MetricIP)
	require.NoError(t, err)

	byID := make(map[uint64]Document, len(documents))
	for position, document := range documents {
		vector, err := sparseValueToCore(document.Fields["sparse"])
		require.NoError(t, err)

		key := uint64(position)
		{
			err := index.AddSparse(context.Background(), key, vector)
			require.NoError(t, err)
		}

		document.DocID = key
		byID[key] = document
	}
	queryVector, err := sparseValueToCore(query)
	require.NoError(t, err)

	results, err := index.SearchSparseWithOptions(context.Background(), queryVector, core.SearchOptions{TopK: topK})
	require.NoError(t, err)

	output := make([]Document, len(results))
	for position, result := range results {
		output[position] = byID[result.Key]
		output[position].Score = result.Score
	}
	return output
}
