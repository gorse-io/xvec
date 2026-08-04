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
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gorse-io/zvec/internal/core"
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
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	documents := annDenseDocuments(core.DefaultHNSWBruteForceThreshold + 200)
	if results, err := collection.Insert(ctx, documents); err != nil {
		t.Fatalf("Insert = %#v, %v", results, err)
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
	if err != nil {
		t.Fatal(err)
	}
	queryParams.Linear = true
	query.Params = queryParams
	exact, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	if recall := documentRecall(approximate, exact); recall < .75 {
		t.Fatalf("collection HNSW recall@20 = %.3f, want >= .75", recall)
	}
	for _, document := range approximate {
		if document.Fields["rating"].(int32) < 1 {
			t.Fatalf("filter admitted %#v", document)
		}
	}
	queryParams.Linear = false
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(documentKeys(refined), documentKeys(approximate)) {
		t.Fatalf("unquantized HNSW refiner changed keys: %v vs %v", documentKeys(refined), documentKeys(approximate))
	}
	if got := collection.Stats().IndexCompleteness["embedding"]; got != 1 {
		t.Fatalf("HNSW completeness = %v", got)
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
	if err != nil {
		t.Fatal(err)
	}
	documents := annRaBitQDocuments(180)
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
	}
	queryVector := documents[73].Fields["embedding"].(VectorFP32)
	exact := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, 15)

	indexParams := NewHNSWRaBitQIndexParams(MetricTypeL2)
	indexParams.TotalBits = 7
	indexParams.NumClusters = 8
	indexParams.M = 8
	indexParams.EFConstruction = 40
	if err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3}); err != nil {
		t.Fatal(err)
	}
	defaulted, err := collection.Query(ctx, VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(defaulted) != 5 {
		t.Fatalf("default HNSW-RaBitQ query returned %d documents", len(defaulted))
	}
	queryParams := NewHNSWRaBitQQueryParams()
	queryParams.EF = 100
	query := VectorQuery{Field: "embedding", DenseVector: queryVector, TopK: 15, Params: queryParams}
	approximate, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	if recall := documentRecall(approximate, exact); recall < .85 {
		t.Fatalf("collection HNSW-RaBitQ recall@15 = %.3f", recall)
	}
	queryParams.Linear = true
	queryParams.UseRefiner = true
	query.Params = queryParams
	refined, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(documentKeys(refined), documentKeys(exact)) || !reflect.DeepEqual(documentScores(refined), documentScores(exact)) {
		t.Fatalf("linear refined HNSW-RaBitQ differs: keys %v vs %v, scores %v vs %v", documentKeys(refined), documentKeys(exact), documentScores(refined), documentScores(exact))
	}

	queryParams.Linear = false
	queryParams.UseRefiner = false
	queryParams.Radius = approximate[len(approximate)-1].Score
	query.Params = queryParams
	query.Filter = "rating >= 1"
	filtered, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	for _, document := range filtered {
		if document.Fields["rating"].(int32) < 1 || document.Score > queryParams.Radius {
			t.Fatalf("filter/radius admitted %#v", document)
		}
	}
	if got := collection.Stats().IndexCompleteness["embedding"]; got != 1 {
		t.Fatalf("HNSW-RaBitQ completeness = %v", got)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2}); err != nil {
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
	reopened, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reopened, filtered) {
		t.Fatal("reopened HNSW-RaBitQ query differs")
	}
	field, _ := collection.Schema().Field("embedding")
	if field.IndexType() != IndexTypeHNSWRaBitQ || collection.Stats().IndexCompleteness["embedding"] != 1 {
		t.Fatalf("reopened HNSW-RaBitQ state = %#v, %#v", field, collection.Stats())
	}
}

func TestCollectionQuantizedIVFRefinementCreateIndexAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "ivf")
	schema := NewCollectionSchema("quantized_ivf",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	documents := annDenseDocuments(320)
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
	}
	queryVector := documents[217].Fields["embedding"].(VectorFP32)
	exact, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 15, Filter: "rating >= 1",
	})
	if err != nil {
		t.Fatal(err)
	}
	indexParams := NewIVFIndexParams(MetricTypeL2)
	indexParams.NList = 16
	indexParams.NIterations = 12
	indexParams.Quantize = QuantizeTypeInt8
	indexParams.Quantizer.EnableRotate = true
	if err := collection.CreateIndex(ctx, "embedding", indexParams, CreateIndexOptions{Concurrency: 3}); err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(documentKeys(refined), documentKeys(exact)) {
		t.Fatalf("refined IVF keys = %v, exact %v", documentKeys(refined), documentKeys(exact))
	}
	for position := range refined {
		if refined[position].Score != exact[position].Score {
			t.Fatalf("refined score %d = %v, exact %v", position, refined[position].Score, exact[position].Score)
		}
	}
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2}); err != nil {
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
	reopened, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reopened, refined) {
		t.Fatalf("reopened refined IVF differs")
	}
	field, _ := collection.Schema().Field("embedding")
	if field.IndexType() != IndexTypeIVF || collection.Stats().IndexCompleteness["embedding"] != 1 {
		t.Fatalf("reopened IVF state = %#v, %#v", field, collection.Stats())
	}
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
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	documents := annDenseDocuments(80)
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
	}
	queryVector := VectorFP32{3.25, -1.5, 7, 2}
	queryParams := NewFlatQueryParams()
	queryParams.UseRefiner = true
	queryParams.ScaleFactor = 100
	refined, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: queryVector, TopK: 12, Params: queryParams,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := exactDenseDocumentResults(t, documents, queryVector, core.MetricL2, 12)
	if !reflect.DeepEqual(documentKeys(refined), documentKeys(want)) {
		t.Fatalf("refined quantized Flat = %v, exact %v", documentKeys(refined), documentKeys(want))
	}
	for position := range refined {
		if refined[position].Score != want[position].Score {
			t.Fatalf("refined score %d = %v, want %v", position, refined[position].Score, want[position].Score)
		}
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
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	documents := make([]Document, 240)
	for position := range documents {
		documents[position] = Document{PrimaryKey: fmt.Sprintf("s%03d", position), Fields: map[string]any{
			"sparse": SparseVectorFP32{
				Indices: []uint32{uint32(position % 31), uint32(100 + position%37), uint32(200 + position%43)},
				Values:  []float32{float32(position%7) + .25, float32(position%11) + .5, float32(position%13) + .75},
			},
			"rating": int32(position % 3),
		}}
	}
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
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
	if err != nil {
		t.Fatal(err)
	}
	queryParams.Linear = true
	query.Params = queryParams
	want, err := collection.Query(ctx, query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("small sparse HNSW differs from linear: %#v vs %#v", got, want)
	}
	queryParams.Linear = false
	queryParams.UseRefiner = true
	query.Params = queryParams
	if _, err := collection.Query(ctx, query); !errors.Is(err, ErrNotSupported) {
		t.Fatalf("sparse refiner error = %v", err)
	}
}

func TestCollectionANNValidationAndBackfillRollback(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("ann_validation",
		FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: NewFlatIndexParams(MetricTypeL2)},
		FieldSchema{Name: "group", DataType: DataTypeString},
	)
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "validation"), schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, []Document{{PrimaryKey: "huge", Fields: map[string]any{
		"embedding": VectorFP32{70000, 1, 2, 3}, "group": "g",
	}}}); err != nil {
		t.Fatal(err)
	}
	before := collection.Schema()
	generation := collection.store.Manifest().Generation
	quantized := NewIVFIndexParams(MetricTypeL2)
	quantized.NList = 1
	quantized.Quantize = QuantizeTypeFP16
	if err := collection.CreateIndex(ctx, "embedding", quantized, CreateIndexOptions{}); err == nil {
		t.Fatal("FP16 overflow backfill succeeded")
	}
	if !reflect.DeepEqual(collection.Schema(), before) || collection.store.Manifest().Generation != generation {
		t.Fatal("failed ANN backfill changed schema generation")
	}
	soar := NewIVFIndexParams(MetricTypeL2)
	soar.UseSOAR = true
	if err := collection.CreateIndex(ctx, "embedding", soar, CreateIndexOptions{}); !errors.Is(err, ErrNotSupported) {
		t.Fatalf("IVF SOAR error = %v", err)
	}
	if !reflect.DeepEqual(collection.Schema(), before) || collection.store.Manifest().Generation != generation {
		t.Fatal("unsupported IVF SOAR changed schema generation")
	}

	hnswParams := NewHNSWQueryParams()
	hnswParams.EF = MaxGraphEFSearch + 1
	if _, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, TopK: 1, Params: hnswParams,
	}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("mismatched query params error = %v", err)
	}

	index := NewHNSWIndexParams(MetricTypeL2)
	if err := collection.CreateIndex(ctx, "embedding", index, CreateIndexOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, TopK: 1, Params: hnswParams,
	}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("oversized HNSW EF error = %v", err)
	}
	if _, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4},
		GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
	}); !errors.Is(err, ErrNotSupported) {
		t.Fatalf("ANN group-by error = %v", err)
	}
	linear := NewHNSWQueryParams()
	linear.Linear = true
	if _, err := collection.GroupByQuery(ctx, GroupByVectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 2, 3, 4}, Params: linear,
		GroupByField: "group", GroupCount: 1, TopKPerGroup: 1,
	}); err != nil {
		t.Fatalf("linear ANN group-by = %v", err)
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
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	results, err := collection.Insert(ctx, []Document{{
		PrimaryKey: "overflow", Fields: map[string]any{"embedding": VectorFP32{70000, 1}},
	}})
	if err == nil || len(results) != 1 || !errors.Is(results[0].Err, ErrInvalidArgument) {
		t.Fatalf("quantized overflow write = %#v, %v", results, err)
	}
	if collection.Stats().DocumentCount != 0 {
		t.Fatal("failed quantized write changed document count")
	}
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
	if err != nil {
		t.Fatal(err)
	}
	output := make([]Document, len(results))
	for position, result := range results {
		output[position] = byID[result.Key]
		output[position].Score = result.Score
	}
	return output
}
