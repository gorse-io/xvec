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

package xvec_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gorse-io/xvec"
)

func ExampleRuntimeConfig_Validate() {
	config := xvec.NewRuntimeConfig()
	config.MemoryLimitBytes = xvec.MinRuntimeMemoryLimit
	config.QueryConcurrency = 4
	config.OptimizeConcurrency = 2

	fmt.Println(config.Validate() == nil)

	// Output:
	// true
}

func ExampleCollection_Query() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")

	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "title", DataType: xvec.DataTypeString},
		xvec.FieldSchema{
			Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 2,
			Index: xvec.NewFlatIndexParams(xvec.MetricTypeIP),
		},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{
		{PrimaryKey: "go", Fields: map[string]any{"title": "The Go Programming Language", "embedding": xvec.VectorFP32{1, 0}}},
		{PrimaryKey: "db", Fields: map[string]any{"title": "Database Internals", "embedding": xvec.VectorFP32{0.5, 0}}},
	})
	if err != nil {
		panic(err)
	}
	results, err := collection.Query(ctx, xvec.VectorQuery{
		Field: "embedding", DenseVector: xvec.VectorFP32{1, 0}, TopK: 1,
		Projection: xvec.Projection{OutputFields: []string{"title"}},
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: %s (%.1f)\n", results[0].PrimaryKey, results[0].Fields["title"], results[0].Score)
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// go: The Go Programming Language (1.0)
}

func ExampleCollection_Query_ann() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-ann-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "items")
	index := xvec.NewHNSWIndexParams(xvec.MetricTypeL2)
	index.M = 8
	index.EFConstruction = 32
	index.Quantize = xvec.QuantizeTypeInt8
	index.Quantizer.EnableRotate = true
	schema := xvec.NewCollectionSchema("items", xvec.FieldSchema{
		Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 4, Index: index,
	})
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{
		{PrimaryKey: "nearest", Fields: map[string]any{"embedding": xvec.VectorFP32{1, 2, 3, 4}}},
		{PrimaryKey: "farther", Fields: map[string]any{"embedding": xvec.VectorFP32{4, 3, 2, 1}}},
	})
	if err != nil {
		panic(err)
	}
	params := xvec.NewHNSWQueryParams()
	params.EF = 32
	params.UseRefiner = true
	results, err := collection.Query(ctx, xvec.VectorQuery{
		Field: "embedding", DenseVector: xvec.VectorFP32{1, 2, 3, 4}, TopK: 1, Params: params,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s %.1f\n", results[0].PrimaryKey, results[0].Score)
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// nearest 0.0
}

func ExampleCollection_MultiQuery() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-multi-query-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "title", DataType: xvec.DataTypeString, Index: xvec.NewFTSIndexParams()},
		xvec.FieldSchema{
			Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 2,
			Index: xvec.NewFlatIndexParams(xvec.MetricTypeIP),
		},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{
		{PrimaryKey: "go", Fields: map[string]any{"title": "Go vector search", "embedding": xvec.VectorFP32{0.8, 0}}},
		{PrimaryKey: "ann", Fields: map[string]any{"title": "Approximate neighbors", "embedding": xvec.VectorFP32{1, 0}}},
	})
	if err != nil {
		panic(err)
	}
	results, err := collection.MultiQuery(ctx, xvec.MultiQuery{
		Queries: []xvec.SubQuery{
			{Field: "embedding", DenseVector: xvec.VectorFP32{1, 0}, NumCandidates: 2},
			{Field: "title", FTS: &xvec.FTSClause{Match: "go search"}, NumCandidates: 2},
		},
		TopK: 1, Projection: xvec.Projection{OutputFields: []string{"title"}},
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: %s\n", results[0].PrimaryKey, results[0].Fields["title"])
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// go: Go vector search
}

func ExampleWeightedReranker_Rerank() {
	reranker := xvec.NewWeightedReranker(0.5, 0.5)
	results, err := reranker.Rerank(context.Background(), []xvec.RerankBatch{
		{
			Field: xvec.FieldSchema{
				Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 2,
				Index: xvec.NewFlatIndexParams(xvec.MetricTypeL2),
			},
			Documents: []xvec.Document{
				{PrimaryKey: "a", DocID: 1, Score: 0},
				{PrimaryKey: "b", DocID: 2, Score: 1},
			},
		},
		{
			Field: xvec.FieldSchema{Name: "body", DataType: xvec.DataTypeString, Index: xvec.NewFTSIndexParams()},
			Documents: []xvec.Document{
				{PrimaryKey: "a", DocID: 1, Score: 1},
			},
		},
	}, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s %.3f\n", results[0].PrimaryKey, results[0].Score)

	// Output:
	// a 0.750
}

func ExampleCallbackReranker_Rerank() {
	reranker := xvec.NewCallbackReranker(func(_ context.Context, batches []xvec.RerankBatch, topK int) ([]xvec.Document, error) {
		result := []xvec.Document{batches[1].Documents[0], batches[0].Documents[0]}
		if len(result) > topK {
			result = result[:topK]
		}
		return result, nil
	})
	results, err := reranker.Rerank(context.Background(), []xvec.RerankBatch{
		{Documents: []xvec.Document{{PrimaryKey: "vector", Score: 0.8}}},
		{Documents: []xvec.Document{{PrimaryKey: "keyword", Score: 2.1}}},
	}, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println(results[0].PrimaryKey)

	// Output:
	// keyword
}

func ExampleCollection_DeleteByFilter() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-delete-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "rating", DataType: xvec.DataTypeInt32, Index: xvec.NewInvertIndexParams()},
		xvec.FieldSchema{
			Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 2,
			Index: xvec.NewFlatIndexParams(xvec.MetricTypeIP),
		},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{
		{PrimaryKey: "keep", Fields: map[string]any{"rating": int32(3), "embedding": xvec.VectorFP32{1, 0}}},
		{PrimaryKey: "remove", Fields: map[string]any{"rating": int32(5), "embedding": xvec.VectorFP32{0.5, 0}}},
	})
	if err != nil {
		panic(err)
	}
	if err := collection.DeleteByFilter(ctx, "rating >= 4"); err != nil {
		panic(err)
	}
	fmt.Println(collection.Stats().DocumentCount)
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// 1
}

func ExampleCollection_CreateIndex() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-index-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "rating", DataType: xvec.DataTypeInt32},
		xvec.FieldSchema{
			Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 2,
			Index: xvec.NewFlatIndexParams(xvec.MetricTypeIP),
		},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{{
		PrimaryKey: "book", Fields: map[string]any{
			"rating": int32(5), "embedding": xvec.VectorFP32{1, 0},
		},
	}})
	if err != nil {
		panic(err)
	}
	if err := collection.CreateIndex(ctx, "rating", xvec.NewInvertIndexParams(), xvec.CreateIndexOptions{Concurrency: 2}); err != nil {
		panic(err)
	}
	field, _ := collection.Schema().Field("rating")
	fmt.Println(field.IndexType())
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// INVERT
}

func ExampleCollection_DropIndex() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-drop-index-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "rating", DataType: xvec.DataTypeInt32, Index: xvec.NewInvertIndexParams()},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	if err := collection.DropIndex(ctx, "rating"); err != nil {
		panic(err)
	}
	field, _ := collection.Schema().Field("rating")
	fmt.Println(field.IndexType())
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// UNDEFINED
}

func ExampleCollection_AddColumn() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-add-column-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "rating", DataType: xvec.DataTypeInt32},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{{
		PrimaryKey: "book", Fields: map[string]any{"rating": int32(4)},
	}})
	if err != nil {
		panic(err)
	}
	field := xvec.FieldSchema{Name: "adjusted", DataType: xvec.DataTypeInt64}
	if err := collection.AddColumn(ctx, field, "rating * 2 + 1", xvec.AddColumnOptions{Concurrency: 2}); err != nil {
		panic(err)
	}
	documents, err := collection.Fetch(ctx, []string{"book"}, xvec.Projection{})
	if err != nil {
		panic(err)
	}
	fmt.Println(documents[0].Fields["adjusted"])
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// 9
}

func ExampleCollection_AlterColumn() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-alter-column-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "rating", DataType: xvec.DataTypeInt32},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{{
		PrimaryKey: "book", Fields: map[string]any{"rating": int32(4)},
	}})
	if err != nil {
		panic(err)
	}
	replacement := xvec.FieldSchema{Name: "adjusted", DataType: xvec.DataTypeInt64}
	if err := collection.AlterColumn(ctx, "rating", "", &replacement, xvec.AlterColumnOptions{Concurrency: 2}); err != nil {
		panic(err)
	}
	documents, err := collection.Fetch(ctx, []string{"book"}, xvec.Projection{})
	if err != nil {
		panic(err)
	}
	fmt.Println(documents[0].Fields["adjusted"])
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// 4
}

func ExampleCollection_DropColumn() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-drop-column-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "title", DataType: xvec.DataTypeString},
		xvec.FieldSchema{Name: "legacy_score", DataType: xvec.DataTypeInt32},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{{
		PrimaryKey: "book", Fields: map[string]any{"title": "Go", "legacy_score": int32(4)},
	}})
	if err != nil {
		panic(err)
	}
	if err := collection.DropColumn(ctx, "legacy_score"); err != nil {
		panic(err)
	}
	documents, err := collection.Fetch(ctx, []string{"book"}, xvec.Projection{})
	if err != nil {
		panic(err)
	}
	_, found := documents[0].Fields["legacy_score"]
	fmt.Println(found)
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// false
}

func ExampleCollection_Optimize() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-optimize-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := xvec.NewCollectionSchema("books",
		xvec.FieldSchema{Name: "rating", DataType: xvec.DataTypeInt32, Index: xvec.NewInvertIndexParams()},
	)
	collection, err := xvec.CreateAndOpen(ctx, path, schema, xvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []xvec.Document{
		{PrimaryKey: "keep", Fields: map[string]any{"rating": int32(5)}},
		{PrimaryKey: "remove", Fields: map[string]any{"rating": int32(1)}},
	})
	if err != nil {
		panic(err)
	}
	if err := collection.Flush(ctx); err != nil {
		panic(err)
	}
	if _, err := collection.Delete(ctx, []string{"remove"}); err != nil {
		panic(err)
	}
	if err := collection.Optimize(ctx, xvec.OptimizeOptions{Concurrency: 2}); err != nil {
		panic(err)
	}
	documents, err := collection.Fetch(ctx, []string{"keep", "remove"}, xvec.Projection{})
	if err != nil {
		panic(err)
	}
	fmt.Println(collection.Stats().DocumentCount)
	fmt.Println(documents[0] != nil, documents[1] == nil)
	if err := collection.Destroy(ctx); err != nil {
		panic(err)
	}
	_ = os.Remove(directory)

	// Output:
	// 1
	// true true
}
