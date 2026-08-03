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

package zvec_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gorse-io/zvec"
)

func ExampleCollection_Query() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")

	schema := zvec.NewCollectionSchema("books",
		zvec.FieldSchema{Name: "title", DataType: zvec.DataTypeString},
		zvec.FieldSchema{
			Name: "embedding", DataType: zvec.DataTypeVectorFP32, Dimension: 2,
			Index: zvec.NewFlatIndexParams(zvec.MetricTypeIP),
		},
	)
	collection, err := zvec.CreateAndOpen(ctx, path, schema, zvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []zvec.Document{
		{PrimaryKey: "go", Fields: map[string]any{"title": "The Go Programming Language", "embedding": zvec.VectorFP32{1, 0}}},
		{PrimaryKey: "db", Fields: map[string]any{"title": "Database Internals", "embedding": zvec.VectorFP32{0.5, 0}}},
	})
	if err != nil {
		panic(err)
	}
	results, err := collection.Query(ctx, zvec.VectorQuery{
		Field: "embedding", DenseVector: zvec.VectorFP32{1, 0}, TopK: 1,
		Projection: zvec.Projection{OutputFields: []string{"title"}},
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

func ExampleCollection_DeleteByFilter() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "zvec-delete-example-")
	if err != nil {
		panic(err)
	}
	path := filepath.Join(directory, "books")
	schema := zvec.NewCollectionSchema("books",
		zvec.FieldSchema{Name: "rating", DataType: zvec.DataTypeInt32, Index: zvec.NewInvertIndexParams()},
		zvec.FieldSchema{
			Name: "embedding", DataType: zvec.DataTypeVectorFP32, Dimension: 2,
			Index: zvec.NewFlatIndexParams(zvec.MetricTypeIP),
		},
	)
	collection, err := zvec.CreateAndOpen(ctx, path, schema, zvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []zvec.Document{
		{PrimaryKey: "keep", Fields: map[string]any{"rating": int32(3), "embedding": zvec.VectorFP32{1, 0}}},
		{PrimaryKey: "remove", Fields: map[string]any{"rating": int32(5), "embedding": zvec.VectorFP32{0.5, 0}}},
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
	schema := zvec.NewCollectionSchema("books",
		zvec.FieldSchema{Name: "rating", DataType: zvec.DataTypeInt32},
		zvec.FieldSchema{
			Name: "embedding", DataType: zvec.DataTypeVectorFP32, Dimension: 2,
			Index: zvec.NewFlatIndexParams(zvec.MetricTypeIP),
		},
	)
	collection, err := zvec.CreateAndOpen(ctx, path, schema, zvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []zvec.Document{{
		PrimaryKey: "book", Fields: map[string]any{
			"rating": int32(5), "embedding": zvec.VectorFP32{1, 0},
		},
	}})
	if err != nil {
		panic(err)
	}
	if err := collection.CreateIndex(ctx, "rating", zvec.NewInvertIndexParams(), zvec.CreateIndexOptions{Concurrency: 2}); err != nil {
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
	schema := zvec.NewCollectionSchema("books",
		zvec.FieldSchema{Name: "rating", DataType: zvec.DataTypeInt32, Index: zvec.NewInvertIndexParams()},
	)
	collection, err := zvec.CreateAndOpen(ctx, path, schema, zvec.NewCollectionOptions())
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
	schema := zvec.NewCollectionSchema("books",
		zvec.FieldSchema{Name: "rating", DataType: zvec.DataTypeInt32},
	)
	collection, err := zvec.CreateAndOpen(ctx, path, schema, zvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []zvec.Document{{
		PrimaryKey: "book", Fields: map[string]any{"rating": int32(4)},
	}})
	if err != nil {
		panic(err)
	}
	field := zvec.FieldSchema{Name: "adjusted", DataType: zvec.DataTypeInt64}
	if err := collection.AddColumn(ctx, field, "rating * 2 + 1", zvec.AddColumnOptions{Concurrency: 2}); err != nil {
		panic(err)
	}
	documents, err := collection.Fetch(ctx, []string{"book"}, zvec.Projection{})
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
	schema := zvec.NewCollectionSchema("books",
		zvec.FieldSchema{Name: "rating", DataType: zvec.DataTypeInt32},
	)
	collection, err := zvec.CreateAndOpen(ctx, path, schema, zvec.NewCollectionOptions())
	if err != nil {
		panic(err)
	}
	_, err = collection.Insert(ctx, []zvec.Document{{
		PrimaryKey: "book", Fields: map[string]any{"rating": int32(4)},
	}})
	if err != nil {
		panic(err)
	}
	replacement := zvec.FieldSchema{Name: "adjusted", DataType: zvec.DataTypeInt64}
	if err := collection.AlterColumn(ctx, "rating", "", &replacement, zvec.AlterColumnOptions{Concurrency: 2}); err != nil {
		panic(err)
	}
	documents, err := collection.Fetch(ctx, []string{"book"}, zvec.Projection{})
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
