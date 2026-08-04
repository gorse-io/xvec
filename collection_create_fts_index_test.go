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

func TestCreateFTSIndexBackfillQueryAndReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "fts-backfill")
	schema := NewCollectionSchema("fts_backfill",
		FieldSchema{Name: "title", DataType: DataTypeString},
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
		{PrimaryKey: "go", Fields: map[string]any{"title": "Go vector search", "embedding": VectorFP32{1, 0}}},
		{PrimaryKey: "db", Fields: map[string]any{"title": "Database internals", "embedding": VectorFP32{0.5, 0}}},
	}); err != nil {
		t.Fatal(err)
	}
	params := FTSIndexParams{Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"}, ExtraParams: `{"stemmer_lang":"english"}`}
	if err := collection.CreateIndex(ctx, "title", params, CreateIndexOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	field, found := collection.Schema().Field("title")
	if !found || !equalIndexParams(field.Index, params) || collection.Stats().IndexCompleteness["title"] != 1 {
		t.Fatalf("FTS schema/stats = %#v / %#v", field, collection.Stats())
	}
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "searching"}, NumCandidates: 2},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 2},
		},
		TopK: 1, Projection: Projection{OutputFields: []string{"title"}},
	}
	want, err := collection.MultiQuery(ctx, query)
	if err != nil || len(want) != 1 || want[0].PrimaryKey != "go" {
		t.Fatalf("backfilled FTS query = %#v, %v", want, err)
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
	got, err := reopened.MultiQuery(ctx, query)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened FTS query = %#v, %v; want %#v", got, err, want)
	}
}

func TestCreateFTSIndexBackfillFailureRollsBack(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "fts-rollback")
	schema := NewCollectionSchema("fts_rollback", FieldSchema{Name: "title", DataType: DataTypeString})
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	defer collection.Close()
	if _, err := collection.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"title": "中文"}}}); err != nil {
		t.Fatal(err)
	}
	beforeSchema := collection.Schema()
	beforeGeneration := collection.store.Manifest().Generation
	params := FTSIndexParams{Tokenizer: "jieba", ExtraParams: `{"jieba_dict_dir":"missing-jieba-resources"}`}
	if err := collection.CreateIndex(ctx, "title", params, CreateIndexOptions{}); err == nil {
		t.Fatal("missing Jieba resources unexpectedly succeeded")
	}
	if !reflect.DeepEqual(collection.Schema(), beforeSchema) || collection.store.Manifest().Generation != beforeGeneration {
		t.Fatal("failed FTS backfill changed schema or manifest")
	}
}
