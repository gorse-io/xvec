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

func TestOptimizeFTSCompactsDeletesAndReopens(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize-fts")
	fts := FTSIndexParams{
		Tokenizer: "whitespace", Filters: []string{"lowercase", "stemmer"},
		ExtraParams: `{"stemmer_lang":"english"}`,
	}
	schema := NewCollectionSchema("optimize_fts",
		FieldSchema{Name: "title", DataType: DataTypeString, Index: fts},
		FieldSchema{
			Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2,
			Index: NewFlatIndexParams(MetricTypeIP),
		},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, []Document{
		{PrimaryKey: "a", Fields: map[string]any{"title": "Go searching", "embedding": VectorFP32{1, 0}}},
		{PrimaryKey: "b", Fields: map[string]any{"title": "Database search", "embedding": VectorFP32{0.7, 0}}},
		{PrimaryKey: "remove", Fields: map[string]any{"title": "Go removed", "embedding": VectorFP32{0.9, 0}}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"title": "Go optimized searching"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Delete(ctx, []string{"remove"}); err != nil {
		t.Fatal(err)
	}
	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "title", FTS: &FTSClause{Match: "optimized search"}, NumCandidates: 3},
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 3},
		},
		TopK: 2, Projection: Projection{OutputFields: []string{"title"}},
	}
	want, err := collection.MultiQuery(ctx, query)
	if err != nil || len(want) != 2 || want[0].PrimaryKey != "a" {
		t.Fatalf("query before Optimize = %#v, %v", want, err)
	}
	before := collection.Stats()
	if before.DeletedDocuments < 2 {
		t.Fatalf("deletions before Optimize = %#v", before)
	}
	if err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2}); err != nil {
		t.Fatal(err)
	}
	after := collection.Stats()
	if after.DocumentCount != 2 || after.DeletedDocuments != 0 || after.MutableDocuments != 0 || after.ImmutableSegments != 2 {
		t.Fatalf("stats after FTS Optimize = %#v", after)
	}
	got, err := collection.MultiQuery(ctx, query)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("query after Optimize = %#v, %v; want %#v", got, err, want)
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
	got, err = reopened.MultiQuery(ctx, query)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened optimized FTS query = %#v, %v; want %#v", got, err, want)
	}
}
