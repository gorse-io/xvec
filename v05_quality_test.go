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
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

// TestV05HybridReleaseReopenMatrix is the release-level behavior gate for all
// three retrieval sources and every public fusion strategy. Results must remain
// value-identical after the durable collection is reopened read-only.
func TestV05HybridReleaseReopenMatrix(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "v05-hybrid")
	collection, err := CreateAndOpen(ctx, path, testMultiQuerySchema(), NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collection.Insert(ctx, testMultiQueryDocuments()); err != nil {
		t.Fatal(err)
	}
	if err := collection.Flush(ctx); err != nil {
		t.Fatal(err)
	}

	queries := []struct {
		name     string
		reranker Reranker
	}{
		{name: "default RRF"},
		{name: "explicit RRF", reranker: NewRRFReranker()},
		{name: "weighted", reranker: NewWeightedReranker(0.4, 0.2, 0.4)},
		{name: "callback", reranker: NewCallbackReranker(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
			return firstDistinctDocuments(batches, topK), nil
		})},
	}
	makeQuery := func(reranker Reranker) MultiQuery {
		return MultiQuery{
			Queries: []SubQuery{
				{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
				{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, NumCandidates: 4},
				{Field: "title", FTS: &FTSClause{Query: `go AND (database OR search)`}, NumCandidates: 4},
			},
			TopK: 2, Filter: "category = 'keep'",
			Projection: Projection{OutputFields: []string{"title"}},
			Reranker:   reranker,
		}
	}
	run := func(handle *Collection) map[string][]Document {
		t.Helper()
		results := make(map[string][]Document, len(queries))
		for _, testCase := range queries {
			documents, queryErr := handle.MultiQuery(ctx, makeQuery(testCase.reranker))
			if queryErr != nil {
				t.Fatalf("%s: %v", testCase.name, queryErr)
			}
			if len(documents) != 2 {
				t.Fatalf("%s returned %d documents", testCase.name, len(documents))
			}
			for _, document := range documents {
				if document.PrimaryKey == "c" || len(document.Fields) != 1 {
					t.Fatalf("%s returned unfiltered/unprojected document %#v", testCase.name, document)
				}
				if score := float64(document.Score); math.IsNaN(score) || math.IsInf(score, 0) {
					t.Fatalf("%s returned non-finite score %v", testCase.name, document.Score)
				}
			}
			results[testCase.name] = documents
		}
		if !reflect.DeepEqual(results["default RRF"], results["explicit RRF"]) {
			t.Fatalf("nil and explicit RRF differ: %#v != %#v", results["default RRF"], results["explicit RRF"])
		}
		return results
	}

	writableResults := run(collection)
	stats := collection.Stats()
	if stats.DocumentCount != 4 || stats.ImmutableSegments != 1 || stats.MutableDocuments != 0 ||
		stats.StorageMemoryBytes == 0 || stats.IndexCompleteness["title"] != 1 {
		t.Fatalf("release collection stats = %#v", stats)
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
	if reopenedResults := run(reopened); !reflect.DeepEqual(reopenedResults, writableResults) {
		t.Fatalf("hybrid results changed across reopen: %#v != %#v", reopenedResults, writableResults)
	}
}
