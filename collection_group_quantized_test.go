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
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gorse-io/zvec/internal/core"
)

func TestCollectionDenseQuantizedLinearGroupByAndRefinement(t *testing.T) {
	flat := NewFlatIndexParams(MetricTypeL2)
	flat.Quantize = QuantizeTypeInt4
	flat.Quantizer.EnableRotate = true
	hnsw := NewHNSWIndexParams(MetricTypeL2)
	hnsw.M, hnsw.EFConstruction = 8, 32
	hnsw.Quantize = QuantizeTypeInt8
	hnsw.Quantizer.EnableRotate = true
	rabitq := NewHNSWRaBitQIndexParams(MetricTypeL2)
	rabitq.TotalBits, rabitq.NumClusters = 5, 4
	rabitq.M, rabitq.EFConstruction = 6, 24

	tests := []struct {
		name   string
		index  IndexParams
		params func(refine bool) QueryParams
	}{
		{
			name: "Flat INT4", index: flat,
			params: func(refine bool) QueryParams {
				value := NewFlatQueryParams()
				value.UseRefiner, value.ScaleFactor = refine, 100
				return value
			},
		},
		{
			name: "HNSW INT8", index: hnsw,
			params: func(refine bool) QueryParams {
				value := NewHNSWQueryParams()
				value.Linear, value.UseRefiner = true, refine
				return value
			},
		},
		{
			name: "HNSW RaBitQ", index: rabitq,
			params: func(refine bool) QueryParams {
				value := NewHNSWRaBitQQueryParams()
				value.Linear, value.UseRefiner = true, refine
				return value
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "collection")
			schema := NewCollectionSchema("dense_group",
				FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 64, Index: testCase.index},
				FieldSchema{Name: "group", DataType: DataTypeString, Nullable: true},
				FieldSchema{Name: "rating", DataType: DataTypeInt32},
			)
			schema.MaxDocsPerSegment = MinMaxDocsPerSegment
			collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			if err != nil {
				t.Fatal(err)
			}
			documents := annRaBitQDocuments(24)
			for position := range documents {
				if position%7 == 0 {
					documents[position].Fields["group"] = nil
				} else {
					documents[position].Fields["group"] = fmt.Sprintf("g%d", position%4)
				}
			}
			if _, err := collection.Insert(ctx, documents); err != nil {
				t.Fatal(err)
			}
			queryVector := append(VectorFP32(nil), documents[11].Fields["embedding"].(VectorFP32)...)
			queryVector[3] += .137
			filter := "rating >= 1"
			groupQuery := GroupByVectorQuery{
				Field: "embedding", DenseVector: queryVector, Filter: filter,
				GroupByField: "group", GroupCount: 4, TopKPerGroup: 2,
			}

			groupQuery.Params = testCase.params(false)
			firstStageGroups, err := collection.GroupByQuery(ctx, groupQuery)
			if err != nil {
				t.Fatal(err)
			}
			firstStage, err := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: queryVector, TopK: len(documents),
				Filter: filter, Params: testCase.params(false),
			})
			if err != nil {
				t.Fatal(err)
			}
			assertCollectionGroupsMatchResults(t, firstStageGroups, firstStage, "group", core.MetricL2, 4, 2)

			groupQuery.Params = testCase.params(true)
			refinedGroups, err := collection.GroupByQuery(ctx, groupQuery)
			if err != nil {
				t.Fatal(err)
			}
			refined, err := collection.Query(ctx, VectorQuery{
				Field: "embedding", DenseVector: queryVector, TopK: len(documents),
				Filter: filter, Params: testCase.params(true),
			})
			if err != nil {
				t.Fatal(err)
			}
			assertCollectionGroupsMatchResults(t, refinedGroups, refined, "group", core.MetricL2, 4, 2)
			if !collectionGroupScoresDiffer(firstStageGroups, refinedGroups) {
				t.Fatal("refinement did not change any quantized group score")
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
			reopened, err := collection.GroupByQuery(ctx, groupQuery)
			if err != nil || !reflect.DeepEqual(reopened, refinedGroups) {
				t.Fatalf("reopened refined groups = %#v, %v; before %#v", reopened, err, refinedGroups)
			}
		})
	}
}

func TestCollectionSparseFP16LinearGroupByAndRefinement(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "collection")
	index := NewHNSWIndexParams(MetricTypeIP)
	index.M, index.EFConstruction = 8, 32
	index.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("sparse_group",
		FieldSchema{Name: "sparse", DataType: DataTypeSparseVectorFP32, Index: index},
		FieldSchema{Name: "group", DataType: DataTypeString, Nullable: true},
		FieldSchema{Name: "rating", DataType: DataTypeInt32},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	if err != nil {
		t.Fatal(err)
	}
	documents := make([]Document, 30)
	for position := range documents {
		group := any(fmt.Sprintf("g%d", position%4))
		if position%9 == 0 {
			group = nil
		}
		documents[position] = Document{PrimaryKey: fmt.Sprintf("s%02d", position), Fields: map[string]any{
			"sparse": SparseVectorFP32{
				Indices: []uint32{uint32(position % 7), uint32(20 + position%11), uint32(50 + position%13)},
				Values:  []float32{float32(position%5) + .12345, float32(position%7) + .33331, float32(position%9) + .77771},
			},
			"group": group, "rating": int32(position % 3),
		}}
	}
	if _, err := collection.Insert(ctx, documents); err != nil {
		t.Fatal(err)
	}
	queryVector := SparseVectorFP32{
		Indices: []uint32{2, 24, 57},
		Values:  []float32{2.0007, 1.0003, 3.0009},
	}
	params := NewHNSWQueryParams()
	params.Linear = true
	filter := "rating >= 1"
	groupQuery := GroupByVectorQuery{
		Field: "sparse", SparseVector: queryVector, Filter: filter, Params: params,
		GroupByField: "group", GroupCount: 4, TopKPerGroup: 2,
	}
	firstStageGroups, err := collection.GroupByQuery(ctx, groupQuery)
	if err != nil {
		t.Fatal(err)
	}
	firstStage, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: queryVector, TopK: len(documents), Filter: filter, Params: params,
	})
	if err != nil {
		t.Fatal(err)
	}
	assertCollectionGroupsMatchResults(t, firstStageGroups, firstStage, "group", core.MetricIP, 4, 2)

	params.UseRefiner = true
	groupQuery.Params = params
	refinedGroups, err := collection.GroupByQuery(ctx, groupQuery)
	if err != nil {
		t.Fatal(err)
	}
	refined, err := collection.Query(ctx, VectorQuery{
		Field: "sparse", SparseVector: queryVector, TopK: len(documents), Filter: filter, Params: params,
	})
	if err != nil {
		t.Fatal(err)
	}
	assertCollectionGroupsMatchResults(t, refinedGroups, refined, "group", core.MetricIP, 4, 2)
	if !collectionGroupScoresDiffer(firstStageGroups, refinedGroups) {
		t.Fatal("sparse refinement did not change any FP16 group score")
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
	reopened, err := collection.GroupByQuery(ctx, groupQuery)
	if err != nil || !reflect.DeepEqual(reopened, refinedGroups) {
		t.Fatalf("reopened sparse refined groups = %#v, %v; before %#v", reopened, err, refinedGroups)
	}
}

func assertCollectionGroupsMatchResults(
	t testing.TB,
	got []GroupResult,
	results []Document,
	groupField string,
	metric core.Metric,
	groupCount, topKPerGroup int,
) {
	t.Helper()
	byGroup := make(map[string][]core.Result)
	byID := make(map[uint64]Document, len(results))
	for _, document := range results {
		value := ""
		if raw := document.Fields[groupField]; raw != nil {
			value = raw.(string)
		}
		byGroup[value] = append(byGroup[value], core.Result{Key: document.DocID, Score: document.Score})
		byID[document.DocID] = document
	}
	batch := make([]core.GroupResult, 0, len(byGroup))
	for value, groupResults := range byGroup {
		batch = append(batch, core.GroupResult{Value: value, Results: groupResults})
	}
	want := core.MergeGroupResults(metric, groupCount, topKPerGroup, batch)
	if len(got) != len(want) {
		t.Fatalf("group count = %d, want %d: %#v", len(got), len(want), got)
	}
	for groupIndex := range want {
		if got[groupIndex].Value != want[groupIndex].Value || len(got[groupIndex].Documents) != len(want[groupIndex].Results) {
			t.Fatalf("group %d = %#v, want %#v", groupIndex, got[groupIndex], want[groupIndex])
		}
		for documentIndex, result := range want[groupIndex].Results {
			document := got[groupIndex].Documents[documentIndex]
			expected := byID[result.Key]
			if document.PrimaryKey != expected.PrimaryKey || document.DocID != result.Key || document.Score != result.Score {
				t.Fatalf("group %d document %d = %#v, want key %q ID %d score %v",
					groupIndex, documentIndex, document, expected.PrimaryKey, result.Key, result.Score)
			}
		}
	}
}

func collectionGroupScoresDiffer(left, right []GroupResult) bool {
	leftScores := make(map[string]float32)
	for _, group := range left {
		for _, document := range group.Documents {
			leftScores[document.PrimaryKey] = document.Score
		}
	}
	for _, group := range right {
		for _, document := range group.Documents {
			if score, found := leftScores[document.PrimaryKey]; found && score != document.Score {
				return true
			}
		}
	}
	return false
}
