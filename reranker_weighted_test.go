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

package xvec

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWeightedRerankerPinnedMetricNormalization(t *testing.T) {
	tests := []struct {
		name  string
		field FieldSchema
		score float32
		want  float64
	}{
		{name: "L2 zero", field: weightedVectorField("l2", MetricTypeL2), score: 0, want: 1},
		{name: "L2 one", field: weightedVectorField("l2", MetricTypeL2), score: 1, want: 0.5},
		{name: "IP zero", field: weightedVectorField("ip", MetricTypeIP), score: 0, want: 0.5},
		{name: "IP one", field: weightedVectorField("ip", MetricTypeIP), score: 1, want: 0.75},
		{name: "cosine zero", field: weightedVectorField("cosine", MetricTypeCosine), score: 0, want: 1},
		{name: "cosine one", field: weightedVectorField("cosine", MetricTypeCosine), score: 1, want: 0.5},
		{name: "cosine two", field: weightedVectorField("cosine", MetricTypeCosine), score: 2, want: 0},
		{name: "FTS one", field: FieldSchema{Name: "body", DataType: DataTypeString, Index: NewFTSIndexParams()}, score: 1, want: 0.5},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			got, err := normalizeWeightedScore(testCase.score, testCase.field)
			require.NoError(t, err)
			require.InDelta(t, testCase.want, got, 1e-12)
		})
	}
	{
		_, err := normalizeWeightedScore(1, weightedVectorField("mips", MetricTypeMIPSL2))
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
}

func TestWeightedRerankerMixedFieldsScoresAndOwnership(t *testing.T) {
	sourceWeights := []float64{0.5, 0.5}
	reranker := NewWeightedReranker(sourceWeights...)
	sourceWeights[0] = 9
	require.Equal(t, []float64{0.5, 0.5}, reranker.Weights)

	batches := []RerankBatch{
		{Field: weightedVectorField("l2", MetricTypeL2), Documents: []Document{
			{PrimaryKey: "a", DocID: 1, Score: 0.5, Fields: map[string]any{"source": "first"}},
			{PrimaryKey: "b", DocID: 2, Score: 0.3},
		}},
		{Field: weightedVectorField("cosine", MetricTypeCosine), Documents: []Document{
			{PrimaryKey: "a", DocID: 1, Score: 0.4, Fields: map[string]any{"source": "second"}},
			{PrimaryKey: "c", DocID: 3, Score: 0.6},
		}},
	}
	results, err := reranker.Rerank(context.Background(), batches, 10)
	require.NoError(t, err)
	{
		got := documentKeys(results)
		require.Equal(t, []string{"a", "b", "c"}, got)
	}

	wantA := float32((1-2*math.Atan(0.5)/math.Pi)*0.5 + (1-0.4/2)*0.5)
	wantB := float32((1 - 2*math.Atan(0.3)/math.Pi) * 0.5)
	wantC := float32((1 - 0.6/2) * 0.5)
	require.Equal(t, wantA, results[0].Score)
	require.Equal(t, wantB, results[1].Score)
	require.Equal(t, wantC, results[2].Score)
	require.Equal(t, map[string]any{"source": "first"}, results[0].Fields)
	require.True(t, batches[0].Documents[0].Score == 0.5)

	top, err := reranker.Rerank(context.Background(), batches, 2)
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(top))
}

func TestWeightedRerankerValidationZeroNegativeAndContext(t *testing.T) {
	field := weightedVectorField("ip", MetricTypeIP)
	batches := []RerankBatch{{Field: field, Documents: []Document{
		{PrimaryKey: "b", DocID: 2, Score: 1},
		{PrimaryKey: "a", DocID: 1, Score: 2},
	}}}
	zero, err := NewWeightedReranker(0).Rerank(context.Background(), batches, 10)
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(zero))
	require.True(t, zero[0].Score == 0)
	require.True(t, zero[1].Score == 0)

	negative, err := NewWeightedReranker(-1).Rerank(context.Background(), batches, 10)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(negative))

	tests := []struct {
		name     string
		reranker WeightedReranker
		batches  []RerankBatch
	}{
		{name: "too few weights", reranker: NewWeightedReranker(), batches: batches},
		{name: "too many weights", reranker: NewWeightedReranker(1, 2), batches: batches},
		{name: "NaN weight", reranker: NewWeightedReranker(math.NaN()), batches: batches},
		{name: "infinite weight", reranker: NewWeightedReranker(math.Inf(1)), batches: batches},
		{name: "NaN score", reranker: NewWeightedReranker(1), batches: []RerankBatch{{Field: field, Documents: []Document{{PrimaryKey: "a", Score: float32(math.NaN())}}}}},
		{name: "infinite score", reranker: NewWeightedReranker(1), batches: []RerankBatch{{Field: field, Documents: []Document{{PrimaryKey: "a", Score: float32(math.Inf(1))}}}}},
		{name: "scalar field", reranker: NewWeightedReranker(1), batches: []RerankBatch{{Field: NewField("value", DataTypeInt32), Documents: []Document{{PrimaryKey: "a", Score: 1}}}}},
		{name: "MIPS-L2 field", reranker: NewWeightedReranker(1), batches: []RerankBatch{{Field: weightedVectorField("mips", MetricTypeMIPSL2), Documents: []Document{{PrimaryKey: "a", Score: 1}}}}},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			{
				_, err := testCase.reranker.Rerank(context.Background(), testCase.batches, 10)
				require.ErrorIs(t, err, ErrInvalidArgument)
			}
		})
	}
	{
		_, err := NewWeightedReranker(1).Rerank(nil, batches, 1)
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := NewWeightedReranker(1).Rerank(canceled, batches, 1)
		require.ErrorIs(t, err, context.Canceled)
	}

	empty, err := NewWeightedReranker().Rerank(context.Background(), nil, 10)
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
}

func TestCollectionMultiQueryWeightedReranker(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "weighted"), testMultiQuerySchema(), NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	{
		_, err := collection.Insert(ctx, testMultiQueryDocuments())
		require.NoError(t, err)
	}

	query := MultiQuery{
		Queries: []SubQuery{
			{Field: "embedding", DenseVector: VectorFP32{1, 0}, NumCandidates: 4},
			{Field: "title", FTS: &FTSClause{Match: "go"}, NumCandidates: 4},
		},
		TopK: 3, Filter: "category = 'keep'",
		Projection: Projection{OutputFields: []string{"title"}},
	}
	var captured []RerankBatch
	query.Reranker = testRerankerFunc(func(_ context.Context, batches []RerankBatch, topK int) ([]Document, error) {
		captured = batches
		return firstDistinctDocuments(batches, topK), nil
	})
	{
		_, err := collection.MultiQuery(ctx, query)
		require.NoError(t, err)
	}

	weighted := NewWeightedReranker(0.25, 0.75)
	want, err := weighted.Rerank(ctx, captured, query.TopK)
	require.NoError(t, err)

	query.Reranker = weighted
	got, err := collection.MultiQuery(ctx, query)
	require.NoError(t, err)
	require.Equal(t, want, got)

	for _, document := range got {
		require.Len(t, document.Fields, 1)
	}
}

func TestWeightedRerankerConcurrentDeterminism(t *testing.T) {
	batches := make([]RerankBatch, 3)
	for batchIndex := range batches {
		batches[batchIndex].Field = weightedVectorField(rrfTestKey(batchIndex), MetricTypeIP)
		batches[batchIndex].Documents = make([]Document, 128)
		for rank := range batches[batchIndex].Documents {
			key := (rank*17 + batchIndex*29) % 181
			batches[batchIndex].Documents[rank] = Document{
				PrimaryKey: rrfTestKey(key), DocID: uint64(key + 1), Score: float32(rank) / 17,
			}
		}
	}
	reranker := NewWeightedReranker(0.2, 0.3, 0.5)
	want, err := reranker.Rerank(context.Background(), batches, 40)
	require.NoError(t, err)

	var wait sync.WaitGroup
	errorsFound := make(chan error, 8)
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 30; iteration++ {
				got, err := reranker.Rerank(context.Background(), batches, 40)
				if err != nil {
					errorsFound <- err
					return
				}
				if !assert.Equal(t, want, got) {
					errorsFound <- errors.New("non-deterministic weighted result")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsFound)
	for err := range errorsFound {
		require.NoError(t, err)
	}
}

func TestWeightedRerankerCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/weighted_reranker_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		BaselineCommit string            `json:"baseline_commit"`
		HeaderHash     string            `json:"header_sha256"`
		SourceHash     string            `json:"source_sha256"`
		TestsHash      string            `json:"tests_sha256"`
		Formulas       map[string]string `json:"formulas"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}

	wantFormulas := map[string]string{
		"FTS": "2*atan(score)/pi", "L2": "1-2*atan(score)/pi",
		"IP": "0.5+atan(score)/pi", "COSINE": "1-score/2",
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.HeaderHash == "bc1949536968bc27f0cb11026d0ab8633dbb46641365455c20b433367837c7d6")
	require.True(t, fixture.SourceHash == "3c93edc12303898af52911589c46c720072f9470446858fc36a61206d538aa1e")
	require.True(t, fixture.TestsHash == "05a03cacf74e7615661cec3153b2d2307f6a991510018386f6650f2625cb9a7d")
	require.Equal(t, wantFormulas, fixture.Formulas)
}

func FuzzWeightedReranker(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6}, uint8(3))
	f.Add([]byte{}, uint8(0))
	f.Fuzz(func(t *testing.T, encoded []byte, topKByte uint8) {
		if len(encoded) > 512 {
			encoded = encoded[:512]
		}
		batches := make([]RerankBatch, 3)
		for index := range batches {
			batches[index].Field = weightedVectorField(rrfTestKey(index), MetricTypeIP)
		}
		for index, value := range encoded {
			batch := index % len(batches)
			key := int(value % 32)
			batches[batch].Documents = append(batches[batch].Documents, Document{
				PrimaryKey: rrfTestKey(key), DocID: uint64(key + 1), Score: float32(int(value)-128) / 17,
			})
		}
		topK := int(topKByte % 40)
		reranker := NewWeightedReranker(-0.25, 0.5, 1.25)
		first, err := reranker.Rerank(context.Background(), batches, topK)
		require.NoError(t, err)

		second, err := reranker.Rerank(context.Background(), batches, topK)
		require.NoError(t, err)
		require.Equal(t, first, second)
		require.True(t, len(first) <= topK)

		seen := make(map[string]struct{}, len(first))
		for index, document := range first {
			{
				_, found := seen[document.PrimaryKey]
				require.False(t, found)
			}

			seen[document.PrimaryKey] = struct{}{}
			require.False(t, math.IsNaN(float64(document.Score)))
			require.False(t, math.IsInf(float64(document.Score), 0))
			require.False(t, index > 0 && first[index-1].Score < document.Score)
		}
	})
}

func BenchmarkWeightedReranker(b *testing.B) {
	batches := make([]RerankBatch, 8)
	weights := make([]float64, len(batches))
	for batchIndex := range batches {
		weights[batchIndex] = 1 / float64(len(batches))
		batches[batchIndex].Field = weightedVectorField(rrfTestKey(batchIndex), MetricTypeIP)
		batches[batchIndex].Documents = make([]Document, 1_000)
		for rank := range batches[batchIndex].Documents {
			key := (rank*37 + batchIndex*101) % 4_000
			batches[batchIndex].Documents[rank] = Document{
				PrimaryKey: rrfTestKey(key), DocID: uint64(key + 1), Score: float32(rank) / 101,
			}
		}
	}
	reranker := NewWeightedReranker(weights...)
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		{
			_, err := reranker.Rerank(context.Background(), batches, 100)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func weightedVectorField(name string, metric MetricType) FieldSchema {
	return FieldSchema{
		Name: name, DataType: DataTypeVectorFP32, Dimension: 2,
		Index: NewFlatIndexParams(metric),
	}
}
