// Copyright 2026-present the xvec project
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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRRFRerankerPinnedScoresAndDeterministicTies(t *testing.T) {
	batches := []RerankBatch{
		{Documents: []Document{
			{PrimaryKey: "a", DocID: 1, Score: float32(math.NaN()), Fields: map[string]any{"source": "first"}},
			{PrimaryKey: "b", DocID: 2, Score: -100},
			{PrimaryKey: "c", DocID: 3, Score: 100},
		}},
		{Documents: []Document{
			{PrimaryKey: "b", DocID: 2, Score: 999},
			{PrimaryKey: "a", DocID: 1, Score: -999, Fields: map[string]any{"source": "second"}},
			{PrimaryKey: "d", DocID: 4, Score: 0},
		}},
	}
	reranker := NewRRFReranker()
	require.True(t, DefaultRRFRankConstant == 60)
	require.Equal(t, DefaultRRFRankConstant, reranker.RankConstant)

	results, err := reranker.Rerank(context.Background(), batches, 10)
	require.NoError(t, err)
	{
		got := documentKeys(results)
		require.Equal(t, []string{"a", "b", "c", "d"}, got)
	}

	wantTop := float32(1.0/61.0 + 1.0/62.0)
	wantTail := float32(1.0 / 63.0)
	require.Equal(t, wantTop, results[0].Score)
	require.Equal(t, wantTop, results[1].Score)
	require.Equal(t, wantTail, results[2].Score)
	require.Equal(t, wantTail, results[3].Score)
	require.Equal(t, map[string]any{"source": "first"}, results[0].Fields)
	require.True(t, math.IsNaN(float64(batches[0].Documents[0].Score)))

	top, err := reranker.Rerank(context.Background(), batches, 2)
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, documentKeys(top))
}

func TestRRFRerankerZeroConstantDuplicateAndArgumentBoundaries(t *testing.T) {
	r := RRFReranker{RankConstant: 0}
	batches := []RerankBatch{{Documents: []Document{
		{PrimaryKey: "a", DocID: 1},
		{PrimaryKey: "a", DocID: 1},
		{PrimaryKey: "b", DocID: 2},
	}}}
	results, err := r.Rerank(context.Background(), batches, 10)
	require.NoError(t, err)
	{
		got := documentKeys(results)
		require.Equal(t, []string{"a", "b"}, got)
	}
	require.Equal(t, float32(1+1.0/2.0), results[0].Score)
	require.Equal(t, float32(1.0/3.0), results[1].Score)
	{
		err := (RRFReranker{RankConstant: -1}).Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := (RRFReranker{RankConstant: -1}).Rerank(context.Background(), batches, 1)
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		_, err := r.Rerank(nil, batches, 1)
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := r.Rerank(canceled, batches, 1)
		require.ErrorIs(t, err, context.Canceled)
	}

	for _, topK := range []int{0, -1} {
		empty, err := r.Rerank(context.Background(), batches, topK)
		require.NoError(t, err)
		require.NotNil(t, empty)
		require.Len(t, empty, 0)
	}
	empty, err := r.Rerank(context.Background(), nil, 10)
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
}

func TestRRFRerankerConcurrentDeterminism(t *testing.T) {
	batches := make([]RerankBatch, 4)
	for batchIndex := range batches {
		batches[batchIndex].Documents = make([]Document, 128)
		for rank := range batches[batchIndex].Documents {
			key := (rank*17 + batchIndex*29) % 181
			batches[batchIndex].Documents[rank] = Document{PrimaryKey: rrfTestKey(key), DocID: uint64(key + 1)}
		}
	}
	reranker := NewRRFReranker()
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
					errorsFound <- errors.New("non-deterministic RRF result")
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

func TestRRFCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/rrf_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		BaselineCommit string `json:"baseline_commit"`
		HeaderHash     string `json:"header_sha256"`
		SourceHash     string `json:"source_sha256"`
		TestsHash      string `json:"tests_sha256"`
		RankConstant   int    `json:"default_rank_constant"`
		Formula        string `json:"formula"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.HeaderHash == "bc1949536968bc27f0cb11026d0ab8633dbb46641365455c20b433367837c7d6")
	require.True(t, fixture.SourceHash == "3c93edc12303898af52911589c46c720072f9470446858fc36a61206d538aa1e")
	require.True(t, fixture.TestsHash == "05a03cacf74e7615661cec3153b2d2307f6a991510018386f6650f2625cb9a7d")
	require.Equal(t, DefaultRRFRankConstant, fixture.RankConstant)
	require.True(t, fixture.Formula == "1/(rank_constant+rank+1)")
}

func FuzzRRFReranker(f *testing.F) {
	f.Add([]byte{0, 1, 2, 1, 0, 3}, uint8(3), uint8(60))
	f.Add([]byte{}, uint8(0), uint8(0))
	f.Fuzz(func(t *testing.T, encoded []byte, topKByte, constantByte uint8) {
		if len(encoded) > 512 {
			encoded = encoded[:512]
		}
		batches := make([]RerankBatch, 3)
		for index, value := range encoded {
			batch := index % len(batches)
			key := int(value % 32)
			batches[batch].Documents = append(batches[batch].Documents, Document{
				PrimaryKey: rrfTestKey(key), DocID: uint64(key + 1), Score: float32(value),
			})
		}
		topK := int(topKByte % 40)
		reranker := RRFReranker{RankConstant: int(constantByte)}
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
			require.True(t, document.Score > 0)
			require.False(t, math.IsNaN(float64(document.Score)))
			require.False(t, math.IsInf(float64(document.Score), 0))
			require.False(t, index > 0 && first[index-1].Score < document.Score)
		}
	})
}

func BenchmarkRRFReranker(b *testing.B) {
	batches := make([]RerankBatch, 8)
	for batchIndex := range batches {
		batches[batchIndex].Documents = make([]Document, 1_000)
		for rank := range batches[batchIndex].Documents {
			key := (rank*37 + batchIndex*101) % 4_000
			batches[batchIndex].Documents[rank] = Document{PrimaryKey: rrfTestKey(key), DocID: uint64(key + 1)}
		}
	}
	reranker := NewRRFReranker()
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

func rrfTestKey(value int) string {
	const digits = "0123456789abcdef"
	return string([]byte{
		'd', digits[(value>>12)&15], digits[(value>>8)&15],
		digits[(value>>4)&15], digits[value&15],
	})
}
