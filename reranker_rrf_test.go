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
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
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
	if DefaultRRFRankConstant != 60 || reranker.RankConstant != DefaultRRFRankConstant {
		t.Fatalf("default RRF = %#v", reranker)
	}
	results, err := reranker.Rerank(context.Background(), batches, 10)
	if err != nil {
		t.Fatal(err)
	}
	if got := documentKeys(results); !reflect.DeepEqual(got, []string{"a", "b", "c", "d"}) {
		t.Fatalf("RRF keys = %#v", got)
	}
	wantTop := float32(1.0/61.0 + 1.0/62.0)
	wantTail := float32(1.0 / 63.0)
	if results[0].Score != wantTop || results[1].Score != wantTop ||
		results[2].Score != wantTail || results[3].Score != wantTail {
		t.Fatalf("RRF scores = %#v, want top %v tail %v", results, wantTop, wantTail)
	}
	if !reflect.DeepEqual(results[0].Fields, map[string]any{"source": "first"}) {
		t.Fatalf("RRF did not retain the first occurrence: %#v", results[0])
	}
	if !math.IsNaN(float64(batches[0].Documents[0].Score)) {
		t.Fatalf("RRF mutated its input document: %#v", batches[0].Documents[0])
	}

	top, err := reranker.Rerank(context.Background(), batches, 2)
	if err != nil || !reflect.DeepEqual(documentKeys(top), []string{"a", "b"}) {
		t.Fatalf("RRF topK = %#v, %v", top, err)
	}
}

func TestRRFRerankerZeroConstantDuplicateAndArgumentBoundaries(t *testing.T) {
	r := RRFReranker{RankConstant: 0}
	batches := []RerankBatch{{Documents: []Document{
		{PrimaryKey: "a", DocID: 1},
		{PrimaryKey: "a", DocID: 1},
		{PrimaryKey: "b", DocID: 2},
	}}}
	results, err := r.Rerank(context.Background(), batches, 10)
	if err != nil {
		t.Fatal(err)
	}
	if got := documentKeys(results); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("zero-constant duplicate keys = %#v", got)
	}
	if results[0].Score != float32(1+1.0/2.0) || results[1].Score != float32(1.0/3.0) {
		t.Fatalf("zero-constant duplicate scores = %#v", results)
	}
	if err := (RRFReranker{RankConstant: -1}).Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("negative Validate = %v", err)
	}
	if _, err := (RRFReranker{RankConstant: -1}).Rerank(context.Background(), batches, 1); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("negative Rerank = %v", err)
	}
	if _, err := r.Rerank(nil, batches, 1); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.Rerank(canceled, batches, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context = %v", err)
	}
	for _, topK := range []int{0, -1} {
		empty, err := r.Rerank(context.Background(), batches, topK)
		if err != nil || empty == nil || len(empty) != 0 {
			t.Fatalf("topK %d = %#v, %v", topK, empty, err)
		}
	}
	empty, err := r.Rerank(context.Background(), nil, 10)
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("empty batches = %#v, %v", empty, err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
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
				if !reflect.DeepEqual(got, want) {
					errorsFound <- errors.New("non-deterministic RRF result")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsFound)
	for err := range errorsFound {
		t.Fatal(err)
	}
}

func TestRRFCompatibilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/rrf_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		BaselineCommit string `json:"baseline_commit"`
		HeaderHash     string `json:"header_sha256"`
		SourceHash     string `json:"source_sha256"`
		TestsHash      string `json:"tests_sha256"`
		RankConstant   int    `json:"default_rank_constant"`
		Formula        string `json:"formula"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.HeaderHash != "bc1949536968bc27f0cb11026d0ab8633dbb46641365455c20b433367837c7d6" ||
		fixture.SourceHash != "3c93edc12303898af52911589c46c720072f9470446858fc36a61206d538aa1e" ||
		fixture.TestsHash != "05a03cacf74e7615661cec3153b2d2307f6a991510018386f6650f2625cb9a7d" ||
		fixture.RankConstant != DefaultRRFRankConstant || fixture.Formula != "1/(rank_constant+rank+1)" {
		t.Fatalf("RRF compatibility fixture mismatch: %#v", fixture)
	}
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
		if err != nil {
			t.Fatal(err)
		}
		second, err := reranker.Rerank(context.Background(), batches, topK)
		if err != nil || !reflect.DeepEqual(first, second) {
			t.Fatalf("non-deterministic output: %#v / %#v, %v", first, second, err)
		}
		if len(first) > topK {
			t.Fatalf("result length %d exceeds %d", len(first), topK)
		}
		seen := make(map[string]struct{}, len(first))
		for index, document := range first {
			if _, found := seen[document.PrimaryKey]; found {
				t.Fatalf("duplicate primary key %q", document.PrimaryKey)
			}
			seen[document.PrimaryKey] = struct{}{}
			if document.Score <= 0 || math.IsNaN(float64(document.Score)) || math.IsInf(float64(document.Score), 0) {
				t.Fatalf("invalid score %v", document.Score)
			}
			if index > 0 && first[index-1].Score < document.Score {
				t.Fatalf("scores are not descending: %#v", first)
			}
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
		if _, err := reranker.Rerank(context.Background(), batches, 100); err != nil {
			b.Fatal(err)
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
