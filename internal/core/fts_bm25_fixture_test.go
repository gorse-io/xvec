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

package core

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBM25MergeBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/fts_bm25_merge_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		BaselineCommit    string `json:"baseline_commit"`
		BM25HeaderSHA256  string `json:"bm25_header_sha256"`
		BM25SourceSHA256  string `json:"bm25_source_sha256"`
		ReducerHeaderHash string `json:"reducer_header_sha256"`
		ReducerSourceHash string `json:"reducer_source_sha256"`
		BM25Cases         []struct {
			Name              string  `json:"name"`
			TotalDocuments    uint64  `json:"total_documents"`
			TotalTokens       uint64  `json:"total_tokens"`
			DocumentFrequency uint64  `json:"document_frequency"`
			TermFrequency     uint32  `json:"term_frequency"`
			DocumentLength    uint32  `json:"document_length"`
			IDF               float64 `json:"idf"`
			Score             float64 `json:"score"`
		} `json:"bm25_cases"`
		MergeCase struct {
			SourceDocumentCounts []uint64   `json:"source_document_counts"`
			DeletedDocumentIDs   [][]uint32 `json:"deleted_document_ids"`
			OutputDocumentCount  uint64     `json:"output_document_count"`
			OutputTotalTokens    uint64     `json:"output_total_tokens"`
			OutputTerms          []string   `json:"output_terms"`
		} `json:"merge_case"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b",

		"baseline identity drift")
	require.True(t, fixture.BM25HeaderSHA256 == "eb1a0a292fb8eddb3854b8256b6a0c7ad4920aabaa31a65b53e8fadde0ef0f8a",

		"baseline identity drift")
	require.True(t, fixture.BM25SourceSHA256 == "73a6090ad0f22c7cc67a57dd4b794a00b8a38a1fe80d51d4700833acc1990389",

		"baseline identity drift")
	require.True(t, fixture.ReducerHeaderHash == "b87f60888e230f39268dea6614d7aadca60f8945bc3bc0862f2fa026d1aeb43b",

		"baseline identity drift")
	require.True(t, fixture.ReducerSourceHash == "7bebfd9d410c598e2970c0d3b7331b9623e2b481b28466b2eae7543af7d58b2e",
		"baseline identity drift")

	for _, test := range fixture.BM25Cases {
		t.Run(test.Name, func(t *testing.T) {
			stats := FTSCorpusStats{
				TotalDocuments: test.TotalDocuments, TotalTokens: test.TotalTokens,
				documentFrequencies: map[string]uint64{"term": test.DocumentFrequency},
			}
			scorer, err := NewBM25Scorer(DefaultBM25Params(), stats)
			require.NoError(t, err)

			assertBM25Close(t, "fixture IDF", scorer.TermIDF("term"), test.IDF, 1e-6)
			assertBM25Close(t, "fixture score", scorer.Score(test.DocumentFrequency, test.TermFrequency, test.DocumentLength), test.Score, 1e-6)
		})
	}
	require.Len(t, fixture.MergeCase.SourceDocumentCounts, 2,

		"merge fixture drift")
	require.Len(t, fixture.MergeCase.DeletedDocumentIDs, 2,

		"merge fixture drift")
	require.True(t, fixture.MergeCase.OutputDocumentCount == 3,

		"merge fixture drift")
	require.True(t, fixture.MergeCase.OutputTotalTokens == 6,

		"merge fixture drift")
	require.Len(t, fixture.MergeCase.OutputTerms, 2,
		"merge fixture drift")
}
