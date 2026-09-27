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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/stretchr/testify/require"
)

func TestRecallPercentileAndSearchSummary(t *testing.T) {
	require.InDelta(t, 0.5, recallAtK(2, map[string]int{"1": 2, "2": 1}, []string{"2", "3"}), 1e-12)
	require.InDelta(t, 0.5, recallAtK(2, map[string]int{"1": 2, "2": 1}, []string{"1", "1"}), 1e-12)
	require.InDelta(t, 1, recallAtK(1, map[string]int{"1": 2, "2": 1}, []string{"1"}), 1e-12)
	require.Zero(t, recallAtK(1, map[string]int{"1": 2, "2": 1}, []string{"2"}))
	require.InDelta(t, 0.5, recallFTSAtK(10, map[string]int{"a": 2, "b": 1}, []string{"a", "x"}), 1e-12)
	require.InDelta(t, 0.5, mrrFTSAtK(10, map[string]int{"a": 1}, []string{"x", "a"}), 1e-12)
	require.InDelta(t, 1, ndcgFTSAtK(10, map[string]int{"a": 2, "b": 1}, []string{"a", "b"}), 1e-12)
	require.InDelta(t, 2.5, percentile([]float64{1, 2, 3}, 0.75), 1e-12)
	metrics := summarizeSearch([]float64{1, 2, 3}, 3*time.Second)
	require.Equal(t, 3, metrics.Queries)
	require.InDelta(t, 1, metrics.QPS, 1e-12)
	require.InDelta(t, 2, metrics.LatencyAvgMS, 1e-12)
}

func TestVectorDBBenchEndToEndCustomDataset(t *testing.T) {
	testVectorDBBenchEndToEndCustomDataset(t, backendXvec)
}

func TestVectorDBBenchFlatEndToEndCustomDataset(t *testing.T) {
	testVectorDBBenchEndToEndCustomDataset(t, backendXvec, "--index-type", indexFlat)
}

func TestVectorDBBenchDiskANNEndToEndCustomDataset(t *testing.T) {
	testVectorDBBenchEndToEndCustomDataset(t, backendXvec, "--index-type", indexDiskANN)
}

func TestVectorDBBenchVamanaEndToEndCustomDataset(t *testing.T) {
	testVectorDBBenchEndToEndCustomDataset(t, backendXvec, "--index-type", indexVamana)
}

func TestVectorDBBenchEndToEndFTSDataset(t *testing.T) {
	testVectorDBBenchEndToEndFTSDataset(t, backendXvec)
}

func TestVectorDBBenchFilteredVectorEndToEnd(t *testing.T) {
	testVectorDBBenchFilteredVectorEndToEnd(t, backendXvec)
}

func testVectorDBBenchFilteredVectorEndToEnd(t *testing.T, backend string) {
	t.Helper()
	testCases := []struct {
		name            string
		caseType        string
		filterRate      float64
		labelPercentage float64
		groundTruthFile string
		withLabels      bool
	}{
		{"integer", caseNewIntFilterPerformance, 0.5, 0, "neighbors_int_50p.parquet", false},
		{"label", caseLabelFilterPerformance, 0, 0.5, "neighbors_labels_label_50p.parquet", true},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			directory := t.TempDir()
			datasetDir := filepath.Join(directory, "dataset")
			require.NoError(t, mkdir(datasetDir))
			training := []vectorParquetRow{
				{ID: 3, Embedding: []float32{3, 0}},
				{ID: 0, Embedding: []float32{0, 0}},
				{ID: 2, Embedding: []float32{2, 0}},
				{ID: 1, Embedding: []float32{1, 0}},
			}
			queries := []vectorParquetRow{{ID: 100, Embedding: []float32{0, 0}}}
			neighbors := []neighborParquetRow{{ID: 100, Neighbors: []int64{2}}}
			require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, "train.parquet"), training))
			require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, testFileName), queries))
			require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, testCase.groundTruthFile), neighbors))
			if testCase.withLabels {
				require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, scalarLabelsFileName), []scalarLabelParquetRow{
					{ID: 0, Label: "other"},
					{ID: 1, Label: "other"},
					{ID: 2, Label: "label_50p"},
					{ID: 3, Label: "label_50p"},
				}))
			}

			config := benchConfig{
				Backend: backend, Path: filepath.Join(directory, "collection"), CaseType: testCase.caseType,
				DatasetDir: datasetDir, DatasetBaseURL: "https://assets.zilliz.com/benchmark",
				FilterRate: testCase.filterRate, LabelPercentage: testCase.labelPercentage,
				K: 1, BatchSize: 2, IndexType: indexFlat, NumConcurrency: "1",
				SkipDownload: true, SkipConcurrentSearch: true, MaxDocsPerSegment: 1000,
				MaxBufferSize: 64 << 20, PayloadProfile: "ids_only",
				caseSpec: benchmarkCase{
					Name: testCase.caseType, Workload: workloadVector, DatasetName: "custom", DatasetFolder: "custom",
					Size: 4, Dimension: 2, Metric: "l2", TrainFiles: []string{"train.parquet"},
				},
			}
			report, err := runBenchmark(context.Background(), config, &bytes.Buffer{})
			require.NoError(t, err)
			require.NotNil(t, report.Serial)
			require.InDelta(t, 1, report.Serial.Recall, 1e-12)
		})
	}
}

func testVectorDBBenchEndToEndFTSDataset(t *testing.T, backend string) {
	t.Helper()
	directory := t.TempDir()
	datasetDir := filepath.Join(directory, "dataset")
	require.NoError(t, mkdir(datasetDir))
	writeJSONL := func(name string, rows ...string) {
		require.NoError(t, os.WriteFile(filepath.Join(datasetDir, name), []byte(strings.Join(rows, "\n")+"\n"), 0o600))
	}
	writeJSONL(ftsDocumentsFileName,
		`{"id":"a","text":"alpha alpha beta"}`,
		`{"id":"b","text":"beta gamma"}`,
		`{"id":"c","text":"delta"}`,
	)
	writeJSONL(ftsQueriesFileName,
		`{"id":"q1","text":"alpha"}`,
		`{"id":"q2","text":"gamma"}`,
	)
	writeJSONL(ftsQrelsFileName,
		`{"query_id":"q1","doc_id":"a","relevance":2}`,
		`{"query_id":"q2","doc_id":"b","relevance":1}`,
	)

	var stdout, stderr bytes.Buffer
	err := runCLI(context.Background(), []string{
		backend, "--path", filepath.Join(directory, "collection"),
		"--case-type", caseFTSBm25Performance,
		"--dataset-with-size-type", ftsMSMarcoSmall,
		"--dataset-dir", datasetDir,
		"--k", "2", "--batch-size", "2", "--num-concurrency", "1",
		"--concurrency-duration", "25ms", "--warmup-queries", "1",
		"--max-docs-per-segment", "1000",
	}, &stdout, &stderr)
	require.NoError(t, err, stderr.String())
	var report benchmarkReport
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &report))
	require.Equal(t, workloadFullText, report.Case.Workload)
	require.Equal(t, int64(3), report.Load.Rows)
	require.InDelta(t, 1, report.Serial.Recall, 1e-12)
	require.InDelta(t, 1, report.Serial.MRR, 1e-12)
	require.InDelta(t, 1, report.Serial.NDCG, 1e-12)
}

func testVectorDBBenchEndToEndCustomDataset(t *testing.T, backend string, extraArgs ...string) {
	t.Helper()
	directory := t.TempDir()
	datasetDir := filepath.Join(directory, "dataset")
	require.NoError(t, mkdir(datasetDir))
	training := make([]vectorParquetRow, 32)
	for index := range training {
		training[index] = vectorParquetRow{ID: int64(index), Embedding: []float32{float32(index), 0}}
	}
	queries := []vectorParquetRow{
		{ID: 100, Embedding: []float32{0, 0}},
		{ID: 101, Embedding: []float32{7, 0}},
		{ID: 102, Embedding: []float32{31, 0}},
	}
	neighbors := []neighborParquetRow{
		{ID: 100, Neighbors: []int64{0}},
		{ID: 101, Neighbors: []int64{7}},
		{ID: 102, Neighbors: []int64{31}},
	}
	require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, "train.parquet"), training))
	require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, testFileName), queries))
	require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, neighborsFileName), neighbors))

	var stdout, stderr bytes.Buffer
	args := []string{
		backend,
		"--path", filepath.Join(directory, "collection"),
		"--case-type", caseCustom,
		"--dataset-dir", datasetDir,
		"--train-files", "train.parquet",
		"--dimension", "2",
		"--metric", "l2",
		"--skip-download",
		"--k", "1",
		"--batch-size", "7",
		"--m", "4",
		"--ef-construction", "16",
		"--ef-search", "16",
		"--num-concurrency", "1,2",
		"--concurrency-duration", "50ms",
		"--warmup-queries", "2",
		"--max-docs-per-segment", "1000",
	}
	args = append(args, extraArgs...)
	err := runCLI(context.Background(), args, &stdout, &stderr)
	require.NoError(t, err, stderr.String())
	var report benchmarkReport
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &report), fmt.Sprintf("stdout: %s", stdout.String()))
	require.Equal(t, backend, report.Config.Backend)
	require.NotNil(t, report.Load)
	require.Equal(t, int64(32), report.Load.Rows)
	require.NotNil(t, report.Serial)
	require.InDelta(t, 1, report.Serial.Recall, 1e-12)
	require.Equal(t, 3, report.Serial.Queries)
	require.Len(t, report.Concurrent, 2)
	require.Positive(t, report.Concurrent[0].QPS)
	require.Positive(t, report.Concurrent[1].QPS)
	require.NotNil(t, report.VectorDBBench)
	require.Equal(t, []int{1, 2}, report.VectorDBBench.Concurrency)
	require.Equal(t, int64(32), report.VectorDBBench.InsertedCount)
	require.InDelta(t, 1, report.VectorDBBench.Recall, 1e-12)
}

func mkdir(path string) error {
	return os.MkdirAll(path, 0o755)
}
