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
	"testing"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/stretchr/testify/require"
)

func TestRecallPercentileAndSearchSummary(t *testing.T) {
	require.InDelta(t, 0.5, recallAtK(2, []int64{1, 2}, []int64{2, 3}), 1e-12)
	require.InDelta(t, 0.5, recallAtK(2, []int64{1, 2}, []int64{1, 1}), 1e-12)
	require.InDelta(t, 2.5, percentile([]float64{1, 2, 3}, 0.75), 1e-12)
	metrics := summarizeSearch([]float64{1, 2, 3}, 3*time.Second)
	require.Equal(t, 3, metrics.Queries)
	require.InDelta(t, 1, metrics.QPS, 1e-12)
	require.InDelta(t, 2, metrics.LatencyAvgMS, 1e-12)
}

func TestVectorDBBenchEndToEndCustomDataset(t *testing.T) {
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
	err := runCLI(context.Background(), []string{
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
	}, &stdout, &stderr)
	require.NoError(t, err, stderr.String())
	var report benchmarkReport
	require.NoError(t, json.Unmarshal(stdout.Bytes(), &report), fmt.Sprintf("stdout: %s", stdout.String()))
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
