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
	"path/filepath"
	"testing"

	"github.com/parquet-go/parquet-go"
	"github.com/stretchr/testify/require"
)

func TestSQLiteVecBackendLoadAndSearch(t *testing.T) {
	datasetDir := t.TempDir()
	require.NoError(t, parquet.WriteFile(filepath.Join(datasetDir, "train.parquet"), []vectorParquetRow{
		{ID: 10, Embedding: []float32{1, 0}},
		{ID: 20, Embedding: []float32{0, 1}},
		{ID: 30, Embedding: []float32{-1, 0}},
	}))
	config := benchConfig{
		Backend:    backendSQLiteVec,
		Path:       filepath.Join(t.TempDir(), "bench.db"),
		DatasetDir: datasetDir,
		BatchSize:  2,
		K:          2,
		IndexType:  indexFlat,
		caseSpec: benchmarkCase{
			Workload: workloadVector, Dimension: 2, Metric: "cosine", TrainFiles: []string{"train.parquet"},
		},
	}

	shutdown, err := initializeBenchmarkBackend(config)
	require.NoError(t, err)
	shutdown()

	metrics, err := loadBenchmarkDataset(context.Background(), config, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, int64(3), metrics.Rows)
	require.Positive(t, metrics.StorageBytes)

	engine, closer, err := openBenchmarkQueryEngine(context.Background(), config)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, closer.Close()) })

	ids, err := engine.search(context.Background(), benchmarkQuery{Vector: []float32{0.9, 0.1}})
	require.NoError(t, err)
	require.Equal(t, []string{"10", "20"}, ids)
}

func TestSQLiteVecVectorBytes(t *testing.T) {
	require.Equal(t,
		[]byte{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0},
		sqliteVecVectorBytes([]float32{1, -2.5}),
	)
}

func TestOpenSQLiteVecQueryEngineRejectsMissingDatabase(t *testing.T) {
	_, _, err := openSQLiteVecQueryEngine(benchConfig{
		Path: filepath.Join(t.TempDir(), "missing.db"),
	})
	require.ErrorContains(t, err, "does not exist")
}
