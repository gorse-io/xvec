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
	"context"
	"path/filepath"
	"testing"

	"github.com/parquet-go/parquet-go"
	"github.com/stretchr/testify/require"
)

func TestReadVectorDBBenchParquetDataAndTrainingBatches(t *testing.T) {
	directory := t.TempDir()
	training := []vectorParquetRow{
		{ID: 0, Embedding: []float32{0, 1}},
		{ID: 1, Embedding: []float32{1, 2}},
		{ID: 2, Embedding: []float32{2, 3}},
		{ID: 3, Embedding: []float32{3, 4}},
	}
	queries := []vectorParquetRow{
		{ID: 10, Embedding: []float32{0, 1}},
		{ID: 11, Embedding: []float32{2, 3}},
	}
	neighbors := []neighborParquetRow{
		{ID: 10, Neighbors: []int64{0, 1}},
		{ID: 11, Neighbors: []int64{2, 3}},
	}
	require.NoError(t, parquet.WriteFile(filepath.Join(directory, "train.parquet"), training))
	require.NoError(t, parquet.WriteFile(filepath.Join(directory, testFileName), queries))
	require.NoError(t, parquet.WriteFile(filepath.Join(directory, neighborsFileName), neighbors))

	data, err := readQueryData(context.Background(), directory, 2, 1)
	require.NoError(t, err)
	require.Equal(t, []int64{10}, data.IDs)
	require.Equal(t, [][]int64{{0, 1}}, data.GroundTruth)

	var batches [][]int64
	count, err := forEachTrainingBatch(
		context.Background(), directory, []string{"train.parquet"}, 2, 3,
		func(rows []vectorParquetRow) error {
			ids := make([]int64, len(rows))
			for index := range rows {
				ids[index] = rows[index].ID
			}
			batches = append(batches, ids)
			return nil
		},
	)
	require.NoError(t, err)
	require.Equal(t, int64(3), count)
	require.Equal(t, [][]int64{{0, 1}, {2}}, batches)
}

func TestDatasetFileNameAndURL(t *testing.T) {
	require.NoError(t, validateDatasetFileName("shuffle_train-00-of-10.parquet"))
	require.Error(t, validateDatasetFileName("../train.parquet"))
	require.Equal(t,
		"https://assets.zilliz.com/benchmark/cohere_medium_1m/test.parquet",
		datasetFileURL("https://assets.zilliz.com/benchmark/", "cohere_medium_1m", "test.parquet"),
	)
}
