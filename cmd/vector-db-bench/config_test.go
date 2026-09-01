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
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestParseConfigVectorDBBenchCases(t *testing.T) {
	config, err := parseConfig([]string{
		backendXvec,
		"--path", t.TempDir(),
		"--case-type", casePerformance768D10M,
		"--num-concurrency", "12,14,16",
		"--concurrency-duration", "3",
		"--serial-cooldown", "0.25",
		"--quantize-type", "int8",
		"--is-using-refiner",
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, backendXvec, config.Backend)
	require.Equal(t, casePerformance768D10M, config.caseSpec.Name)
	require.Equal(t, 768, config.caseSpec.Dimension)
	require.Equal(t, int64(10_000_000), config.caseSpec.Size)
	require.Len(t, config.caseSpec.TrainFiles, 10)
	require.Equal(t, []int{12, 14, 16}, config.concurrency)
	require.Equal(t, 3*time.Second, config.ConcurrencyDuration)
	require.Equal(t, 250*time.Millisecond, config.SerialCooldown)
	require.True(t, config.UseRefiner)
}

func TestResolveVectorDBBenchDatasetCases(t *testing.T) {
	testCases := []struct {
		caseType      string
		datasetName   string
		datasetFolder string
		size          int64
		dimension     int
		metric        string
		trainCount    int
		firstTrain    string
		lastTrain     string
	}{
		{"Performance768D100K", "cohere", "cohere_small_100k", 100_000, 768, "cosine", 1, "shuffle_train.parquet", "shuffle_train.parquet"},
		{"Performance768D1M", "cohere", "cohere_medium_1m", 1_000_000, 768, "cosine", 1, "shuffle_train.parquet", "shuffle_train.parquet"},
		{"Performance768D10M", "cohere", "cohere_large_10m", 10_000_000, 768, "cosine", 10, "shuffle_train-00-of-10.parquet", "shuffle_train-09-of-10.parquet"},
		{"Performance768D100M", "laion", "laion_large_100m", 100_000_000, 768, "l2", 100, "train-00-of-100.parquet", "train-99-of-100.parquet"},
		{"Performance1024D1M", "bioasq", "bioasq_medium_1m", 1_000_000, 1024, "cosine", 1, "shuffle_train.parquet", "shuffle_train.parquet"},
		{"Performance1024D10M", "bioasq", "bioasq_large_10m", 10_000_000, 1024, "cosine", 10, "shuffle_train-00-of-10.parquet", "shuffle_train-09-of-10.parquet"},
		{"Performance1536D50K", "openai", "openai_small_50k", 50_000, 1536, "cosine", 1, "shuffle_train.parquet", "shuffle_train.parquet"},
		{"Performance1536D500K", "openai", "openai_medium_500k", 500_000, 1536, "cosine", 1, "shuffle_train.parquet", "shuffle_train.parquet"},
		{"Performance1536D5M", "openai", "openai_large_5m", 5_000_000, 1536, "cosine", 10, "shuffle_train-00-of-10.parquet", "shuffle_train-09-of-10.parquet"},
	}
	for _, testCase := range testCases {
		t.Run(testCase.caseType, func(t *testing.T) {
			resolved, err := resolveBenchmarkCase(benchConfig{CaseType: testCase.caseType})
			require.NoError(t, err)
			require.Equal(t, testCase.caseType, resolved.Name)
			require.Equal(t, testCase.datasetName, resolved.DatasetName)
			require.Equal(t, testCase.datasetFolder, resolved.DatasetFolder)
			require.Equal(t, testCase.size, resolved.Size)
			require.Equal(t, testCase.dimension, resolved.Dimension)
			require.Equal(t, testCase.metric, resolved.Metric)
			require.Len(t, resolved.TrainFiles, testCase.trainCount)
			require.Equal(t, testCase.firstTrain, resolved.TrainFiles[0])
			require.Equal(t, testCase.lastTrain, resolved.TrainFiles[len(resolved.TrainFiles)-1])
		})
	}
}

func TestParseConfigCustomAndValidation(t *testing.T) {
	datasetDir := t.TempDir()
	config, err := parseConfig([]string{
		backendZvec,
		"--path", t.TempDir(), "--case-type", caseCustom,
		"--dataset-dir", datasetDir, "--dimension", "3", "--metric", "l2",
		"--train-files", "part-0.parquet,part-1.parquet",
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, backendZvec, config.Backend)
	require.Equal(t, []string{"part-0.parquet", "part-1.parquet"}, config.caseSpec.TrainFiles)
	require.Equal(t, "l2", config.caseSpec.Metric)

	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--skip-load"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "skip-load requires skip-drop-old")
	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--num-concurrency", "1,1"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "duplicate concurrency")
	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--serial-cooldown", "-1"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "serial-cooldown cannot be negative")
	_, err = parseConfig([]string{"--path", t.TempDir()}, &bytes.Buffer{})
	require.ErrorContains(t, err, "backend is required")
	_, err = parseConfig([]string{"unknown", "--path", t.TempDir()}, &bytes.Buffer{})
	require.ErrorContains(t, err, "unsupported backend")
}

func TestResolveVectorDBBenchFTSCases(t *testing.T) {
	testCases := []struct {
		label, dataset, folder string
		size                   int64
	}{
		{ftsMSMarcoSmall, "msmarco", "msmarco_small_100k", 100_000},
		{ftsMSMarcoMedium, "msmarco", "msmarco_medium_1m", 1_000_000},
		{ftsMSMarcoLarge, "msmarco", "msmarco_large_8.8m", 8_841_823},
		{ftsHotpotSmall, "hotpotqa", "hotpotqa_small_100k", 100_000},
		{ftsHotpotMedium, "hotpotqa", "hotpotqa_medium_1m", 1_000_000},
		{ftsHotpotLarge, "hotpotqa", "hotpotqa_large_5.2m", 5_233_329},
	}
	for _, testCase := range testCases {
		resolved, err := resolveBenchmarkCase(benchConfig{CaseType: caseFTSBm25Performance, DatasetWithSizeType: testCase.label})
		require.NoError(t, err)
		require.Equal(t, workloadFullText, resolved.Workload)
		require.Equal(t, testCase.dataset, resolved.DatasetName)
		require.Equal(t, testCase.folder, resolved.DatasetFolder)
		require.Equal(t, testCase.size, resolved.Size)
	}
}

func TestParseConfigDiskANN(t *testing.T) {
	config, err := parseConfig([]string{
		backendXvec, "--path", t.TempDir(), "--index-type", indexDiskANN,
		"--diskann-max-degree", "64", "--diskann-build-list", "128",
		"--diskann-pq-chunks", "32", "--diskann-query-list", "200",
		"--optimize-concurrency", "4",
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, indexDiskANN, config.IndexType)
	require.Equal(t, 64, config.DiskANNMaxDegree)
	require.Equal(t, 128, config.DiskANNBuildList)
	require.Equal(t, 32, config.DiskANNPQChunks)
	require.Equal(t, 200, config.DiskANNQueryList)
	require.Equal(t, 4, config.OptimizeConcurrency)

	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--index-type", "bad"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "unsupported index-type")
	_, err = parseConfig([]string{backendXvec, "--path", t.TempDir(), "--index-type", indexDiskANN, "--diskann-query-list", "0"}, &bytes.Buffer{})
	require.ErrorContains(t, err, "DiskANN parameters")
}

func TestParseConfigFlat(t *testing.T) {
	config, err := parseConfig([]string{
		backendXvec, "--path", t.TempDir(), "--index-type", indexFlat,
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, indexFlat, config.IndexType)
}

func TestParseConfigIVF(t *testing.T) {
	config, err := parseConfig([]string{
		backendXvec, "--path", t.TempDir(), "--index-type", indexIVF,
		"--ivf-n-list", "512", "--ivf-n-iterations", "12",
		"--ivf-use-soar", "--ivf-n-probe", "32", "--ivf-scale-factor", "8",
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, indexIVF, config.IndexType)
	require.Equal(t, 512, config.IVFNList)
	require.Equal(t, 12, config.IVFNIterations)
	require.True(t, config.IVFUseSOAR)
	require.Equal(t, 32, config.IVFNProbe)
	require.Equal(t, 8.0, config.IVFScaleFactor)

	_, err = parseConfig([]string{
		backendXvec, "--path", t.TempDir(), "--index-type", indexIVF, "--ivf-n-probe", "0",
	}, &bytes.Buffer{})
	require.ErrorContains(t, err, "IVF parameters")
}

func TestParseConfigVamana(t *testing.T) {
	config, err := parseConfig([]string{
		backendXvec, "--path", t.TempDir(), "--index-type", indexVamana,
	}, &bytes.Buffer{})
	require.NoError(t, err)
	require.Equal(t, indexVamana, config.IndexType)
	require.Equal(t, defaultVamanaEFSearch, config.EFSearch)

	_, err = parseConfig([]string{
		backendZvec, "--path", t.TempDir(), "--index-type", indexVamana,
		"--ef-search", "300",
	}, &bytes.Buffer{})
	require.ErrorContains(t, err, "zvec Vamana requires")

	_, err = parseConfig([]string{
		backendXvec, "--path", t.TempDir(), "--index-type", indexVamana,
		"--ef-search", "0",
	}, &bytes.Buffer{})
	require.ErrorContains(t, err, "vamana parameters")
}

func TestParseFlexibleDuration(t *testing.T) {
	duration, err := parseFlexibleDuration("0.25")
	require.NoError(t, err)
	require.Equal(t, 250*time.Millisecond, duration)
	duration, err = parseFlexibleDuration("250ms")
	require.NoError(t, err)
	require.Equal(t, 250*time.Millisecond, duration)
}
