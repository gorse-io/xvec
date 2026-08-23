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
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNativeMSMarcoFTSSource(t *testing.T) {
	directory := t.TempDir()
	writeTarGzip(t, filepath.Join(directory, msMarcoArchiveName), map[string]string{
		"qrels.dev.small.tsv":   "q1\t0\tlate\t1\n",
		"queries.dev.small.tsv": "q1\talpha query\nq2\tignored\n",
		"collection.tsv":        "0\talpha\n1\tbeta\n2\tgamma\nlate\tlate alpha\n",
	})
	require.NoError(t, extractMSMarcoSourceFiles(context.Background(), filepath.Join(directory, msMarcoArchiveName), directory, true))
	require.True(t, hasExtractedMSMarcoDataset(directory, true))
	config := benchConfig{DatasetDir: directory, caseSpec: benchmarkCase{
		Workload: workloadFullText, DatasetName: "msmarco", Size: 3,
	}}
	data, err := readFTSQueryData(context.Background(), config, 0)
	require.NoError(t, err)
	require.Equal(t, []string{"q1"}, data.IDs)
	require.Equal(t, []map[string]int{{"late": 1}}, data.GroundTruth)

	var ids []string
	count, err := forEachFTSDocumentBatch(context.Background(), config, 2, 0, func(rows []ftsDocumentRow) error {
		for _, row := range rows {
			ids = append(ids, row.ID)
		}
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, int64(3), count)
	require.Equal(t, []string{"0", "1", "late"}, ids)
	require.Equal(t, "–", fixMSMarcoEncoding("\u00e2\u0080\u0093"))
}

func TestNativeHotpotQAFTSSource(t *testing.T) {
	directory := t.TempDir()
	writeZip(t, filepath.Join(directory, hotpotArchiveName), map[string]string{
		"hotpotqa/qrels/test.tsv": "query-id\tcorpus-id\tscore\nq1\td2\t2\n",
		"hotpotqa/queries.jsonl":  "{\"_id\":\"q1\",\"text\":\"alpha query\"}\n",
		"hotpotqa/corpus.jsonl":   "{\"_id\":\"d1\",\"title\":\"First\",\"text\":\"alpha\"}\n{\"_id\":\"d2\",\"title\":\"Second\",\"text\":\"beta\"}\n",
	})
	config := benchConfig{DatasetDir: directory, caseSpec: benchmarkCase{
		Workload: workloadFullText, DatasetName: "hotpotqa", Size: 2,
	}}
	data, err := readFTSQueryData(context.Background(), config, 0)
	require.NoError(t, err)
	require.Equal(t, []string{"q1"}, data.IDs)
	require.Equal(t, []map[string]int{{"d2": 2}}, data.GroundTruth)

	var documents []ftsDocumentRow
	count, err := forEachFTSDocumentBatch(context.Background(), config, 2, 0, func(rows []ftsDocumentRow) error {
		documents = append(documents, rows...)
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, int64(2), count)
	require.Equal(t, "First alpha", documents[0].Text)
	require.Equal(t, "Second beta", documents[1].Text)
}

func TestFTSFilterIDIsPermutation(t *testing.T) {
	seen := make(map[int64]struct{})
	for ordinal := int64(0); ordinal < 100; ordinal++ {
		value := ftsFilterID(ordinal, 100)
		require.GreaterOrEqual(t, value, int64(0))
		require.Less(t, value, int64(100))
		_, duplicate := seen[value]
		require.False(t, duplicate)
		seen[value] = struct{}{}
	}
}

func writeTarGzip(t *testing.T, path string, files map[string]string) {
	t.Helper()
	output, err := os.Create(path)
	require.NoError(t, err)
	compressed := gzip.NewWriter(output)
	archive := tar.NewWriter(compressed)
	for name, content := range files {
		require.NoError(t, archive.WriteHeader(&tar.Header{Name: name, Mode: 0o600, Size: int64(len(content))}))
		_, err := archive.Write([]byte(content))
		require.NoError(t, err)
	}
	require.NoError(t, archive.Close())
	require.NoError(t, compressed.Close())
	require.NoError(t, output.Close())
}

func writeZip(t *testing.T, path string, files map[string]string) {
	t.Helper()
	output, err := os.Create(path)
	require.NoError(t, err)
	archive := zip.NewWriter(output)
	for name, content := range files {
		file, err := archive.Create(name)
		require.NoError(t, err)
		_, err = file.Write([]byte(content))
		require.NoError(t, err)
	}
	require.NoError(t, archive.Close())
	require.NoError(t, output.Close())
}
