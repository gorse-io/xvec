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
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

const (
	testFileName      = "test.parquet"
	neighborsFileName = "neighbors.parquet"
)

type vectorParquetRow struct {
	ID        int64     `parquet:"id"`
	Embedding []float32 `parquet:"emb,list"`
}

type neighborParquetRow struct {
	ID        int64   `parquet:"id"`
	Neighbors []int64 `parquet:"neighbors_id,list"`
}

type queryData struct {
	IDs         []int64
	Vectors     [][]float32
	GroundTruth [][]int64
}

func prepareDataset(ctx context.Context, config benchConfig, includeTrain bool, log io.Writer) error {
	files := []string{testFileName, neighborsFileName}
	if includeTrain {
		files = append(files, config.caseSpec.TrainFiles...)
	}
	if err := os.MkdirAll(config.DatasetDir, 0o755); err != nil {
		return fmt.Errorf("create dataset directory: %w", err)
	}
	client := &http.Client{Timeout: 0}
	for _, name := range files {
		if err := validateDatasetFileName(name); err != nil {
			return err
		}
		localPath := filepath.Join(config.DatasetDir, name)
		remoteURL := datasetFileURL(config.DatasetBaseURL, config.caseSpec.DatasetFolder, name)
		if config.SkipDownload {
			if err := requireNonEmptyFile(localPath); err != nil {
				return err
			}
			continue
		}
		complete, err := localFileMatchesRemote(ctx, client, localPath, remoteURL)
		if err != nil {
			return err
		}
		if complete {
			continue
		}
		if err := downloadDatasetFile(ctx, client, remoteURL, localPath, log); err != nil {
			return err
		}
	}
	return nil
}

func validateDatasetFileName(name string) error {
	if name == "" || filepath.IsAbs(name) || filepath.Base(name) != name || name == "." || name == ".." {
		return fmt.Errorf("unsafe dataset file name %q", name)
	}
	return nil
}

func datasetFileURL(baseURL, datasetFolder, name string) string {
	return strings.TrimRight(baseURL, "/") + "/" + url.PathEscape(datasetFolder) + "/" + url.PathEscape(name)
}

func requireNonEmptyFile(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("required dataset file %s: %w", path, err)
	}
	if !info.Mode().IsRegular() || info.Size() == 0 {
		return fmt.Errorf("required dataset file %s is not a non-empty regular file", path)
	}
	return nil
}

func localFileMatchesRemote(ctx context.Context, client *http.Client, localPath, remoteURL string) (bool, error) {
	info, err := os.Stat(localPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return false, fmt.Errorf("stat dataset file %s: %w", localPath, err)
	}
	if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
		return false, nil
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodHead, remoteURL, nil)
	if err != nil {
		return false, fmt.Errorf("create dataset HEAD request: %w", err)
	}
	response, err := client.Do(request)
	if err != nil {
		return false, fmt.Errorf("check remote dataset file %s: %w", remoteURL, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return false, fmt.Errorf("check remote dataset file %s: %s", remoteURL, response.Status)
	}
	return response.ContentLength < 0 || response.ContentLength == info.Size(), nil
}

func downloadDatasetFile(ctx context.Context, client *http.Client, remoteURL, localPath string, log io.Writer) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, remoteURL, nil)
	if err != nil {
		return fmt.Errorf("create dataset download request: %w", err)
	}
	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("download dataset file %s: %w", remoteURL, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download dataset file %s: %s", remoteURL, response.Status)
	}
	temporaryPath := localPath + ".part"
	file, err := os.OpenFile(temporaryPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("create partial dataset file %s: %w", temporaryPath, err)
	}
	_, _ = fmt.Fprintf(log, "downloading %s -> %s\n", remoteURL, localPath)
	started := time.Now()
	written, copyErr := io.Copy(file, response.Body)
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("write dataset file %s: %w", temporaryPath, errors.Join(copyErr, closeErr))
	}
	if response.ContentLength >= 0 && written != response.ContentLength {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("download dataset file %s: got %d bytes, want %d", remoteURL, written, response.ContentLength)
	}
	if err := os.Rename(temporaryPath, localPath); err != nil {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("publish dataset file %s: %w", localPath, err)
	}
	_, _ = fmt.Fprintf(log, "downloaded %s (%d bytes in %s)\n", localPath, written, time.Since(started).Round(time.Millisecond))
	return nil
}

func readQueryData(ctx context.Context, datasetDir string, dimension, limit int) (queryData, error) {
	queries, err := readVectorParquetRows(ctx, filepath.Join(datasetDir, testFileName), limit)
	if err != nil {
		return queryData{}, fmt.Errorf("read test vectors: %w", err)
	}
	neighbors, err := readNeighborParquetRows(ctx, filepath.Join(datasetDir, neighborsFileName), limit)
	if err != nil {
		return queryData{}, fmt.Errorf("read ground truth: %w", err)
	}
	if len(queries) == 0 || len(queries) != len(neighbors) {
		return queryData{}, fmt.Errorf("test and ground-truth row counts differ: %d and %d", len(queries), len(neighbors))
	}
	data := queryData{
		IDs: make([]int64, len(queries)), Vectors: make([][]float32, len(queries)), GroundTruth: make([][]int64, len(queries)),
	}
	for index := range queries {
		if err := ctx.Err(); err != nil {
			return queryData{}, err
		}
		if queries[index].ID != neighbors[index].ID {
			return queryData{}, fmt.Errorf("query ID mismatch at row %d: %d and %d", index, queries[index].ID, neighbors[index].ID)
		}
		if len(queries[index].Embedding) != dimension {
			return queryData{}, fmt.Errorf("query %d has dimension %d, want %d", queries[index].ID, len(queries[index].Embedding), dimension)
		}
		if len(neighbors[index].Neighbors) == 0 {
			return queryData{}, fmt.Errorf("query %d has empty ground truth", queries[index].ID)
		}
		data.IDs[index] = queries[index].ID
		data.Vectors[index] = queries[index].Embedding
		data.GroundTruth[index] = neighbors[index].Neighbors
	}
	return data, nil
}

func forEachParquetRow(ctx context.Context, path string, limit int64, consume func(parquet.Row) error) (count int64, err error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	info, err := file.Stat()
	if err != nil {
		return 0, err
	}
	parquetFile, err := parquet.OpenFile(file, info.Size())
	if err != nil {
		return 0, err
	}
	buffer := make([]parquet.Row, 256)
	for _, rowGroup := range parquetFile.RowGroups() {
		rows := rowGroup.Rows()
		for limit == 0 || count < limit {
			if err := ctx.Err(); err != nil {
				return count, errors.Join(err, rows.Close())
			}
			readBuffer := buffer
			if limit > 0 && int64(len(readBuffer)) > limit-count {
				readBuffer = readBuffer[:limit-count]
			}
			read, readErr := rows.ReadRows(readBuffer)
			for index := range read {
				if err := consume(readBuffer[index]); err != nil {
					return count, errors.Join(err, rows.Close())
				}
				count++
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			if readErr != nil {
				return count, errors.Join(readErr, rows.Close())
			}
			if read == 0 {
				break
			}
		}
		if err := rows.Close(); err != nil {
			return count, err
		}
		if limit > 0 && count >= limit {
			break
		}
	}
	return count, nil
}

func decodeVectorParquetRow(row parquet.Row) (vectorParquetRow, error) {
	decoded := vectorParquetRow{}
	hasID := false
	for _, value := range row {
		if value.IsNull() {
			continue
		}
		switch value.Column() {
		case 0:
			decoded.ID = value.Int64()
			hasID = true
		case 1:
			switch value.Kind() {
			case parquet.Float:
				decoded.Embedding = append(decoded.Embedding, value.Float())
			case parquet.Double:
				decoded.Embedding = append(decoded.Embedding, float32(value.Double()))
			default:
				return vectorParquetRow{}, fmt.Errorf("vector element has physical type %s, want FLOAT or DOUBLE", value.Kind())
			}
		}
	}
	if !hasID {
		return vectorParquetRow{}, errors.New("vector row has no id")
	}
	return decoded, nil
}

func decodeNeighborParquetRow(row parquet.Row) (neighborParquetRow, error) {
	decoded := neighborParquetRow{}
	hasID := false
	for _, value := range row {
		if value.IsNull() {
			continue
		}
		switch value.Column() {
		case 0:
			decoded.ID = value.Int64()
			hasID = true
		case 1:
			decoded.Neighbors = append(decoded.Neighbors, value.Int64())
		}
	}
	if !hasID {
		return neighborParquetRow{}, errors.New("neighbor row has no id")
	}
	return decoded, nil
}

func readVectorParquetRows(ctx context.Context, path string, limit int) ([]vectorParquetRow, error) {
	rows := make([]vectorParquetRow, 0)
	_, err := forEachParquetRow(ctx, path, int64(limit), func(row parquet.Row) error {
		decoded, err := decodeVectorParquetRow(row)
		if err == nil {
			rows = append(rows, decoded)
		}
		return err
	})
	return rows, err
}

func readNeighborParquetRows(ctx context.Context, path string, limit int) ([]neighborParquetRow, error) {
	rows := make([]neighborParquetRow, 0)
	_, err := forEachParquetRow(ctx, path, int64(limit), func(row parquet.Row) error {
		decoded, err := decodeNeighborParquetRow(row)
		if err == nil {
			rows = append(rows, decoded)
		}
		return err
	})
	return rows, err
}

func forEachTrainingBatch(
	ctx context.Context,
	datasetDir string,
	files []string,
	batchSize int,
	limit int64,
	consume func([]vectorParquetRow) error,
) (int64, error) {
	var total int64
	for _, name := range files {
		if limit > 0 && total >= limit {
			break
		}
		path := filepath.Join(datasetDir, name)
		buffer := make([]vectorParquetRow, 0, batchSize)
		remaining := int64(0)
		if limit > 0 {
			remaining = limit - total
		}
		count, err := forEachParquetRow(ctx, path, remaining, func(row parquet.Row) error {
			decoded, err := decodeVectorParquetRow(row)
			if err != nil {
				return err
			}
			buffer = append(buffer, decoded)
			if len(buffer) == batchSize {
				if err := consume(buffer); err != nil {
					return err
				}
				buffer = buffer[:0]
			}
			return nil
		})
		if err != nil {
			return total, fmt.Errorf("read training file %s: %w", path, err)
		}
		if len(buffer) > 0 {
			if err := consume(buffer); err != nil {
				return total, err
			}
		}
		total += count
	}
	return total, nil
}
