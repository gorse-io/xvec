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
	"bufio"
	"compress/gzip"
	"context"
	"crypto/md5" // #nosec G501 -- MD5 verifies published dataset artifacts, not security credentials.
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	msMarcoArchiveName = "collectionandqueries.tar.gz"
	msMarcoArchiveURL  = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
	msMarcoArchiveMD5  = "31644046b18952c1386cd4564ba2ae69"
	hotpotArchiveName  = "hotpotqa.zip"
	hotpotArchiveURL   = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"
	hotpotArchiveMD5   = "f412724f78b0d91183a0e86805e16114"
)

var errStopFTSIteration = errors.New("stop FTS iteration")

func hasExportedFTSDataset(datasetDir string, includeTrain bool) bool {
	files := []string{ftsQueriesFileName, ftsQrelsFileName}
	if includeTrain {
		files = append(files, ftsDocumentsFileName)
	}
	for _, name := range files {
		if err := requireNonEmptyFile(filepath.Join(datasetDir, name)); err != nil {
			return false
		}
	}
	return true
}

func prepareFTSDataset(ctx context.Context, config benchConfig, includeTrain bool, log io.Writer) error {
	if hasExportedFTSDataset(config.DatasetDir, includeTrain) {
		return nil
	}
	if config.caseSpec.DatasetName == "msmarco" && hasExtractedMSMarcoDataset(config.DatasetDir, includeTrain) {
		return nil
	}
	if err := os.MkdirAll(config.DatasetDir, 0o755); err != nil {
		return fmt.Errorf("create FTS dataset directory: %w", err)
	}
	name, sourceURL, checksum, err := ftsArchiveSpec(config.caseSpec.DatasetName)
	if err != nil {
		return err
	}
	path := filepath.Join(config.DatasetDir, name)
	if config.SkipDownload {
		if err := requireNonEmptyFile(path); err != nil {
			return fmt.Errorf("required FTS source archive: %w", err)
		}
	} else {
		client := &http.Client{Timeout: 0}
		complete, err := localFileMatchesRemote(ctx, client, path, sourceURL)
		if err != nil {
			return err
		}
		if !complete {
			if err := downloadDatasetFile(ctx, client, sourceURL, path, log); err != nil {
				return err
			}
		}
	}
	if err := verifyFileMD5(ctx, path, checksum); err != nil {
		return err
	}
	if config.caseSpec.DatasetName == "msmarco" {
		if err := extractMSMarcoSourceFiles(ctx, path, config.DatasetDir, includeTrain); err != nil {
			return err
		}
	}
	return nil
}

func hasExtractedMSMarcoDataset(datasetDir string, includeTrain bool) bool {
	files := []string{ftsQueriesSourceName("msmarco"), ftsQrelsSourceName("msmarco")}
	if includeTrain {
		files = append(files, ftsDocumentsSourceName("msmarco"))
	}
	for _, name := range files {
		if requireNonEmptyFile(filepath.Join(datasetDir, name)) != nil {
			return false
		}
	}
	return true
}

func extractMSMarcoSourceFiles(ctx context.Context, archivePath, datasetDir string, includeTrain bool) error {
	wanted := map[string]bool{
		ftsQueriesSourceName("msmarco"): false,
		ftsQrelsSourceName("msmarco"):   false,
	}
	if includeTrain {
		wanted[ftsDocumentsSourceName("msmarco")] = false
	}
	remaining := 0
	for name := range wanted {
		if requireNonEmptyFile(filepath.Join(datasetDir, name)) == nil {
			wanted[name] = true
		} else {
			remaining++
		}
	}
	if remaining == 0 {
		return nil
	}
	file, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open MS MARCO archive: %w", err)
	}
	defer func() { _ = file.Close() }()
	compressed, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("open MS MARCO archive compression: %w", err)
	}
	defer func() { _ = compressed.Close() }()
	archive := tar.NewReader(compressed)
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		header, err := archive.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("read MS MARCO archive: %w", err)
		}
		name := filepath.ToSlash(header.Name)
		complete, found := wanted[name]
		if !found || complete {
			continue
		}
		destination := filepath.Join(datasetDir, name)
		temporary := destination + ".part"
		output, err := os.OpenFile(temporary, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
		if err != nil {
			return fmt.Errorf("create extracted MS MARCO file: %w", err)
		}
		_, copyErr := io.Copy(output, archive)
		closeErr := output.Close()
		if err := errors.Join(copyErr, closeErr); err != nil {
			_ = os.Remove(temporary)
			return fmt.Errorf("extract MS MARCO file %s: %w", name, err)
		}
		if err := os.Rename(temporary, destination); err != nil {
			_ = os.Remove(temporary)
			return fmt.Errorf("publish extracted MS MARCO file %s: %w", name, err)
		}
		wanted[name] = true
		remaining--
	}
	if remaining != 0 {
		missing := make([]string, 0, remaining)
		for name, complete := range wanted {
			if !complete {
				missing = append(missing, name)
			}
		}
		return fmt.Errorf("MS MARCO archive is missing: %s", strings.Join(missing, ", "))
	}
	return nil
}

func ftsArchiveSpec(datasetName string) (name, sourceURL, checksum string, err error) {
	switch datasetName {
	case "msmarco":
		return msMarcoArchiveName, msMarcoArchiveURL, msMarcoArchiveMD5, nil
	case "hotpotqa":
		return hotpotArchiveName, hotpotArchiveURL, hotpotArchiveMD5, nil
	default:
		return "", "", "", fmt.Errorf("unsupported FTS dataset %q", datasetName)
	}
}

func verifyFileMD5(ctx context.Context, path, expected string) (err error) {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open FTS source archive for checksum: %w", err)
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	hash := md5.New() // #nosec G401 -- matches the checksum published by the dataset provider.
	buffer := make([]byte, 1<<20)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		read, readErr := file.Read(buffer)
		if read > 0 {
			_, _ = hash.Write(buffer[:read])
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return fmt.Errorf("checksum FTS source archive: %w", readErr)
		}
	}
	actual := hex.EncodeToString(hash.Sum(nil))
	if !strings.EqualFold(actual, expected) {
		return fmt.Errorf("FTS source archive %s checksum is %s, want %s", path, actual, expected)
	}
	return nil
}

func forEachFTSDocumentBatch(ctx context.Context, config benchConfig, batchSize int, limit int64, consume func([]ftsDocumentRow) error) (int64, error) {
	if hasExportedFTSDataset(config.DatasetDir, true) {
		return forEachFTSDocumentJSONLBatch(ctx, config.DatasetDir, batchSize, limit, consume)
	}
	qrels, err := readFTSSourceQrels(ctx, config)
	if err != nil {
		return 0, err
	}
	selected := make(map[string]struct{})
	for _, documents := range qrels {
		for documentID := range documents {
			selected[documentID] = struct{}{}
		}
	}
	target := config.caseSpec.Size
	if limit > 0 && limit < target {
		target = limit
	}
	if int64(len(selected)) > target && limit == 0 {
		return 0, fmt.Errorf("FTS corpus size %d cannot contain %d qrel documents", target, len(selected))
	}
	buffer := make([]ftsDocumentRow, 0, batchSize)
	var emitted int64
	err = iterateFTSSourceDocuments(ctx, config, func(row ftsDocumentRow) error {
		_, required := selected[row.ID]
		if !required && int64(len(selected)) >= config.caseSpec.Size {
			return nil
		}
		selected[row.ID] = struct{}{}
		if emitted >= target {
			return errStopFTSIteration
		}
		row.FilterID = ftsFilterID(emitted, config.caseSpec.Size)
		buffer = append(buffer, row)
		emitted++
		if len(buffer) == batchSize {
			if err := consume(buffer); err != nil {
				return err
			}
			buffer = buffer[:0]
		}
		if emitted >= target {
			return errStopFTSIteration
		}
		return nil
	})
	if err != nil && !errors.Is(err, errStopFTSIteration) {
		return emitted, err
	}
	if len(buffer) > 0 {
		if err := consume(buffer); err != nil {
			return emitted, err
		}
	}
	if emitted != target {
		return emitted, fmt.Errorf("FTS source contains %d selected documents, want %d", emitted, target)
	}
	return emitted, nil
}

func readFTSQueryData(ctx context.Context, config benchConfig, limit int) (queryData, error) {
	if hasExportedFTSDataset(config.DatasetDir, false) {
		return readFTSJSONLQueryData(ctx, config.DatasetDir, limit)
	}
	qrels, err := readFTSSourceQrels(ctx, config)
	if err != nil {
		return queryData{}, err
	}
	data := queryData{}
	err = iterateFTSSourceQueries(ctx, config, func(row ftsQueryRow) error {
		if limit > 0 && len(data.IDs) >= limit {
			return errStopFTSIteration
		}
		groundTruth := qrels[row.ID]
		if row.ID == "" || row.Text == "" || len(groundTruth) == 0 {
			return nil
		}
		data.IDs = append(data.IDs, row.ID)
		data.Texts = append(data.Texts, row.Text)
		data.GroundTruth = append(data.GroundTruth, groundTruth)
		return nil
	})
	if err != nil && !errors.Is(err, errStopFTSIteration) {
		return queryData{}, err
	}
	if len(data.IDs) == 0 {
		return queryData{}, errors.New("FTS dataset has no queries with positive qrels")
	}
	return data, nil
}

func readFTSSourceQrels(ctx context.Context, config benchConfig) (map[string]map[string]int, error) {
	qrels := make(map[string]map[string]int)
	err := withFTSSourceFile(config, ftsQrelsSourceName(config.caseSpec.DatasetName), func(reader io.Reader) error {
		scanner := newFTSScanner(reader)
		first := true
		for scanner.Scan() {
			if err := ctx.Err(); err != nil {
				return err
			}
			fields := strings.Fields(scanner.Text())
			if first && config.caseSpec.DatasetName == "hotpotqa" {
				first = false
				continue
			}
			first = false
			var queryID, documentID, relevanceText string
			if config.caseSpec.DatasetName == "msmarco" && len(fields) == 4 {
				queryID, documentID, relevanceText = fields[0], fields[2], fields[3]
			} else if config.caseSpec.DatasetName == "hotpotqa" && len(fields) == 3 {
				queryID, documentID, relevanceText = fields[0], fields[1], fields[2]
			} else if len(fields) != 0 {
				return fmt.Errorf("invalid FTS qrel row %q", scanner.Text())
			} else {
				continue
			}
			relevance, err := strconv.Atoi(relevanceText)
			if err != nil {
				return fmt.Errorf("parse FTS relevance %q: %w", relevanceText, err)
			}
			if relevance <= 0 {
				continue
			}
			if qrels[queryID] == nil {
				qrels[queryID] = make(map[string]int)
			}
			qrels[queryID][documentID] = max(qrels[queryID][documentID], relevance)
		}
		return scanner.Err()
	})
	return qrels, err
}

func iterateFTSSourceQueries(ctx context.Context, config benchConfig, consume func(ftsQueryRow) error) error {
	return withFTSSourceFile(config, ftsQueriesSourceName(config.caseSpec.DatasetName), func(reader io.Reader) error {
		scanner := newFTSScanner(reader)
		for scanner.Scan() {
			if err := ctx.Err(); err != nil {
				return err
			}
			var row ftsQueryRow
			if config.caseSpec.DatasetName == "msmarco" {
				fields := strings.SplitN(scanner.Text(), "\t", 2)
				if len(fields) != 2 {
					return fmt.Errorf("invalid MS MARCO query row %q", scanner.Text())
				}
				row = ftsQueryRow{ID: fields[0], Text: fields[1]}
			} else {
				var source struct {
					ID   string `json:"_id"`
					Text string `json:"text"`
				}
				if err := json.Unmarshal(scanner.Bytes(), &source); err != nil {
					return fmt.Errorf("decode HotpotQA query: %w", err)
				}
				row = ftsQueryRow{ID: source.ID, Text: source.Text}
			}
			if err := consume(row); err != nil {
				return err
			}
		}
		return scanner.Err()
	})
}

func iterateFTSSourceDocuments(ctx context.Context, config benchConfig, consume func(ftsDocumentRow) error) error {
	return withFTSSourceFile(config, ftsDocumentsSourceName(config.caseSpec.DatasetName), func(reader io.Reader) error {
		scanner := newFTSScanner(reader)
		for scanner.Scan() {
			if err := ctx.Err(); err != nil {
				return err
			}
			var row ftsDocumentRow
			if config.caseSpec.DatasetName == "msmarco" {
				fields := strings.SplitN(scanner.Text(), "\t", 2)
				if len(fields) != 2 {
					return fmt.Errorf("invalid MS MARCO document row")
				}
				row = ftsDocumentRow{ID: fields[0], Text: fixMSMarcoEncoding(fields[1])}
			} else {
				var source struct {
					ID    string `json:"_id"`
					Title string `json:"title"`
					Text  string `json:"text"`
				}
				if err := json.Unmarshal(scanner.Bytes(), &source); err != nil {
					return fmt.Errorf("decode HotpotQA document: %w", err)
				}
				row = ftsDocumentRow{ID: source.ID, Text: strings.TrimSpace(source.Title + " " + source.Text)}
			}
			row.Text = strings.NewReplacer("\t", " ", "\n", " ").Replace(row.Text)
			if err := consume(row); err != nil {
				return err
			}
		}
		return scanner.Err()
	})
}

func newFTSScanner(reader io.Reader) *bufio.Scanner {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64<<10), 16<<20)
	return scanner
}

func withFTSSourceFile(config benchConfig, wanted string, consume func(io.Reader) error) error {
	extractedPath := filepath.Join(config.DatasetDir, filepath.FromSlash(wanted))
	if info, err := os.Stat(extractedPath); err == nil && info.Mode().IsRegular() && info.Size() > 0 {
		file, err := os.Open(extractedPath)
		if err != nil {
			return err
		}
		defer func() { _ = file.Close() }()
		return consume(file)
	}
	archiveName, _, _, err := ftsArchiveSpec(config.caseSpec.DatasetName)
	if err != nil {
		return err
	}
	path := filepath.Join(config.DatasetDir, archiveName)
	if config.caseSpec.DatasetName == "msmarco" {
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer func() { _ = file.Close() }()
		compressed, err := gzip.NewReader(file)
		if err != nil {
			return fmt.Errorf("open MS MARCO archive: %w", err)
		}
		defer func() { _ = compressed.Close() }()
		archive := tar.NewReader(compressed)
		for {
			header, err := archive.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return fmt.Errorf("read MS MARCO archive: %w", err)
			}
			if filepath.ToSlash(header.Name) == wanted {
				return consume(archive)
			}
		}
	} else {
		archive, err := zip.OpenReader(path)
		if err != nil {
			return fmt.Errorf("open HotpotQA archive: %w", err)
		}
		defer func() { _ = archive.Close() }()
		for _, file := range archive.File {
			if filepath.ToSlash(file.Name) != wanted {
				continue
			}
			reader, err := file.Open()
			if err != nil {
				return err
			}
			consumeErr := consume(reader)
			return errors.Join(consumeErr, reader.Close())
		}
	}
	return fmt.Errorf("FTS source file %s is missing from %s", wanted, path)
}

func ftsDocumentsSourceName(dataset string) string {
	if dataset == "msmarco" {
		return "collection.tsv"
	}
	return "hotpotqa/corpus.jsonl"
}

func ftsQueriesSourceName(dataset string) string {
	if dataset == "msmarco" {
		return "queries.dev.small.tsv"
	}
	return "hotpotqa/queries.jsonl"
}

func ftsQrelsSourceName(dataset string) string {
	if dataset == "msmarco" {
		return "qrels.dev.small.tsv"
	}
	return "hotpotqa/qrels/test.tsv"
}

func ftsFilterID(ordinal, size int64) int64 {
	if size <= 1 {
		return 0
	}
	high, _ := bits.Mul64(uint64(size), 0x9E3779B97F4A7C15)
	multiplier := int64(max(uint64(1), high))
	for gcd64(multiplier, size) != 1 {
		multiplier++
		if multiplier >= size {
			multiplier = 1
		}
	}
	return (multiplier*ordinal + int64(uint64(0xD1B54A32D192ED03)%uint64(size))) % size
}

func gcd64(left, right int64) int64 {
	for right != 0 {
		left, right = right, left%right
	}
	return left
}

func fixMSMarcoEncoding(value string) string {
	runes := []rune(value)
	for _, width := range []int{4, 3, 2} {
		for position := 0; position+width <= len(runes); position++ {
			candidate := runes[position : position+width]
			encoded := make([]byte, len(candidate))
			suspicious, latin1 := false, true
			for index, character := range candidate {
				if character > 255 {
					latin1 = false
					break
				}
				encoded[index] = byte(character)
				suspicious = suspicious || character >= 128
			}
			if !latin1 || !suspicious || !utf8.Valid(encoded) || utf8.RuneCount(encoded) != 1 {
				continue
			}
			decoded, _ := utf8.DecodeRune(encoded)
			runes = append(append(append([]rune{}, runes[:position]...), decoded), runes[position+width:]...)
		}
	}
	return string(runes)
}
