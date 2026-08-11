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

package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	casePerformance768D1M  = "Performance768D1M"
	casePerformance768D10M = "Performance768D10M"
	caseCustom             = "Custom"
)

type benchmarkCase struct {
	Name          string   `json:"name"`
	DatasetName   string   `json:"dataset_name"`
	DatasetFolder string   `json:"dataset_folder"`
	Size          int64    `json:"size"`
	Dimension     int      `json:"dimension"`
	Metric        string   `json:"metric"`
	TrainFiles    []string `json:"train_files"`
}

type benchConfig struct {
	Path                 string
	CaseType             string
	DatasetDir           string
	DatasetBaseURL       string
	Dimension            int
	Metric               string
	TrainFiles           string
	K                    int
	BatchSize            int
	LoadLimit            int64
	QueryLimit           int
	M                    int
	EFConstruction       int
	EFSearch             int
	Quantize             string
	UseRefiner           bool
	NumConcurrency       string
	ConcurrencyDuration  time.Duration
	SerialCooldown       time.Duration
	WarmupQueries        int
	OptimizeConcurrency  int
	MaxDocsPerSegment    uint64
	MaxBufferSize        uint
	EnableMmap           bool
	SkipDownload         bool
	SkipDropOld          bool
	SkipLoad             bool
	SkipSerialSearch     bool
	SkipConcurrentSearch bool
	DryRun               bool
	Output               string
	DBLabel              string
	Note                 string
	OperationTimeout     time.Duration
	Seed                 int64
	caseSpec             benchmarkCase
	concurrency          []int
}

func parseConfig(args []string, stderr io.Writer) (benchConfig, error) {
	var config benchConfig
	flags := flag.NewFlagSet("vector-db-bench", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&config.Path, "path", "", "xvec collection path (required)")
	flags.StringVar(&config.CaseType, "case-type", casePerformance768D1M, "benchmark case: Performance768D1M, Performance768D10M, or Custom")
	flags.StringVar(&config.DatasetDir, "dataset-dir", "", "local VectorDBBench dataset directory")
	flags.StringVar(&config.DatasetBaseURL, "dataset-base-url", "https://assets.zilliz.com/benchmark", "VectorDBBench dataset base URL")
	flags.IntVar(&config.Dimension, "dimension", 0, "vector dimension for Custom case")
	flags.StringVar(&config.Metric, "metric", "cosine", "vector metric for Custom case: cosine, l2, or ip")
	flags.StringVar(&config.TrainFiles, "train-files", "", "comma-separated Parquet train files for Custom case")
	flags.IntVar(&config.K, "k", 100, "number of nearest neighbors")
	flags.IntVar(&config.BatchSize, "batch-size", 100, "documents per insert batch")
	flags.Int64Var(&config.LoadLimit, "load-limit", 0, "maximum training rows to load; zero means all")
	flags.IntVar(&config.QueryLimit, "query-limit", 0, "maximum test queries to use; zero means all")
	flags.IntVar(&config.M, "m", 50, "HNSW M")
	flags.IntVar(&config.EFConstruction, "ef-construction", 500, "HNSW construction EF")
	flags.IntVar(&config.EFSearch, "ef-search", 300, "HNSW search EF")
	flags.StringVar(&config.Quantize, "quantize-type", "", "HNSW quantization: fp16, int8, int4, or empty")
	flags.BoolVar(&config.UseRefiner, "is-using-refiner", false, "refine quantized candidates with original vectors")
	flags.StringVar(&config.NumConcurrency, "num-concurrency", "12,14,16,18,20", "comma-separated concurrent search worker counts")
	concurrencyDuration := flags.String("concurrency-duration", "30s", "duration per concurrency level, for example 30s or 30")
	serialCooldown := flags.String("serial-cooldown", "0", "cooldown between concurrent and serial search, for example 3s or 3")
	flags.IntVar(&config.WarmupQueries, "warmup-queries", 100, "queries executed before measured search")
	flags.IntVar(&config.OptimizeConcurrency, "optimize-concurrency", 0, "xvec optimize workers; zero uses runtime default")
	flags.Uint64Var(&config.MaxDocsPerSegment, "max-docs-per-segment", 10_000_000, "maximum documents per immutable segment")
	flags.UintVar(&config.MaxBufferSize, "max-buffer-size", 64<<20, "maximum native index buffer bytes")
	flags.BoolVar(&config.EnableMmap, "enable-mmap", true, "enable mmap-backed index access")
	flags.BoolVar(&config.SkipDownload, "skip-download", false, "require dataset files to exist locally")
	flags.BoolVar(&config.SkipDropOld, "skip-drop-old", false, "keep an existing collection")
	flags.BoolVar(&config.SkipLoad, "skip-load", false, "reuse an already loaded collection")
	flags.BoolVar(&config.SkipSerialSearch, "skip-search-serial", false, "skip serial recall and latency search")
	flags.BoolVar(&config.SkipConcurrentSearch, "skip-search-concurrent", false, "skip sustained concurrent search")
	flags.BoolVar(&config.DryRun, "dry-run", false, "validate and print configuration without downloading or running")
	flags.StringVar(&config.Output, "output", "", "write result JSON to this file; empty writes JSON to stdout")
	flags.StringVar(&config.DBLabel, "db-label", "xvec-go", "label stored in the result")
	flags.StringVar(&config.Note, "note", "", "non-sensitive run context stored in the result")
	operationTimeout := flags.String("operation-timeout", "0", "whole-run timeout; zero disables it")
	flags.Int64Var(&config.Seed, "seed", 0, "deterministic concurrent-query seed")
	if err := flags.Parse(args); err != nil {
		return benchConfig{}, err
	}
	if flags.NArg() != 0 {
		return benchConfig{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}
	var err error
	config.ConcurrencyDuration, err = parseFlexibleDuration(*concurrencyDuration)
	if err != nil {
		return benchConfig{}, fmt.Errorf("concurrency-duration: %w", err)
	}
	config.SerialCooldown, err = parseFlexibleDuration(*serialCooldown)
	if err != nil {
		return benchConfig{}, fmt.Errorf("serial-cooldown: %w", err)
	}
	config.OperationTimeout, err = parseFlexibleDuration(*operationTimeout)
	if err != nil {
		return benchConfig{}, fmt.Errorf("operation-timeout: %w", err)
	}
	config.concurrency, err = parseConcurrency(config.NumConcurrency)
	if err != nil {
		return benchConfig{}, err
	}
	config.caseSpec, err = resolveBenchmarkCase(config)
	if err != nil {
		return benchConfig{}, err
	}
	if config.DatasetDir == "" {
		config.DatasetDir = filepath.Join("/tmp/vectordb_bench/dataset", config.caseSpec.DatasetName, config.caseSpec.DatasetFolder)
	}
	if err := config.validate(); err != nil {
		return benchConfig{}, err
	}
	return config, nil
}

func (c benchConfig) validate() error {
	if strings.TrimSpace(c.Path) == "" {
		return errors.New("path is required")
	}
	if c.K <= 0 {
		return errors.New("k must be positive")
	}
	if c.BatchSize <= 0 {
		return errors.New("batch-size must be positive")
	}
	if c.SkipLoad && !c.SkipDropOld {
		return errors.New("skip-load requires skip-drop-old to preserve the existing collection")
	}
	if c.LoadLimit < 0 || c.QueryLimit < 0 {
		return errors.New("load-limit and query-limit cannot be negative")
	}
	if c.M <= 0 || c.EFConstruction < c.M || c.EFSearch <= 0 {
		return errors.New("HNSW parameters require m > 0, ef-construction >= m, and ef-search > 0")
	}
	if c.OptimizeConcurrency < 0 {
		return errors.New("optimize-concurrency cannot be negative")
	}
	if c.MaxDocsPerSegment < 1_000 {
		return errors.New("max-docs-per-segment must be at least 1000")
	}
	if c.MaxBufferSize == 0 || uint64(c.MaxBufferSize) > uint64(^uint32(0)) {
		return errors.New("max-buffer-size must fit a positive uint32")
	}
	if c.ConcurrencyDuration <= 0 && !c.SkipConcurrentSearch {
		return errors.New("concurrency-duration must be positive")
	}
	if c.SerialCooldown < 0 {
		return errors.New("serial-cooldown cannot be negative")
	}
	if c.WarmupQueries < 0 {
		return errors.New("warmup-queries cannot be negative")
	}
	if c.OperationTimeout < 0 {
		return errors.New("operation-timeout cannot be negative")
	}
	switch strings.ToLower(c.Quantize) {
	case "", "none", "fp16", "int8", "int4":
	default:
		return fmt.Errorf("unsupported quantize-type %q", c.Quantize)
	}
	if _, err := url.ParseRequestURI(strings.TrimRight(c.DatasetBaseURL, "/")); err != nil {
		return fmt.Errorf("invalid dataset-base-url: %w", err)
	}
	return nil
}

func resolveBenchmarkCase(config benchConfig) (benchmarkCase, error) {
	switch strings.ToLower(config.CaseType) {
	case strings.ToLower(casePerformance768D1M):
		return benchmarkCase{
			Name: casePerformance768D1M, DatasetName: "cohere", DatasetFolder: "cohere_medium_1m",
			Size: 1_000_000, Dimension: 768, Metric: "cosine", TrainFiles: []string{"shuffle_train.parquet"},
		}, nil
	case strings.ToLower(casePerformance768D10M):
		files := make([]string, 10)
		for index := range files {
			files[index] = fmt.Sprintf("shuffle_train-%02d-of-10.parquet", index)
		}
		return benchmarkCase{
			Name: casePerformance768D10M, DatasetName: "cohere", DatasetFolder: "cohere_large_10m",
			Size: 10_000_000, Dimension: 768, Metric: "cosine", TrainFiles: files,
		}, nil
	case strings.ToLower(caseCustom):
		files := splitNonEmpty(config.TrainFiles)
		if config.Dimension <= 0 || len(files) == 0 || strings.TrimSpace(config.DatasetDir) == "" {
			return benchmarkCase{}, errors.New("Custom case requires dataset-dir, positive dimension, and train-files")
		}
		metric := strings.ToLower(config.Metric)
		if metric != "cosine" && metric != "l2" && metric != "ip" {
			return benchmarkCase{}, fmt.Errorf("unsupported Custom metric %q", config.Metric)
		}
		return benchmarkCase{
			Name: caseCustom, DatasetName: "custom", DatasetFolder: filepath.Base(config.DatasetDir),
			Dimension: config.Dimension, Metric: metric, TrainFiles: files,
		}, nil
	default:
		return benchmarkCase{}, fmt.Errorf("unsupported case-type %q", config.CaseType)
	}
}

func parseFlexibleDuration(value string) (time.Duration, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, errors.New("duration is empty")
	}
	if seconds, err := strconv.ParseFloat(value, 64); err == nil {
		return time.Duration(seconds * float64(time.Second)), nil
	}
	return time.ParseDuration(value)
}

func parseConcurrency(value string) ([]int, error) {
	parts := splitNonEmpty(value)
	if len(parts) == 0 {
		return nil, errors.New("num-concurrency is empty")
	}
	values := make([]int, 0, len(parts))
	seen := make(map[int]struct{}, len(parts))
	for _, part := range parts {
		value, err := strconv.Atoi(part)
		if err != nil || value <= 0 {
			return nil, fmt.Errorf("invalid concurrency %q", part)
		}
		if _, exists := seen[value]; exists {
			return nil, fmt.Errorf("duplicate concurrency %d", value)
		}
		seen[value] = struct{}{}
		values = append(values, value)
	}
	return values, nil
}

func splitNonEmpty(value string) []string {
	parts := strings.Split(value, ",")
	output := make([]string, 0, len(parts))
	for _, part := range parts {
		if part = strings.TrimSpace(part); part != "" {
			output = append(output, part)
		}
	}
	return output
}
