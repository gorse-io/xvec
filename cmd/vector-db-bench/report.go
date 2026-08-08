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
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

const reportSchemaVersion = "vector-db-bench/v1"

type reportConfig struct {
	Path                string `json:"path"`
	DBLabel             string `json:"db_label"`
	M                   int    `json:"m"`
	EFConstruction      int    `json:"ef_construction"`
	EFSearch            int    `json:"ef_search"`
	QuantizeType        string `json:"quantize_type"`
	UseRefiner          bool   `json:"use_refiner"`
	K                   int    `json:"k"`
	BatchSize           int    `json:"batch_size"`
	LoadLimit           int64  `json:"load_limit,omitempty"`
	QueryLimit          int    `json:"query_limit,omitempty"`
	ConcurrencyDuration string `json:"concurrency_duration"`
	SerialCooldown      string `json:"serial_cooldown"`
	NumConcurrency      []int  `json:"num_concurrency"`
	MaxDocsPerSegment   uint64 `json:"max_docs_per_segment"`
	EnableMmap          bool   `json:"enable_mmap"`
}

type loadMetrics struct {
	Rows                int64   `json:"rows"`
	InsertDurationSec   float64 `json:"insert_duration_sec"`
	OptimizeDurationSec float64 `json:"optimize_duration_sec"`
	LoadDurationSec     float64 `json:"load_duration_sec"`
	RowsPerSecond       float64 `json:"rows_per_second"`
	ImmutableSegments   uint64  `json:"immutable_segments"`
	StorageBytes        uint64  `json:"storage_bytes"`
}

type searchMetrics struct {
	Queries      int     `json:"queries"`
	QPS          float64 `json:"qps"`
	Recall       float64 `json:"recall,omitempty"`
	LatencyAvgMS float64 `json:"latency_avg_ms"`
	LatencyP95MS float64 `json:"latency_p95_ms"`
	LatencyP99MS float64 `json:"latency_p99_ms"`
}

type concurrentMetrics struct {
	Concurrency int `json:"concurrency"`
	searchMetrics
}

type benchmarkReport struct {
	SchemaVersion string               `json:"schema_version"`
	Tool          string               `json:"tool"`
	Timestamp     time.Time            `json:"timestamp"`
	Case          benchmarkCase        `json:"case"`
	DatasetDir    string               `json:"dataset_dir"`
	Config        reportConfig         `json:"config"`
	Note          string               `json:"note,omitempty"`
	System        systemInfo           `json:"system"`
	Load          *loadMetrics         `json:"load,omitempty"`
	Serial        *searchMetrics       `json:"serial,omitempty"`
	Concurrent    []concurrentMetrics  `json:"concurrent,omitempty"`
	VectorDBBench *vectorDBBenchMetric `json:"vectordbbench_metrics,omitempty"`
}

// vectorDBBenchMetric mirrors the names used by VectorDBBench's Metric model
// so result consumers can ingest the core zvec performance fields directly.
type vectorDBBenchMetric struct {
	InsertedCount         int64     `json:"inserted_count"`
	InsertDuration        float64   `json:"insert_duration"`
	OptimizeDuration      float64   `json:"optimize_duration"`
	LoadDuration          float64   `json:"load_duration"`
	QPS                   float64   `json:"qps"`
	Recall                float64   `json:"recall"`
	SerialLatencyP99      float64   `json:"serial_latency_p99"`
	SerialLatencyP95      float64   `json:"serial_latency_p95"`
	Concurrency           []int     `json:"conc_num_list"`
	ConcurrentQPS         []float64 `json:"conc_qps_list"`
	ConcurrentLatencyP99  []float64 `json:"conc_latency_p99_list"`
	ConcurrentLatencyP95  []float64 `json:"conc_latency_p95_list"`
	ConcurrentLatencyMean []float64 `json:"conc_latency_avg_list"`
}

type systemInfo struct {
	GOOS     string `json:"goos"`
	GOARCH   string `json:"goarch"`
	Go       string `json:"go_version"`
	NumCPU   int    `json:"num_cpu"`
	Compiler string `json:"compiler"`
}

func newBenchmarkReport(config benchConfig) benchmarkReport {
	quantize := config.Quantize
	if quantize == "" {
		quantize = "none"
	}
	return benchmarkReport{
		SchemaVersion: reportSchemaVersion,
		Tool:          "zvec-go/cmd/vector-db-bench",
		Timestamp:     time.Now().UTC(),
		Case:          config.caseSpec,
		DatasetDir:    config.DatasetDir,
		Note:          config.Note,
		Config: reportConfig{
			Path: config.Path, DBLabel: config.DBLabel,
			M: config.M, EFConstruction: config.EFConstruction, EFSearch: config.EFSearch,
			QuantizeType: quantize, UseRefiner: config.UseRefiner, K: config.K,
			BatchSize: config.BatchSize, LoadLimit: config.LoadLimit, QueryLimit: config.QueryLimit,
			ConcurrencyDuration: config.ConcurrencyDuration.String(), SerialCooldown: config.SerialCooldown.String(),
			NumConcurrency:    config.concurrency,
			MaxDocsPerSegment: config.MaxDocsPerSegment, EnableMmap: config.EnableMmap,
		},
		System: systemInfo{
			GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, Go: runtime.Version(),
			NumCPU: runtime.NumCPU(), Compiler: runtime.Compiler,
		},
	}
}

func (r *benchmarkReport) populateVectorDBBenchMetric() {
	metric := &vectorDBBenchMetric{}
	if r.Load != nil {
		metric.InsertedCount = r.Load.Rows
		metric.InsertDuration = r.Load.InsertDurationSec
		metric.OptimizeDuration = r.Load.OptimizeDurationSec
		metric.LoadDuration = r.Load.LoadDurationSec
	}
	if r.Serial != nil {
		metric.QPS = r.Serial.QPS
		metric.Recall = r.Serial.Recall
		metric.SerialLatencyP99 = r.Serial.LatencyP99MS / 1000
		metric.SerialLatencyP95 = r.Serial.LatencyP95MS / 1000
	}
	for _, result := range r.Concurrent {
		metric.Concurrency = append(metric.Concurrency, result.Concurrency)
		metric.ConcurrentQPS = append(metric.ConcurrentQPS, result.QPS)
		metric.ConcurrentLatencyP99 = append(metric.ConcurrentLatencyP99, result.LatencyP99MS/1000)
		metric.ConcurrentLatencyP95 = append(metric.ConcurrentLatencyP95, result.LatencyP95MS/1000)
		metric.ConcurrentLatencyMean = append(metric.ConcurrentLatencyMean, result.LatencyAvgMS/1000)
		if result.QPS > metric.QPS {
			metric.QPS = result.QPS
		}
	}
	r.VectorDBBench = metric
}

func writeReport(report benchmarkReport, output string, stdout io.Writer) error {
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("encode benchmark report: %w", err)
	}
	encoded = append(encoded, '\n')
	if output == "" || output == "-" {
		_, err = stdout.Write(encoded)
		return err
	}
	absolute, err := filepath.Abs(output)
	if err != nil {
		return fmt.Errorf("resolve output path: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(absolute), 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	temporary := absolute + ".tmp"
	if err := os.WriteFile(temporary, encoded, 0o644); err != nil {
		return fmt.Errorf("write temporary report: %w", err)
	}
	if err := os.Rename(temporary, absolute); err != nil {
		_ = os.Remove(temporary)
		return fmt.Errorf("publish report: %w", err)
	}
	return nil
}

func printSummary(report benchmarkReport, writer io.Writer) {
	fmt.Fprintf(writer, "case=%s dataset=%s\n", report.Case.Name, report.DatasetDir)
	if report.Load != nil {
		fmt.Fprintf(writer, "load rows=%d duration=%.3fs insert=%.3fs optimize=%.3fs rows/s=%.1f\n",
			report.Load.Rows, report.Load.LoadDurationSec, report.Load.InsertDurationSec,
			report.Load.OptimizeDurationSec, report.Load.RowsPerSecond)
	}
	if report.Serial != nil {
		fmt.Fprintf(writer, "serial queries=%d qps=%.2f recall=%.4f avg=%.3fms p95=%.3fms p99=%.3fms\n",
			report.Serial.Queries, report.Serial.QPS, report.Serial.Recall,
			report.Serial.LatencyAvgMS, report.Serial.LatencyP95MS, report.Serial.LatencyP99MS)
	}
	for _, result := range report.Concurrent {
		fmt.Fprintf(writer, "concurrent workers=%d queries=%d qps=%.2f avg=%.3fms p95=%.3fms p99=%.3fms\n",
			result.Concurrency, result.Queries, result.QPS,
			result.LatencyAvgMS, result.LatencyP95MS, result.LatencyP99MS)
	}
}
