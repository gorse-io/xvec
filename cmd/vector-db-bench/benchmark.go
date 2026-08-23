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
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorse-io/xvec"
)

func runBenchmark(ctx context.Context, config benchConfig, log io.Writer) (benchmarkReport, error) {
	report := newBenchmarkReport(config)
	shutdown, err := initializeBenchmarkBackend(config)
	if err != nil {
		return report, err
	}
	defer shutdown()
	needsSearch := !config.SkipSerialSearch || !config.SkipConcurrentSearch
	if !config.SkipLoad || needsSearch {
		if err := prepareDataset(ctx, config, !config.SkipLoad, log); err != nil {
			return report, err
		}
	}
	if !config.SkipLoad {
		metrics, err := loadBenchmarkDataset(ctx, config, log)
		if err != nil {
			return report, err
		}
		report.Load = &metrics
	}
	if !needsSearch {
		report.populateVectorDBBenchMetric()
		return report, nil
	}

	var data queryData
	if config.caseSpec.Workload == workloadFullText {
		data, err = readFTSQueryData(ctx, config, config.QueryLimit)
	} else {
		data, err = readQueryData(ctx, config.DatasetDir, config.caseSpec.Dimension, config.QueryLimit)
	}
	if err != nil {
		return report, err
	}
	engine, closer, err := openBenchmarkQueryEngine(ctx, config)
	if err != nil {
		return report, err
	}
	defer func() { _ = closer.Close() }()
	if err := warmupSearch(ctx, engine, data, config.WarmupQueries); err != nil {
		return report, err
	}
	if !config.SkipConcurrentSearch {
		for _, concurrency := range config.concurrency {
			metrics, err := runConcurrentSearch(ctx, engine, data, concurrency, config.ConcurrencyDuration, config.Seed)
			if err != nil {
				return report, err
			}
			report.Concurrent = append(report.Concurrent, concurrentMetrics{Concurrency: concurrency, searchMetrics: metrics})
		}
	}
	if !config.SkipSerialSearch {
		if !config.SkipConcurrentSearch && config.SerialCooldown > 0 {
			_, _ = fmt.Fprintf(log, "cooling down for %s before serial search\n", config.SerialCooldown)
			timer := time.NewTimer(config.SerialCooldown)
			select {
			case <-ctx.Done():
				timer.Stop()
				return report, ctx.Err()
			case <-timer.C:
			}
		}
		metrics, err := runSerialSearch(ctx, engine, data, config.K, config.caseSpec.Workload == workloadFullText)
		if err != nil {
			return report, err
		}
		report.Serial = &metrics
	}
	report.populateVectorDBBenchMetric()
	return report, nil
}

func loadXvecDataset(ctx context.Context, config benchConfig, log io.Writer) (loadMetrics, error) {
	collection, err := writableBenchmarkCollection(ctx, config)
	if err != nil {
		return loadMetrics{}, err
	}
	closed := false
	defer func() {
		if !closed {
			_ = collection.Close()
		}
	}()

	loadStarted := time.Now()
	insertStarted := loadStarted
	inserted := int64(0)
	nextProgress := int64(100_000)
	var rows int64
	if config.caseSpec.Workload == workloadFullText {
		rows, err = forEachFTSDocumentBatch(ctx, config, config.BatchSize, config.LoadLimit, func(rows []ftsDocumentRow) error {
			documents := make([]xvec.Document, len(rows))
			for index, row := range rows {
				documents[index] = xvec.Document{PrimaryKey: row.ID, Fields: map[string]any{
					"text": row.Text, "filter_id": row.FilterID,
				}}
			}
			results, insertErr := collection.Insert(ctx, documents)
			if insertErr != nil {
				return fmt.Errorf("insert FTS batch at row %d: %w", inserted, insertErr)
			}
			for _, result := range results {
				if result.Err != nil {
					return fmt.Errorf("insert document %s: %w", result.PrimaryKey, result.Err)
				}
			}
			inserted += int64(len(documents))
			if inserted >= nextProgress {
				_, _ = fmt.Fprintf(log, "inserted %d documents (%.1f rows/s)\n", inserted, float64(inserted)/time.Since(insertStarted).Seconds())
				nextProgress = (inserted/100_000 + 1) * 100_000
			}
			return nil
		})
	} else {
		rows, err = forEachTrainingBatch(
			ctx, config.DatasetDir, config.caseSpec.TrainFiles, config.BatchSize, config.LoadLimit,
			func(rows []vectorParquetRow) error {
				documents := make([]xvec.Document, len(rows))
				for index, row := range rows {
					if len(row.Embedding) != config.caseSpec.Dimension {
						return fmt.Errorf("training vector %d has dimension %d, want %d", row.ID, len(row.Embedding), config.caseSpec.Dimension)
					}
					documents[index] = xvec.Document{
						PrimaryKey: strconv.FormatInt(row.ID, 10),
						Fields: map[string]any{
							"id": row.ID, "dense": xvec.VectorFP32(row.Embedding),
						},
					}
				}
				results, err := collection.Insert(ctx, documents)
				if err != nil {
					return fmt.Errorf("insert batch at row %d: %w", inserted, err)
				}
				for _, result := range results {
					if result.Err != nil {
						return fmt.Errorf("insert document %s: %w", result.PrimaryKey, result.Err)
					}
				}
				inserted += int64(len(documents))
				if inserted >= nextProgress {
					_, _ = fmt.Fprintf(log, "inserted %d vectors (%.1f rows/s)\n", inserted, float64(inserted)/time.Since(insertStarted).Seconds())
					nextProgress = (inserted/100_000 + 1) * 100_000
				}
				return nil
			},
		)
	}
	if err != nil {
		return loadMetrics{}, err
	}
	insertDuration := time.Since(insertStarted)
	optimizeStarted := time.Now()
	if err := collection.Optimize(ctx, xvec.OptimizeOptions{Concurrency: config.OptimizeConcurrency}); err != nil {
		return loadMetrics{}, fmt.Errorf("optimize benchmark collection: %w", err)
	}
	optimizeDuration := time.Since(optimizeStarted)
	stats := collection.Stats()
	if err := collection.Close(); err != nil {
		return loadMetrics{}, fmt.Errorf("close loaded collection: %w", err)
	}
	closed = true
	metrics := loadMetrics{
		Rows: rows, InsertDurationSec: insertDuration.Seconds(), OptimizeDurationSec: optimizeDuration.Seconds(),
		LoadDurationSec: time.Since(loadStarted).Seconds(), ImmutableSegments: stats.ImmutableSegments,
		StorageBytes: stats.StorageMemoryBytes,
	}
	if insertDuration > 0 {
		metrics.RowsPerSecond = float64(rows) / insertDuration.Seconds()
	}
	return metrics, nil
}

func writableBenchmarkCollection(ctx context.Context, config benchConfig) (*xvec.Collection, error) {
	options := xvec.CollectionOptions{
		EnableMmap: config.EnableMmap, MaxBufferSize: uint32(config.MaxBufferSize),
	}
	if config.SkipDropOld {
		collection, err := xvec.Open(ctx, config.Path, options)
		if err != nil {
			return nil, fmt.Errorf("open existing benchmark collection: %w", err)
		}
		return collection, nil
	}
	if _, err := os.Stat(config.Path); err == nil {
		old, err := xvec.Open(ctx, config.Path, options)
		if err != nil {
			return nil, fmt.Errorf("open old benchmark collection before destroy: %w", err)
		}
		if err := old.Destroy(ctx); err != nil {
			return nil, fmt.Errorf("destroy old benchmark collection: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("stat benchmark collection: %w", err)
	}
	if config.caseSpec.Workload == workloadFullText {
		params := xvec.FTSIndexParams{
			Tokenizer: strings.ToLower(config.FTSTokenizer), Filters: splitNonEmpty(strings.ToLower(config.FTSTokenFilters)),
			ExtraParams: config.FTSExtraParams,
		}
		schema := xvec.NewCollectionSchema("fts_bench_test",
			xvec.FieldSchema{Name: "text", DataType: xvec.DataTypeString, Index: params},
			xvec.FieldSchema{Name: "filter_id", DataType: xvec.DataTypeInt64, Index: xvec.NewInvertIndexParams()},
		)
		schema.MaxDocsPerSegment = config.MaxDocsPerSegment
		collection, err := xvec.CreateAndOpen(ctx, config.Path, schema, options)
		if err != nil {
			return nil, fmt.Errorf("create FTS benchmark collection: %w", err)
		}
		return collection, nil
	}
	metric, err := parseMetric(config.caseSpec.Metric)
	if err != nil {
		return nil, err
	}
	quantize, err := parseQuantization(config.Quantize)
	if err != nil {
		return nil, err
	}
	var index xvec.IndexParams
	switch strings.ToLower(config.IndexType) {
	case indexHNSW:
		params := xvec.NewHNSWIndexParams(metric)
		params.M = config.M
		params.EFConstruction = config.EFConstruction
		params.Quantize = quantize
		params.Quantizer.EnableRotate = quantize == xvec.QuantizeTypeInt8 || quantize == xvec.QuantizeTypeInt4
		index = params
	case indexDiskANN:
		params := xvec.NewDiskANNIndexParams(metric)
		params.MaxDegree = config.DiskANNMaxDegree
		params.ListSize = config.DiskANNBuildList
		params.PQChunks = config.DiskANNPQChunks
		params.Quantize = quantize
		params.Quantizer.EnableRotate = quantize == xvec.QuantizeTypeInt8 || quantize == xvec.QuantizeTypeInt4
		index = params
	default:
		return nil, fmt.Errorf("unsupported index type %q", config.IndexType)
	}
	schema := xvec.NewCollectionSchema("vector_bench_test",
		xvec.FieldSchema{Name: "id", DataType: xvec.DataTypeInt64, Index: xvec.NewInvertIndexParams()},
		xvec.FieldSchema{
			Name: "dense", DataType: xvec.DataTypeVectorFP32, Dimension: uint32(config.caseSpec.Dimension), Index: index,
		},
	)
	schema.MaxDocsPerSegment = config.MaxDocsPerSegment
	collection, err := xvec.CreateAndOpen(ctx, config.Path, schema, options)
	if err != nil {
		return nil, fmt.Errorf("create benchmark collection: %w", err)
	}
	return collection, nil
}

func parseMetric(value string) (xvec.MetricType, error) {
	switch strings.ToLower(value) {
	case "cosine":
		return xvec.MetricTypeCosine, nil
	case "l2":
		return xvec.MetricTypeL2, nil
	case "ip":
		return xvec.MetricTypeIP, nil
	default:
		return 0, fmt.Errorf("unsupported metric %q", value)
	}
}

func parseQuantization(value string) (xvec.QuantizeType, error) {
	switch strings.ToLower(value) {
	case "", "none":
		return xvec.QuantizeTypeUndefined, nil
	case "fp16":
		return xvec.QuantizeTypeFP16, nil
	case "int8":
		return xvec.QuantizeTypeInt8, nil
	case "int4":
		return xvec.QuantizeTypeInt4, nil
	default:
		return 0, fmt.Errorf("unsupported quantization %q", value)
	}
}

type benchmarkQueryEngine interface {
	search(context.Context, benchmarkQuery) ([]string, error)
}

type benchmarkQuery struct {
	Vector []float32
	Text   string
}

type xvecQueryEngine struct {
	collection *xvec.Collection
	params     xvec.QueryParams
	k          int
	fullText   bool
	returnText bool
	ftsParams  xvec.FTSQueryParams
}

func newXvecQueryEngine(collection *xvec.Collection, config benchConfig) xvecQueryEngine {
	var params xvec.QueryParams
	if strings.EqualFold(config.IndexType, indexDiskANN) {
		diskANN := xvec.NewDiskANNQueryParams()
		diskANN.ListSize = config.DiskANNQueryList
		diskANN.UseRefiner = config.UseRefiner
		params = diskANN
	} else {
		hnsw := xvec.NewHNSWQueryParams()
		hnsw.EF = config.EFSearch
		hnsw.UseRefiner = config.UseRefiner
		params = hnsw
	}
	ftsParams := xvec.NewFTSQueryParams()
	ftsParams.DefaultOperator = strings.ToUpper(config.FTSDefaultOperator)
	return xvecQueryEngine{
		collection: collection, params: params, k: config.K,
		fullText:   config.caseSpec.Workload == workloadFullText,
		returnText: config.caseSpec.Workload == workloadFullText && strings.EqualFold(config.PayloadProfile, "text"), ftsParams: ftsParams,
	}
}

func (e xvecQueryEngine) search(ctx context.Context, query benchmarkQuery) ([]string, error) {
	projection := xvec.Projection{OutputFields: []string{}}
	if e.returnText {
		projection.OutputFields = []string{"text"}
	}
	request := xvec.VectorQuery{
		Field: "dense", DenseVector: xvec.VectorFP32(query.Vector), TopK: e.k, Params: e.params, Projection: projection,
	}
	if e.fullText {
		request.Field = "text"
		request.DenseVector = nil
		request.FTS = &xvec.FTSClause{Match: query.Text}
		request.Params = e.ftsParams
	}
	results, err := e.collection.Query(ctx, request)
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(results))
	for index, result := range results {
		ids[index] = result.PrimaryKey
	}
	return ids, nil
}

func warmupSearch(ctx context.Context, engine benchmarkQueryEngine, data queryData, count int) error {
	if count > len(data.IDs) {
		count = len(data.IDs)
	}
	for index := 0; index < count; index++ {
		if _, err := engine.search(ctx, data.query(index)); err != nil {
			return fmt.Errorf("warmup query %s: %w", data.IDs[index], err)
		}
	}
	return nil
}

func runSerialSearch(ctx context.Context, engine benchmarkQueryEngine, data queryData, k int, fullText bool) (searchMetrics, error) {
	latencies := make([]float64, len(data.IDs))
	var recall, mrr, ndcg float64
	started := time.Now()
	for index := range data.IDs {
		queryStarted := time.Now()
		ids, err := engine.search(ctx, data.query(index))
		if err != nil {
			return searchMetrics{}, fmt.Errorf("serial query %s: %w", data.IDs[index], err)
		}
		latencies[index] = float64(time.Since(queryStarted)) / float64(time.Millisecond)
		if fullText {
			recall += recallFTSAtK(k, data.GroundTruth[index], ids)
			mrr += mrrFTSAtK(k, data.GroundTruth[index], ids)
			ndcg += ndcgFTSAtK(k, data.GroundTruth[index], ids)
		} else {
			recall += recallAtK(k, data.GroundTruth[index], ids)
		}
	}
	elapsed := time.Since(started)
	metrics := summarizeSearch(latencies, elapsed)
	metrics.Recall = recall / float64(len(data.IDs))
	if fullText {
		metrics.MRR = mrr / float64(len(data.IDs))
		metrics.NDCG = ndcg / float64(len(data.IDs))
	}
	return metrics, nil
}

type concurrentWorkerResult struct {
	latencies []float64
	err       error
}

func runConcurrentSearch(
	ctx context.Context,
	engine benchmarkQueryEngine,
	data queryData,
	concurrency int,
	duration time.Duration,
	seed int64,
) (searchMetrics, error) {
	runContext, cancel := context.WithTimeout(ctx, duration)
	defer cancel()
	results := make(chan concurrentWorkerResult, concurrency)
	started := time.Now()
	for worker := 0; worker < concurrency; worker++ {
		go func(worker int) {
			random := rand.New(rand.NewSource(seed + int64(worker)))
			latencies := make([]float64, 0, 1024)
			for runContext.Err() == nil {
				index := random.Intn(len(data.IDs))
				queryStarted := time.Now()
				_, err := engine.search(runContext, data.query(index))
				if err != nil {
					if runContext.Err() != nil {
						break
					}
					results <- concurrentWorkerResult{err: fmt.Errorf("worker %d query %s: %w", worker, data.IDs[index], err)}
					cancel()
					return
				}
				latencies = append(latencies, float64(time.Since(queryStarted))/float64(time.Millisecond))
			}
			results <- concurrentWorkerResult{latencies: latencies}
		}(worker)
	}
	latencies := make([]float64, 0)
	var firstError error
	for range concurrency {
		result := <-results
		latencies = append(latencies, result.latencies...)
		if firstError == nil && result.err != nil {
			firstError = result.err
		}
	}
	elapsed := time.Since(started)
	if firstError != nil {
		return searchMetrics{}, firstError
	}
	if err := ctx.Err(); err != nil {
		return searchMetrics{}, err
	}
	if len(latencies) == 0 {
		return searchMetrics{}, errors.New("concurrent search completed no queries")
	}
	return summarizeSearch(latencies, elapsed), nil
}

func (d queryData) query(index int) benchmarkQuery {
	query := benchmarkQuery{}
	if index < len(d.Vectors) {
		query.Vector = d.Vectors[index]
	}
	if index < len(d.Texts) {
		query.Text = d.Texts[index]
	}
	return query
}

func recallAtK(k int, groundTruth map[string]int, got []string) float64 {
	denominator := min(k, len(groundTruth))
	if denominator == 0 {
		return 0
	}
	type relevantDocument struct {
		id        string
		relevance int
	}
	ranked := make([]relevantDocument, 0, len(groundTruth))
	for id, relevance := range groundTruth {
		ranked = append(ranked, relevantDocument{id: id, relevance: relevance})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].relevance == ranked[j].relevance {
			return ranked[i].id < ranked[j].id
		}
		return ranked[i].relevance > ranked[j].relevance
	})
	relevant := make(map[string]struct{}, denominator)
	for _, document := range ranked[:denominator] {
		relevant[document.id] = struct{}{}
	}
	hits := 0
	seen := make(map[string]struct{}, min(k, len(got)))
	for _, id := range got[:min(k, len(got))] {
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		if _, found := relevant[id]; found {
			hits++
		}
	}
	return float64(hits) / float64(denominator)
}

func recallFTSAtK(k int, groundTruth map[string]int, got []string) float64 {
	if k <= 0 || len(groundTruth) == 0 {
		return 0
	}
	hits := 0
	seen := make(map[string]struct{}, min(k, len(got)))
	for _, id := range got[:min(k, len(got))] {
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		if groundTruth[id] > 0 {
			hits++
		}
	}
	return float64(hits) / float64(len(groundTruth))
}

func mrrFTSAtK(k int, groundTruth map[string]int, got []string) float64 {
	for rank, id := range got[:min(k, len(got))] {
		if groundTruth[id] > 0 {
			return 1 / float64(rank+1)
		}
	}
	return 0
}

func ndcgFTSAtK(k int, groundTruth map[string]int, got []string) float64 {
	if k <= 0 || len(groundTruth) == 0 {
		return 0
	}
	var dcg float64
	seen := make(map[string]struct{}, min(k, len(got)))
	for rank, id := range got[:min(k, len(got))] {
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		if relevance := groundTruth[id]; relevance > 0 {
			dcg += float64(relevance) / math.Log2(float64(rank+2))
		}
	}
	relevances := make([]int, 0, len(groundTruth))
	for _, relevance := range groundTruth {
		if relevance > 0 {
			relevances = append(relevances, relevance)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(relevances)))
	var ideal float64
	for rank, relevance := range relevances[:min(k, len(relevances))] {
		ideal += float64(relevance) / math.Log2(float64(rank+2))
	}
	if ideal == 0 {
		return 0
	}
	return dcg / ideal
}

func summarizeSearch(latencies []float64, elapsed time.Duration) searchMetrics {
	values := append([]float64(nil), latencies...)
	sort.Float64s(values)
	var sum float64
	for _, latency := range values {
		sum += latency
	}
	metrics := searchMetrics{
		Queries: len(values), LatencyAvgMS: sum / float64(len(values)),
		LatencyP95MS: percentile(values, 0.95), LatencyP99MS: percentile(values, 0.99),
	}
	if elapsed > 0 {
		metrics.QPS = float64(len(values)) / elapsed.Seconds()
	}
	return metrics
}

func percentile(sorted []float64, quantile float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	position := quantile * float64(len(sorted)-1)
	lower := int(position)
	upper := min(lower+1, len(sorted)-1)
	fraction := position - float64(lower)
	return sorted[lower] + (sorted[upper]-sorted[lower])*fraction
}
