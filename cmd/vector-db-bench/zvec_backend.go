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

//go:build purego || !cgo

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	zvec "github.com/zvec-ai/zvec-go"
)

func initializeZvecBackend() (func(), error) {
	if err := zvec.Initialize(nil); err != nil {
		return nil, fmt.Errorf("initialize zvec: %w", err)
	}
	return func() { _ = zvec.Shutdown() }, nil
}

func loadZvecDataset(ctx context.Context, config benchConfig, log io.Writer) (loadMetrics, error) {
	collection, err := writableZvecBenchmarkCollection(config)
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
	rows, err := forEachTrainingBatch(
		ctx, config.DatasetDir, config.caseSpec.TrainFiles, config.BatchSize, config.LoadLimit,
		func(rows []vectorParquetRow) error {
			documents := make([]*zvec.Doc, len(rows))
			defer func() {
				for _, document := range documents {
					if document != nil {
						document.Destroy()
					}
				}
			}()
			for index, row := range rows {
				if len(row.Embedding) != config.caseSpec.Dimension {
					return fmt.Errorf("training vector %d has dimension %d, want %d", row.ID, len(row.Embedding), config.caseSpec.Dimension)
				}
				document := zvec.NewDoc()
				if document == nil {
					return errors.New("create zvec document")
				}
				documents[index] = document
				document.SetPK(strconv.FormatInt(row.ID, 10))
				if err := document.AddInt64Field("id", row.ID); err != nil {
					return fmt.Errorf("set zvec document id %d: %w", row.ID, err)
				}
				if err := document.AddVectorFP32Field("dense", row.Embedding); err != nil {
					return fmt.Errorf("set zvec document vector %d: %w", row.ID, err)
				}
			}
			result, err := collection.Insert(documents)
			if err != nil {
				return fmt.Errorf("insert zvec batch at row %d: %w", inserted, err)
			}
			if result.ErrorCount != 0 || result.SuccessCount != uint64(len(documents)) {
				return fmt.Errorf("insert zvec batch at row %d: %d succeeded, %d failed", inserted, result.SuccessCount, result.ErrorCount)
			}
			inserted += int64(len(documents))
			if inserted >= nextProgress {
				_, _ = fmt.Fprintf(log, "inserted %d vectors (%.1f rows/s)\n", inserted, float64(inserted)/time.Since(insertStarted).Seconds())
				nextProgress = (inserted/100_000 + 1) * 100_000
			}
			return nil
		},
	)
	if err != nil {
		return loadMetrics{}, err
	}
	insertDuration := time.Since(insertStarted)
	optimizeStarted := time.Now()
	if err := collection.Optimize(); err != nil {
		return loadMetrics{}, fmt.Errorf("optimize zvec benchmark collection: %w", err)
	}
	optimizeDuration := time.Since(optimizeStarted)
	if err := collection.Flush(); err != nil {
		return loadMetrics{}, fmt.Errorf("flush zvec benchmark collection: %w", err)
	}
	if err := collection.Close(); err != nil {
		return loadMetrics{}, fmt.Errorf("close loaded zvec collection: %w", err)
	}
	closed = true
	metrics := loadMetrics{
		Rows: rows, InsertDurationSec: insertDuration.Seconds(), OptimizeDurationSec: optimizeDuration.Seconds(),
		LoadDurationSec: time.Since(loadStarted).Seconds(),
	}
	if insertDuration > 0 {
		metrics.RowsPerSecond = float64(rows) / insertDuration.Seconds()
	}
	return metrics, nil
}

func writableZvecBenchmarkCollection(config benchConfig) (*zvec.Collection, error) {
	options, err := newZvecCollectionOptions(config, false)
	if err != nil {
		return nil, err
	}
	defer options.Destroy()
	if config.SkipDropOld {
		collection, err := zvec.Open(config.Path, options)
		if err != nil {
			return nil, fmt.Errorf("open existing zvec benchmark collection: %w", err)
		}
		return collection, nil
	}
	if _, err := os.Stat(config.Path); err == nil {
		old, err := zvec.Open(config.Path, options)
		if err != nil {
			return nil, fmt.Errorf("open old zvec benchmark collection before destroy: %w", err)
		}
		if err := old.Destroy(); err != nil {
			return nil, fmt.Errorf("destroy old zvec benchmark collection: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("stat zvec benchmark collection: %w", err)
	}

	metric, err := parseZvecMetric(config.caseSpec.Metric)
	if err != nil {
		return nil, err
	}
	index, err := zvec.NewHNSWIndexParams(metric, config.M, config.EFConstruction)
	if err != nil {
		return nil, fmt.Errorf("create zvec HNSW index params: %w", err)
	}
	defer index.Destroy()
	quantize, err := parseZvecQuantization(config.Quantize)
	if err != nil {
		return nil, err
	}
	if err := index.SetQuantizeType(quantize); err != nil {
		return nil, fmt.Errorf("set zvec quantization: %w", err)
	}

	schema := zvec.NewCollectionSchema("vector_bench_test")
	if schema == nil {
		return nil, errors.New("create zvec collection schema")
	}
	defer schema.Destroy()
	if err := schema.SetMaxDocCountPerSegment(config.MaxDocsPerSegment); err != nil {
		return nil, fmt.Errorf("set zvec maximum documents per segment: %w", err)
	}
	idField := zvec.NewFieldSchema("id", zvec.DataTypeInt64, false, 0)
	if idField == nil {
		return nil, errors.New("create zvec id field")
	}
	defer idField.Destroy()
	if err := schema.AddField(idField); err != nil {
		return nil, fmt.Errorf("add zvec id field: %w", err)
	}
	vectorField := zvec.NewFieldSchema("dense", zvec.DataTypeVectorFP32, false, uint32(config.caseSpec.Dimension))
	if vectorField == nil {
		return nil, errors.New("create zvec vector field")
	}
	defer vectorField.Destroy()
	if err := vectorField.SetIndexParams(index); err != nil {
		return nil, fmt.Errorf("set zvec vector index: %w", err)
	}
	if err := schema.AddField(vectorField); err != nil {
		return nil, fmt.Errorf("add zvec vector field: %w", err)
	}
	collection, err := zvec.CreateAndOpen(config.Path, schema, options)
	if err != nil {
		return nil, fmt.Errorf("create zvec benchmark collection: %w", err)
	}
	return collection, nil
}

func newZvecCollectionOptions(config benchConfig, readOnly bool) (*zvec.CollectionOptions, error) {
	options := zvec.NewCollectionOptions()
	if options == nil {
		return nil, errors.New("create zvec collection options")
	}
	if err := options.SetEnableMmap(config.EnableMmap); err != nil {
		options.Destroy()
		return nil, fmt.Errorf("set zvec mmap option: %w", err)
	}
	if err := options.SetMaxBufferSize(uint64(config.MaxBufferSize)); err != nil {
		options.Destroy()
		return nil, fmt.Errorf("set zvec maximum buffer size: %w", err)
	}
	if err := options.SetReadOnly(readOnly); err != nil {
		options.Destroy()
		return nil, fmt.Errorf("set zvec read-only option: %w", err)
	}
	return options, nil
}

func parseZvecMetric(value string) (zvec.MetricType, error) {
	switch strings.ToLower(value) {
	case "cosine":
		return zvec.MetricTypeCosine, nil
	case "l2":
		return zvec.MetricTypeL2, nil
	case "ip":
		return zvec.MetricTypeIP, nil
	default:
		return 0, fmt.Errorf("unsupported metric %q", value)
	}
}

func parseZvecQuantization(value string) (zvec.QuantizeType, error) {
	switch strings.ToLower(value) {
	case "", "none":
		return zvec.QuantizeTypeUndefined, nil
	case "fp16":
		return zvec.QuantizeTypeFP16, nil
	case "int8":
		return zvec.QuantizeTypeInt8, nil
	case "int4":
		return zvec.QuantizeTypeInt4, nil
	default:
		return 0, fmt.Errorf("unsupported quantization %q", value)
	}
}

type zvecQueryEngine struct {
	collection *zvec.Collection
	ef         int
	useRefiner bool
	k          int
}

func openZvecQueryEngine(config benchConfig) (benchmarkQueryEngine, io.Closer, error) {
	options, err := newZvecCollectionOptions(config, true)
	if err != nil {
		return nil, nil, err
	}
	defer options.Destroy()
	collection, err := zvec.Open(config.Path, options)
	if err != nil {
		return nil, nil, fmt.Errorf("open zvec benchmark collection for search: %w", err)
	}
	return zvecQueryEngine{collection: collection, ef: config.EFSearch, useRefiner: config.UseRefiner, k: config.K}, collection, nil
}

func (e zvecQueryEngine) search(ctx context.Context, vector []float32) ([]int64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	query := zvec.NewSearchQuery()
	if query == nil {
		return nil, errors.New("create zvec search query")
	}
	defer query.Destroy()
	if err := query.SetFieldName("dense"); err != nil {
		return nil, fmt.Errorf("set zvec query field: %w", err)
	}
	if err := query.SetQueryVector(vector); err != nil {
		return nil, fmt.Errorf("set zvec query vector: %w", err)
	}
	if err := query.SetTopK(e.k); err != nil {
		return nil, fmt.Errorf("set zvec query top K: %w", err)
	}
	if err := query.SetIncludeVector(false); err != nil {
		return nil, fmt.Errorf("set zvec query vector projection: %w", err)
	}
	if err := query.SetIncludeDocID(false); err != nil {
		return nil, fmt.Errorf("set zvec query document ID projection: %w", err)
	}
	params := zvec.NewHNSWQueryParams(e.ef, -1, false, e.useRefiner)
	if params == nil {
		return nil, errors.New("create zvec HNSW query params")
	}
	defer params.Destroy()
	if err := query.SetHNSWParams(params); err != nil {
		return nil, fmt.Errorf("set zvec HNSW query params: %w", err)
	}
	results, err := e.collection.Query(query)
	if err != nil {
		return nil, err
	}
	defer zvec.FreeDocs(results)
	ids := make([]int64, len(results))
	for index, result := range results {
		ids[index], err = strconv.ParseInt(result.GetPK(), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse zvec result primary key %q: %w", result.GetPK(), err)
		}
	}
	return ids, ctx.Err()
}
