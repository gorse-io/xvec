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
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	_ "modernc.org/sqlite"
	_ "modernc.org/sqlite/vec"
)

const sqliteVecTable = "vectors"

func loadSQLiteVecDataset(ctx context.Context, config benchConfig, log io.Writer) (loadMetrics, error) {
	if !config.SkipDropOld {
		for _, suffix := range []string{"", "-shm", "-wal"} {
			if err := os.Remove(config.Path + suffix); err != nil && !errors.Is(err, os.ErrNotExist) {
				return loadMetrics{}, fmt.Errorf("remove old sqlite-vec database: %w", err)
			}
		}
	}
	if err := os.MkdirAll(filepath.Dir(config.Path), 0o755); err != nil {
		return loadMetrics{}, fmt.Errorf("create sqlite-vec database directory: %w", err)
	}
	database, err := sql.Open("sqlite", config.Path)
	if err != nil {
		return loadMetrics{}, fmt.Errorf("open sqlite-vec database: %w", err)
	}
	closed := false
	defer func() {
		if !closed {
			_ = database.Close()
		}
	}()
	if _, err := database.ExecContext(ctx, "PRAGMA journal_mode=WAL"); err != nil {
		return loadMetrics{}, fmt.Errorf("enable sqlite-vec WAL: %w", err)
	}
	if _, err := database.ExecContext(ctx, "PRAGMA synchronous=NORMAL"); err != nil {
		return loadMetrics{}, fmt.Errorf("configure sqlite-vec synchronization: %w", err)
	}
	if !config.SkipDropOld {
		statement := fmt.Sprintf(
			"CREATE VIRTUAL TABLE %s USING vec0(id integer primary key, embedding float[%d] distance_metric=%s)",
			sqliteVecTable, config.caseSpec.Dimension, config.caseSpec.Metric,
		)
		if _, err := database.ExecContext(ctx, statement); err != nil {
			return loadMetrics{}, fmt.Errorf("create sqlite-vec table: %w", err)
		}
	}

	loadStarted := time.Now()
	insertStarted := loadStarted
	inserted := int64(0)
	nextProgress := int64(100_000)
	rows, err := forEachTrainingBatch(
		ctx, config.DatasetDir, config.caseSpec.TrainFiles, config.BatchSize, config.LoadLimit,
		func(rows []vectorParquetRow) error {
			transaction, err := database.BeginTx(ctx, nil)
			if err != nil {
				return fmt.Errorf("begin sqlite-vec insert batch: %w", err)
			}
			committed := false
			defer func() {
				if !committed {
					_ = transaction.Rollback()
				}
			}()
			statement, err := transaction.PrepareContext(ctx, "INSERT INTO "+sqliteVecTable+"(id, embedding) VALUES (?, ?)")
			if err != nil {
				return fmt.Errorf("prepare sqlite-vec insert: %w", err)
			}
			defer func() { _ = statement.Close() }()
			for _, row := range rows {
				if len(row.Embedding) != config.caseSpec.Dimension {
					return fmt.Errorf("training vector %d has dimension %d, want %d", row.ID, len(row.Embedding), config.caseSpec.Dimension)
				}
				if _, err := statement.ExecContext(ctx, row.ID, sqliteVecVectorBytes(row.Embedding)); err != nil {
					return fmt.Errorf("insert sqlite-vec vector %d: %w", row.ID, err)
				}
			}
			if err := statement.Close(); err != nil {
				return fmt.Errorf("close sqlite-vec insert statement: %w", err)
			}
			if err := transaction.Commit(); err != nil {
				return fmt.Errorf("commit sqlite-vec insert batch: %w", err)
			}
			committed = true
			inserted += int64(len(rows))
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
	if _, err := database.ExecContext(ctx, "PRAGMA wal_checkpoint(TRUNCATE)"); err != nil {
		return loadMetrics{}, fmt.Errorf("checkpoint sqlite-vec database: %w", err)
	}
	if err := database.Close(); err != nil {
		return loadMetrics{}, fmt.Errorf("close loaded sqlite-vec database: %w", err)
	}
	closed = true
	info, err := os.Stat(config.Path)
	if err != nil {
		return loadMetrics{}, fmt.Errorf("stat sqlite-vec database: %w", err)
	}
	metrics := loadMetrics{
		Rows: rows, InsertDurationSec: insertDuration.Seconds(), LoadDurationSec: time.Since(loadStarted).Seconds(),
		StorageBytes: uint64(info.Size()),
	}
	if insertDuration > 0 {
		metrics.RowsPerSecond = float64(rows) / insertDuration.Seconds()
	}
	return metrics, nil
}

type sqliteVecQueryEngine struct {
	database *sql.DB
	k        int
}

func openSQLiteVecQueryEngine(config benchConfig) (benchmarkQueryEngine, io.Closer, error) {
	if _, err := os.Stat(config.Path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil, fmt.Errorf("sqlite-vec benchmark database does not exist: %s", config.Path)
		}
		return nil, nil, fmt.Errorf("stat sqlite-vec benchmark database: %w", err)
	}
	database, err := sql.Open("sqlite", config.Path)
	if err != nil {
		return nil, nil, fmt.Errorf("open sqlite-vec benchmark database for search: %w", err)
	}
	if err := database.Ping(); err != nil {
		_ = database.Close()
		return nil, nil, fmt.Errorf("ping sqlite-vec benchmark database: %w", err)
	}
	return sqliteVecQueryEngine{database: database, k: config.K}, database, nil
}

func (e sqliteVecQueryEngine) search(ctx context.Context, query benchmarkQuery) ([]string, error) {
	rows, err := e.database.QueryContext(ctx,
		"SELECT id FROM "+sqliteVecTable+" WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
		sqliteVecVectorBytes(query.Vector), e.k,
	)
	if err != nil {
		return nil, fmt.Errorf("query sqlite-vec: %w", err)
	}
	defer func() { _ = rows.Close() }()
	ids := make([]string, 0, e.k)
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan sqlite-vec result: %w", err)
		}
		ids = append(ids, strconv.FormatInt(id, 10))
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sqlite-vec results: %w", err)
	}
	if err := rows.Close(); err != nil {
		return nil, fmt.Errorf("close sqlite-vec results: %w", err)
	}
	return ids, nil
}

func sqliteVecVectorBytes(vector []float32) []byte {
	data := make([]byte, len(vector)*4)
	for index, value := range vector {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return data
}
