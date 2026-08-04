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

package core

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gorse-io/zvec/internal/ailego"
)

// PQDistanceTable stores chunk-major public scores for one query. L2 entries
// are squared distances and inner-product entries are similarities.
type PQDistanceTable struct {
	modelFingerprint uint64
	metric           Metric
	chunks           int
	values           []float32
}

func (t *PQDistanceTable) Metric() Metric {
	if t == nil {
		return 0
	}
	return t.metric
}

func (t *PQDistanceTable) Chunks() int {
	if t == nil {
		return 0
	}
	return t.chunks
}

func (t *PQDistanceTable) Centroids() int {
	if t == nil {
		return 0
	}
	return PQCentroidCount
}

func (t *PQDistanceTable) Values() []float32 {
	if t == nil {
		return nil
	}
	return slices.Clone(t.values)
}

// DistanceTable computes all 256 query scores for every chunk.
func (m *PQModel) DistanceTable(query []float32) (*PQDistanceTable, error) {
	if err := m.validate(); err != nil {
		return nil, err
	}
	if err := validateTrainingVector(query, m.dimension); err != nil {
		return nil, err
	}
	values := make([]float32, m.Chunks()*PQCentroidCount)
	for chunk := 0; chunk < m.Chunks(); chunk++ {
		start, end := m.chunkOffsets[chunk], m.chunkOffsets[chunk+1]
		for centroid := 0; centroid < PQCentroidCount; centroid++ {
			score, err := m.subspaceScore(query, centroid, start, end)
			if err != nil {
				return nil, fmt.Errorf("core: build PQ distance table chunk %d centroid %d: %w", chunk, centroid, err)
			}
			values[chunk*PQCentroidCount+centroid] = score
		}
	}
	return &PQDistanceTable{
		modelFingerprint: m.fingerprint, metric: m.metric,
		chunks: m.Chunks(), values: values,
	}, nil
}

// Lookup sums one precomputed entry per code byte.
func (t *PQDistanceTable) Lookup(code PQCode) (float32, error) {
	if err := t.validate(); err != nil {
		return 0, err
	}
	if err := code.validate(); err != nil {
		return 0, err
	}
	if code.modelFingerprint != t.modelFingerprint {
		return 0, ErrPQModelMismatch
	}
	if len(code.codes) != t.chunks {
		return 0, ErrInvalidPQCode
	}
	var score float64
	for chunk, centroid := range code.codes {
		score += float64(t.values[chunk*PQCentroidCount+int(centroid)])
	}
	if math.IsNaN(score) || math.IsInf(score, 0) || score > math.MaxFloat32 || score < -math.MaxFloat32 {
		return 0, ErrPQScoreOverflow
	}
	return float32(score), nil
}

// LookupBatch evaluates codes concurrently while preserving input order.
func (t *PQDistanceTable) LookupBatch(ctx context.Context, codes []PQCode, workers int) ([]float32, error) {
	if ctx == nil {
		return nil, errors.New("core: nil PQ lookup context")
	}
	if err := t.validate(); err != nil {
		return nil, err
	}
	if workers < 0 {
		return nil, fmt.Errorf("%w: workers cannot be negative", ErrInvalidPQOptions)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]float32, len(codes))
	err := ailego.ParallelFor(ctx, len(codes), workers, func(ctx context.Context, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		score, err := t.Lookup(codes[index])
		if err != nil {
			return fmt.Errorf("core: lookup PQ code %d: %w", index, err)
		}
		result[index] = score
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Distance builds a table and evaluates one code.
func (m *PQModel) Distance(query []float32, code PQCode) (float32, error) {
	table, err := m.DistanceTable(query)
	if err != nil {
		return 0, err
	}
	return table.Lookup(code)
}

func (t *PQDistanceTable) validate() error {
	if t == nil || t.modelFingerprint == 0 || (t.metric != MetricL2 && t.metric != MetricIP) ||
		t.chunks <= 0 || len(t.values) != t.chunks*PQCentroidCount {
		return ErrInvalidPQModel
	}
	return nil
}
