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
	"reflect"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

var exactCandidates = []Candidate{
	{Key: 30, Vector: []float32{1, 0}},
	{Key: 10, Vector: []float32{2, 0}},
	{Key: 20, Vector: []float32{0, 1}},
	{Key: 5, Vector: []float32{2, 0}},
	{Key: 40, Vector: []float32{-1, 0}},
}

func TestTopKMetricOrdering(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		metric   Metric
		expected []Result
	}{
		{
			name:   "l2 lower is better",
			metric: MetricL2,
			expected: []Result{
				{Key: 30, Score: 0},
				{Key: 5, Score: 1},
				{Key: 10, Score: 1},
			},
		},
		{
			name:   "inner product higher is better",
			metric: MetricIP,
			expected: []Result{
				{Key: 5, Score: 2},
				{Key: 10, Score: 2},
				{Key: 30, Score: 1},
			},
		},
		{
			name:   "cosine distance lower is better",
			metric: MetricCosine,
			expected: []Result{
				{Key: 5, Score: 0},
				{Key: 10, Score: 0},
				{Key: 30, Score: 0},
			},
		},
		{
			name:   "mips l2 lower is better",
			metric: MetricMIPSL2,
			expected: []Result{
				{Key: 30, Score: 0},
				{Key: 5, Score: 1},
				{Key: 10, Score: 1},
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			results, err := TopK(context.Background(), testCase.metric, []float32{1, 0}, exactCandidates, 3)
			if err != nil {
				t.Fatalf("top-k: %v", err)
			}
			if !reflect.DeepEqual(results, testCase.expected) {
				t.Fatalf("results = %#v, want %#v", results, testCase.expected)
			}
		})
	}
}

func TestTopKStableAcrossCandidateOrder(t *testing.T) {
	t.Parallel()

	forward, err := TopK(context.Background(), MetricIP, []float32{1, 0}, exactCandidates, 4)
	if err != nil {
		t.Fatalf("forward top-k: %v", err)
	}
	reversedCandidates := append([]Candidate(nil), exactCandidates...)
	for left, right := 0, len(reversedCandidates)-1; left < right; left, right = left+1, right-1 {
		reversedCandidates[left], reversedCandidates[right] = reversedCandidates[right], reversedCandidates[left]
	}
	reverse, err := TopK(context.Background(), MetricIP, []float32{1, 0}, reversedCandidates, 4)
	if err != nil {
		t.Fatalf("reverse top-k: %v", err)
	}
	if !reflect.DeepEqual(forward, reverse) {
		t.Fatalf("forward = %#v, reverse = %#v", forward, reverse)
	}
}

func TestTopKBoundsAndValidation(t *testing.T) {
	t.Parallel()

	results, err := TopK(context.Background(), MetricL2, []float32{1, 0}, exactCandidates[:2], 8)
	if err != nil {
		t.Fatalf("top-k with oversized k: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("result count = %d, want 2", len(results))
	}

	results, err = TopK(context.Background(), MetricL2, []float32{1, 0}, exactCandidates, 0)
	if err != nil {
		t.Fatalf("top-k zero: %v", err)
	}
	if results == nil || len(results) != 0 {
		t.Fatalf("zero results = %#v, want non-nil empty slice", results)
	}

	if _, err = TopK(context.Background(), MetricL2, []float32{1}, nil, -1); err == nil {
		t.Fatal("negative k succeeded")
	}
	if _, err = TopK(context.Background(), Metric(255), []float32{1}, nil, 1); err == nil {
		t.Fatal("invalid metric succeeded")
	}
	if _, err = TopK(context.Background(), MetricL2, nil, nil, 1); !errors.Is(err, ailego.ErrEmptyVector) {
		t.Fatalf("empty query error = %v", err)
	}
	if _, err = TopK(context.Background(), MetricL2, []float32{1}, []Candidate{{Key: 1, Vector: []float32{1, 2}}}, 1); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("candidate dimension error = %v", err)
	}
	if _, err = TopK(nil, MetricL2, []float32{1}, nil, 1); err == nil {
		t.Fatal("nil context succeeded")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = TopK(ctx, MetricL2, []float32{1}, nil, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
}

func TestBatchTopK(t *testing.T) {
	t.Parallel()

	queries := [][]float32{{1, 0}, {0, 1}, {-1, 0}}
	batch, err := BatchTopK(context.Background(), MetricIP, queries, exactCandidates, 2, 2)
	if err != nil {
		t.Fatalf("batch top-k: %v", err)
	}
	if len(batch) != len(queries) {
		t.Fatalf("batch count = %d, want %d", len(batch), len(queries))
	}
	for index, query := range queries {
		expected, err := TopK(context.Background(), MetricIP, query, exactCandidates, 2)
		if err != nil {
			t.Fatalf("sequential query %d: %v", index, err)
		}
		if !reflect.DeepEqual(batch[index], expected) {
			t.Fatalf("query %d = %#v, want %#v", index, batch[index], expected)
		}
	}

	empty, err := BatchTopK(context.Background(), MetricL2, nil, exactCandidates, 2, 4)
	if err != nil {
		t.Fatalf("empty batch: %v", err)
	}
	if empty == nil || len(empty) != 0 {
		t.Fatalf("empty batch = %#v, want non-nil empty slice", empty)
	}

	if _, err = BatchTopK(context.Background(), MetricIP, [][]float32{{1, 0}, {1}}, exactCandidates, 2, 2); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("malformed query error = %v", err)
	}
	defaultWorkers, err := BatchTopK(context.Background(), MetricIP, queries, exactCandidates, 2, 0)
	if err != nil {
		t.Fatalf("default workers: %v", err)
	}
	if !reflect.DeepEqual(defaultWorkers, batch) {
		t.Fatalf("default workers = %#v, want %#v", defaultWorkers, batch)
	}
	if _, err = BatchTopK(nil, MetricIP, queries, exactCandidates, 2, 1); err == nil {
		t.Fatal("nil context succeeded")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = BatchTopK(ctx, MetricIP, queries, exactCandidates, 2, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled batch error = %v", err)
	}
}
