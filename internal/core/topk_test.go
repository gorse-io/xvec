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
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
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
			require.NoError(t, err)
			require.Equal(t, testCase.expected, results)
		})
	}
}

func TestTopKStableAcrossCandidateOrder(t *testing.T) {
	t.Parallel()

	forward, err := TopK(context.Background(), MetricIP, []float32{1, 0}, exactCandidates, 4)
	require.NoError(t, err)

	reversedCandidates := append([]Candidate(nil), exactCandidates...)
	for left, right := 0, len(reversedCandidates)-1; left < right; left, right = left+1, right-1 {
		reversedCandidates[left], reversedCandidates[right] = reversedCandidates[right], reversedCandidates[left]
	}
	reverse, err := TopK(context.Background(), MetricIP, []float32{1, 0}, reversedCandidates, 4)
	require.NoError(t, err)
	require.Equal(t, reverse, forward)
}

func TestTopKBoundsAndValidation(t *testing.T) {
	t.Parallel()

	results, err := TopK(context.Background(), MetricL2, []float32{1, 0}, exactCandidates[:2], 8)
	require.NoError(t, err)
	require.Len(t, results, 2)

	results, err = TopK(context.Background(), MetricL2, []float32{1, 0}, exactCandidates, 0)
	require.NoError(t, err)
	require.NotNil(t, results)
	require.Len(t, results, 0)
	{
		_, err = TopK(context.Background(), MetricL2, []float32{1}, nil, -1)
		require.Error(t, err,
			"negative k succeeded")
	}
	{
		_, err = TopK(context.Background(), Metric(255), []float32{1}, nil, 1)
		require.Error(t, err,
			"invalid metric succeeded")
	}
	{
		_, err = TopK(context.Background(), MetricL2, nil, nil, 1)
		require.ErrorIs(t, err, ailego.ErrEmptyVector)
	}
	{
		_, err = TopK(context.Background(), MetricL2, []float32{1}, []Candidate{{Key: 1, Vector: []float32{1, 2}}}, 1)
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		_, err = TopK(nil, MetricL2, []float32{1}, nil, 1)
		require.Error(t, err,
			"nil context succeeded")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err = TopK(ctx, MetricL2, []float32{1}, nil, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestBatchTopK(t *testing.T) {
	t.Parallel()

	queries := [][]float32{{1, 0}, {0, 1}, {-1, 0}}
	batch, err := BatchTopK(context.Background(), MetricIP, queries, exactCandidates, 2, 2)
	require.NoError(t, err)
	require.Len(t, batch, len(queries))

	for index, query := range queries {
		expected, err := TopK(context.Background(), MetricIP, query, exactCandidates, 2)
		require.NoError(t, err)
		require.Equal(t, expected, batch[index])
	}

	empty, err := BatchTopK(context.Background(), MetricL2, nil, exactCandidates, 2, 4)
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
	{
		_, err = BatchTopK(context.Background(), MetricIP, [][]float32{{1, 0}, {1}}, exactCandidates, 2, 2)
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}

	defaultWorkers, err := BatchTopK(context.Background(), MetricIP, queries, exactCandidates, 2, 0)
	require.NoError(t, err)
	require.Equal(t, batch, defaultWorkers)
	{
		_, err = BatchTopK(nil, MetricIP, queries, exactCandidates, 2, 1)
		require.Error(t, err,
			"nil context succeeded")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err = BatchTopK(ctx, MetricIP, queries, exactCandidates, 2, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
}
