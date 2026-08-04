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
	"math"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDenseFlatSearchMetrics(t *testing.T) {
	t.Parallel()
	tests := []struct {
		metric Metric
		want   []Result
	}{
		{MetricL2, []Result{{Key: 30, Score: 0}, {Key: 5, Score: 1}, {Key: 10, Score: 1}}},
		{MetricIP, []Result{{Key: 5, Score: 2}, {Key: 10, Score: 2}, {Key: 30, Score: 1}}},
		{MetricCosine, []Result{{Key: 5, Score: 0}, {Key: 10, Score: 0}, {Key: 30, Score: 0}}},
		{MetricMIPSL2, []Result{{Key: 30, Score: 0}, {Key: 5, Score: 1}, {Key: 10, Score: 1}}},
	}
	for _, testCase := range tests {
		index, err := NewDenseFlatIndex(2, testCase.metric)
		require.NoError(t, err)

		for _, candidate := range exactCandidates {
			{
				err := index.Add(context.Background(), candidate.Key, candidate.Vector)
				require.NoError(t, err)
			}
		}
		got, err := index.Search(context.Background(), []float32{1, 0}, 3)
		require.NoError(t, err)
		require.Equal(t, testCase.want, got)
	}
}

func TestDenseFlatProviderAndInputOwnership(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricIP)
	require.NoError(t, err)

	vector := []float32{1, 2}
	{
		err := index.Add(context.Background(), 42, vector)
		require.NoError(t, err)
	}

	vector[0] = 99
	stored, found := index.Vector(42)
	require.True(t, found)
	require.Equal(t, []float32{1, 2}, stored)

	stored[0] = 88
	again, found := index.Vector(42)
	require.True(t, found)
	require.Equal(t, []float32{1, 2}, again)
	{
		_, found := index.Vector(7)
		require.False(t, found,
			"missing key was found")
	}
	require.True(t, index.Dimension() == 2)
	require.Equal(t, MetricIP, index.Metric())
	require.True(t, index.Len() == 1)
}

func TestDenseFlatValidation(t *testing.T) {
	{
		_, err := NewDenseFlatIndex(0, MetricL2)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
	{
		_, err := NewDenseFlatIndex(2, Metric(99))
		require.Error(t, err,
			"invalid metric succeeded")
	}

	index, err := NewDenseFlatIndex(2, MetricL2)
	require.NoError(t, err)
	{
		err := index.Add(context.Background(), 1, []float32{1})
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
	{
		err := index.Add(context.Background(), 1, []float32{1, float32(math.NaN())})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}
	{
		err := index.Add(context.Background(), 1, []float32{1, 2})
		require.NoError(t, err)
	}
	{
		err := index.Add(context.Background(), 1, []float32{3, 4})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		_, err := index.Search(context.Background(), []float32{1}, 1)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}
	{
		_, err := index.Search(context.Background(), []float32{1, 2}, -1)
		require.Error(t, err,
			"negative top-k succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Add(canceled, 2, []float32{1, 2})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := index.Search(canceled, []float32{1, 2}, 1)
		require.ErrorIs(t, err, context.Canceled)
	}

	var nilIndex *DenseFlatIndex
	{
		err := nilIndex.Add(context.Background(), 1, []float32{1})
		require.Error(t, err,
			"nil index add succeeded")
	}
	{
		_, err := nilIndex.Search(context.Background(), []float32{1}, 1)
		require.Error(t, err,
			"nil index search succeeded")
	}
}

func TestDenseFlatBuilderLifecycle(t *testing.T) {
	builder, err := NewDenseFlatBuilder(2, MetricIP)
	require.NoError(t, err)
	{
		err := builder.Add(context.Background(), 2, []float32{2, 0})
		require.NoError(t, err)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1, 0})
		require.NoError(t, err)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	{
		err := builder.Add(context.Background(), 3, []float32{3, 0})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		err := index.Add(context.Background(), 3, []float32{3, 0})
		require.NoError(t, err)
	}

	results, err := index.Search(context.Background(), []float32{1, 0}, 3)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 3, Score: 3}, {Key: 2, Score: 2}, {Key: 1, Score: 1}}, results)

	var nilBuilder *DenseFlatIndexBuilder
	{
		err := nilBuilder.Add(context.Background(), 1, []float32{1})
		require.Error(t, err,
			"nil builder add succeeded")
	}
	{
		_, err := nilBuilder.Build(context.Background())
		require.Error(t, err,
			"nil builder build succeeded")
	}
}

func TestDenseFlatConcurrentStreamingAndSearch(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricL2)
	require.NoError(t, err)

	const count = 128
	var wait sync.WaitGroup
	for key := range count {
		wait.Add(1)
		go func() {
			defer wait.Done()
			value := float32(key)
			{
				err := index.Add(context.Background(), uint64(key), []float32{value, value})
				assert.NoError(t, err)
			}
			{
				_, err := index.Search(context.Background(), []float32{0, 0}, 10)
				assert.NoError(t, err)
			}
		}()
	}
	wait.Wait()
	require.Equal(t, count, index.Len())

	results, err := index.Search(context.Background(), []float32{0, 0}, 3)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 0, Score: 0}, {Key: 1, Score: 2}, {Key: 2, Score: 8}}, results)
}

func BenchmarkDenseFlatSearch(b *testing.B) {
	index, err := NewDenseFlatIndex(32, MetricL2)
	if err != nil {
		require.NoError(b, err)
	}

	for key := range 10_000 {
		vector := make([]float32, 32)
		for dimension := range vector {
			vector[dimension] = float32(key+dimension) / 10_000
		}
		{
			err := index.Add(context.Background(), uint64(key), vector)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
	query := make([]float32, 32)
	b.ResetTimer()
	for range b.N {
		{
			_, err := index.Search(context.Background(), query, 10)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}
