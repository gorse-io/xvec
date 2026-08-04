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

func TestSparseFlatExactSearch(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	require.NoError(t, err)

	vectors := []struct {
		key    uint64
		vector SparseVector
	}{
		{30, SparseVector{Indices: []uint32{1, 4}, Values: []float32{1, 1}}},
		{10, SparseVector{Indices: []uint32{1, 3}, Values: []float32{2, 3}}},
		{5, SparseVector{Indices: []uint32{1, 9}, Values: []float32{2, 8}}},
		{20, SparseVector{Indices: []uint32{2}, Values: []float32{7}}},
	}
	for _, item := range vectors {
		{
			err := index.AddSparse(context.Background(), item.key, item.vector)
			require.NoError(t, err)
		}
	}
	results, err := index.SearchSparse(context.Background(), SparseVector{
		Indices: []uint32{1, 3}, Values: []float32{1, 1},
	}, 3)
	require.NoError(t, err)

	want := []Result{{Key: 10, Score: 5}, {Key: 5, Score: 2}, {Key: 30, Score: 1}}
	require.Equal(t, want, results)

	emptyQuery, err := index.SearchSparse(context.Background(), SparseVector{}, 4)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 5}, {Key: 10}, {Key: 20}, {Key: 30}}, emptyQuery)
}

func TestSparseFlatProviderOwnsInput(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	require.NoError(t, err)

	vector := SparseVector{Indices: []uint32{1, 3}, Values: []float32{2, 4}}
	{
		err := index.AddSparse(context.Background(), 7, vector)
		require.NoError(t, err)
	}

	vector.Indices[0], vector.Values[0] = 99, 99
	stored, found := index.SparseVector(7)
	require.True(t, found)
	require.Equal(t, SparseVector{Indices: []uint32{1, 3}, Values: []float32{2, 4}}, stored)

	stored.Indices[0], stored.Values[0] = 88, 88
	again, found := index.SparseVector(7)
	require.True(t, found)
	require.True(t, again.Indices[0] == 1)
	require.True(t, again.Values[0] == 2)
	{
		_, found := index.SparseVector(8)
		require.False(t, found,
			"missing sparse vector was found")
	}
}

func TestSparseFlatValidation(t *testing.T) {
	{
		_, err := NewSparseFlatIndex(MetricL2)
		require.Error(t, err,
			"sparse L2 succeeded")
	}

	index, err := NewSparseFlatIndex(MetricIP)
	require.NoError(t, err)

	tests := []struct {
		name   string
		vector SparseVector
		want   error
	}{
		{"length", SparseVector{Indices: []uint32{1}, Values: nil}, ailego.ErrDimensionMismatch},
		{"order", SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{"duplicate", SparseVector{Indices: []uint32{1, 1}, Values: []float32{1, 2}}, ailego.ErrInvalidSparseOrder},
		{"non-finite", SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.Inf(1))}}, ailego.ErrNonFiniteVector},
	}
	for _, testCase := range tests {
		{
			err := index.AddSparse(context.Background(), 1, testCase.vector)
			require.ErrorIs(t, err, testCase.want)
		}
	}
	{
		err := index.AddSparse(context.Background(), 1, SparseVector{})
		require.NoError(t, err)
	}
	{
		err := index.AddSparse(context.Background(), 1, SparseVector{})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}
	{
		err := index.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{math.MaxFloat32}})
		require.NoError(t, err)
	}
	{
		_, err := index.SearchSparse(context.Background(), SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 1}}, 1)
		require.ErrorIs(t, err, ailego.ErrInvalidSparseOrder)
	}
	{
		_, err := index.SearchSparse(context.Background(), SparseVector{}, -1)
		require.Error(t, err,
			"negative top-k succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.AddSparse(canceled, 2, SparseVector{})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := index.SearchSparse(canceled, SparseVector{}, 1)
		require.ErrorIs(t, err, context.Canceled)
	}

	var nilIndex *SparseFlatIndex
	{
		err := nilIndex.AddSparse(context.Background(), 1, SparseVector{})
		require.Error(t, err,
			"nil add succeeded")
	}
	{
		_, err := nilIndex.SearchSparse(context.Background(), SparseVector{}, 1)
		require.Error(t, err,
			"nil search succeeded")
	}
}

func TestSparseFlatBuilderAndStreaming(t *testing.T) {
	builder, err := NewSparseFlatBuilder(MetricIP)
	require.NoError(t, err)
	{
		err := builder.AddSparse(context.Background(), 1, SparseVector{Indices: []uint32{1}, Values: []float32{1}})
		require.NoError(t, err)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	{
		err := builder.AddSparse(context.Background(), 2, SparseVector{})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		err := index.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{2}})
		require.NoError(t, err)
	}

	results, err := index.SearchSparse(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, 2)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 2, Score: 2}, {Key: 1, Score: 1}}, results)

	var nilBuilder *SparseFlatIndexBuilder
	{
		err := nilBuilder.AddSparse(context.Background(), 1, SparseVector{})
		require.Error(t, err,
			"nil builder add succeeded")
	}
	{
		_, err := nilBuilder.Build(context.Background())
		require.Error(t, err,
			"nil builder build succeeded")
	}
}

func TestSparseFlatConcurrentStreamingAndSearch(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	require.NoError(t, err)

	const count = 128
	var wait sync.WaitGroup
	for key := range count {
		wait.Add(1)
		go func() {
			defer wait.Done()
			vector := SparseVector{Indices: []uint32{uint32(key)}, Values: []float32{float32(key)}}
			{
				err := index.AddSparse(context.Background(), uint64(key), vector)
				assert.NoError(t, err)
			}
			{
				_, err := index.SearchSparse(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, 10)
				assert.NoError(t, err)
			}
		}()
	}
	wait.Wait()
	require.Equal(t, count, index.Len())
}

func BenchmarkSparseFlatSearch(b *testing.B) {
	index, err := NewSparseFlatIndex(MetricIP)
	if err != nil {
		require.NoError(b, err)
	}

	for key := range 10_000 {
		vector := SparseVector{
			Indices: []uint32{uint32(key % 100), uint32(100 + key%100)},
			Values:  []float32{1, 2},
		}
		{
			err := index.AddSparse(context.Background(), uint64(key), vector)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
	query := SparseVector{Indices: []uint32{1, 101}, Values: []float32{1, 1}}
	b.ResetTimer()
	for range b.N {
		{
			_, err := index.SearchSparse(context.Background(), query, 10)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}
