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
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIVFBuildPartitionsVectors(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 2
	options.NIterations = 20
	options.Seed = 7
	builder, err := NewIVFBuilder(2, options)
	require.NoError(t, err)

	input := []Candidate{
		{Key: 10, Vector: []float32{0, 0}},
		{Key: 11, Vector: []float32{0, 1}},
		{Key: 20, Vector: []float32{10, 10}},
		{Key: 21, Vector: []float32{10, 11}},
	}
	for _, candidate := range input {
		{
			err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Dimension() == 2)
	require.Equal(t, MetricL2, index.Metric())
	require.True(t, index.Len() == 4)
	require.True(t, index.NList() == 2)

	centroids := index.Centroids()
	require.Len(t, centroids, 2)

	firstList, ok := index.ListForKey(10)
	require.True(t, ok,
		"key 10 is not assigned")
	{
		list, _ := index.ListForKey(11)
		require.Equal(t, firstList, list)
	}

	secondList, ok := index.ListForKey(20)
	require.True(t, ok)
	require.NotEqual(t, firstList, secondList)
	{
		list, _ := index.ListForKey(21)
		require.Equal(t, secondList, list)
	}

	for _, list := range []int{firstList, secondList} {
		candidates, err := index.List(list)
		require.NoError(t, err)
		require.Len(t, candidates, 2)
		require.True(t, candidates[0].Key <= candidates[1].Key)
	}
	require.False(t, index.TrainingIterations() == 0)
	require.False(t, math.IsNaN(index.TrainingCost()))
	require.False(t, math.IsInf(index.TrainingCost(), 0))
}

func TestIVFBuildOwnsOriginalVectors(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 1
	builder, _ := NewIVFBuilder(2, options)
	input := []float32{1, 2}
	{
		err := builder.Add(context.Background(), 7, input)
		require.NoError(t, err)
	}

	input[0] = 99
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	vector, found := index.Vector(7)
	require.True(t, found)
	require.True(t, slices.Equal(vector, []float32{1, 2}))

	vector[0] = 88
	candidates, _ := index.List(0)
	candidates[0].Vector[0] = 77
	centroids := index.Centroids()
	centroids[0][0] = 66
	vector, _ = index.Vector(7)
	require.True(t, vector[0] == 1,
		"IVF accessors expose mutable index state")
	require.True(t, index.Centroids()[0][0] == 1,
		"IVF accessors expose mutable index state")
}

func TestIVFBuildDeterministicAcrossWorkers(t *testing.T) {
	t.Parallel()
	build := func(workers int) *IVFIndex {
		options := DefaultIVFBuildOptions(MetricL2)
		options.NList = 7
		options.NIterations = 8
		options.Seed = 123
		options.Workers = workers
		builder, err := NewIVFBuilder(3, options)
		require.NoError(t, err)

		for index := 0; index < 100; index++ {
			vector := []float32{float32(index % 11), float32(index%7) / 2, float32(index%5) - 2}
			{
				err := builder.Add(context.Background(), uint64(index+1), vector)
				require.NoError(t, err)
			}
		}
		built, err := builder.Build(context.Background())
		require.NoError(t, err)

		return built
	}
	one, many := build(1), build(8)
	require.Equal(t, many.Centroids(), one.Centroids(),
		"IVF training differs across worker counts")
	require.Equal(t, many.TrainingCost(), one.TrainingCost(),
		"IVF training differs across worker counts")

	for key := uint64(1); key <= 100; key++ {
		left, _ := one.ListForKey(key)
		right, _ := many.ListForKey(key)
		require.Equal(t, right, left)
	}
}

func TestIVFBuildEmptyAndClusterCap(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	builder, _ := NewIVFBuilder(3, options)
	empty, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, empty.Len() == 0)
	require.True(t, empty.NList() == 0)
	require.Len(t, empty.Centroids(), 0)
	{
		_, err := empty.List(0)
		require.ErrorIs(t, err, ErrInvalidIVFList)
	}

	options.NList = 100
	builder, _ = NewIVFBuilder(1, options)
	for key := uint64(1); key <= 3; key++ {
		{
			err := builder.Add(context.Background(), key, []float32{float32(key)})
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.NList() == 3)
}

func TestIVFBuilderLifecycleAndValidation(t *testing.T) {
	t.Parallel()
	valid := DefaultIVFBuildOptions(MetricL2)
	for _, options := range []IVFBuildOptions{
		{},
		func() IVFBuildOptions { value := valid; value.Metric = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.NList = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.NIterations = 0; return value }(),
		func() IVFBuildOptions { value := valid; value.Tolerance = -1; return value }(),
		func() IVFBuildOptions { value := valid; value.Tolerance = math.NaN(); return value }(),
	} {
		{
			_, err := NewIVFBuilder(2, options)
			assert.ErrorIs(t, err, ErrInvalidIVFOptions)
		}
	}
	{
		_, err := NewIVFBuilder(0, valid)
		require.ErrorIs(t, err, ErrInvalidDimension)
	}

	builder, _ := NewIVFBuilder(2, valid)
	{
		err := builder.Add(nil, 1, []float32{1, 2})
		require.Error(t, err,
			"nil add context accepted")
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1})
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1, float32(math.Inf(1))})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{1, 2})
		require.NoError(t, err)
	}
	{
		err := builder.Add(context.Background(), 1, []float32{3, 4})
		require.ErrorIs(t, err, ErrDuplicateKey)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Len() == 1)
	{
		_, err := builder.Build(context.Background())
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		err := builder.Add(context.Background(), 2, []float32{3, 4})
		require.ErrorIs(t, err, ErrBuilderClosed)
	}
	{
		_, found := index.Vector(99)
		require.False(t, found,
			"missing vector found")
	}
	{
		_, found := index.ListForKey(99)
		require.False(t, found,
			"missing list assignment found")
	}
	{
		_, err := index.List(-1)
		require.ErrorIs(t, err, ErrInvalidIVFList)
	}
}

func TestIVFBuilderCancellationCanRetry(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricL2)
	options.NList = 2
	builder, _ := NewIVFBuilder(1, options)
	for key := uint64(1); key <= 4; key++ {
		{
			err := builder.Add(context.Background(), key, []float32{float32(key)})
			require.NoError(t, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := builder.Build(ctx)
		require.ErrorIs(t, err, context.Canceled)
	}

	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.Len() == 4)
}

func TestIVFBuildOptionsAreValueSemantic(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricIP)
	options.NList = 2
	builder, _ := NewIVFBuilder(1, options)
	for key, value := range map[uint64]float32{1: -2, 2: -1, 3: 1, 4: 2} {
		{
			err := builder.Add(context.Background(), key, []float32{value})
			require.NoError(t, err)
		}
	}
	options.NList = 99
	index, err := builder.Build(context.Background())
	require.NoError(t, err)
	require.True(t, index.BuildOptions().NList == 2)
	require.Equal(t, MetricIP, index.Metric())
}
