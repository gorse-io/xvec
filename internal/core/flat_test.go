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

package core

import (
	"context"
	"math"
	"strconv"
	"sync"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/math"
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
		require.ErrorIs(t, err, mathutil.ErrNonFiniteVector)
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
		{"length", SparseVector{Indices: []uint32{1}, Values: nil}, mathutil.ErrDimensionMismatch},
		{"order", SparseVector{Indices: []uint32{2, 1}, Values: []float32{1, 2}}, mathutil.ErrInvalidSparseOrder},
		{"duplicate", SparseVector{Indices: []uint32{1, 1}, Values: []float32{1, 2}}, mathutil.ErrInvalidSparseOrder},
		{"non-finite", SparseVector{Indices: []uint32{1}, Values: []float32{float32(math.Inf(1))}}, mathutil.ErrNonFiniteVector},
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
		require.ErrorIs(t, err, mathutil.ErrInvalidSparseOrder)
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

func TestGroupByOptionsValidation(t *testing.T) {
	resolver := func(uint64) (string, bool) { return "group", true }
	{
		err := (GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: resolver}).Validate()
		require.NoError(t, err)
	}

	tests := []struct {
		name    string
		options GroupByOptions
		want    error
	}{
		{name: "group count", options: GroupByOptions{TopKPerGroup: 1, Resolve: resolver}, want: ErrInvalidGroupCount},
		{name: "per-group top-k", options: GroupByOptions{GroupCount: 1, Resolve: resolver}, want: ErrInvalidGroupTopK},
		{name: "negative radius", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Radius: -1, Resolve: resolver}, want: ErrInvalidRadius},
		{name: "NaN radius", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Radius: float32(math.NaN()), Resolve: resolver}, want: ErrInvalidRadius},
		{name: "infinite radius", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Radius: float32(math.Inf(1)), Resolve: resolver}, want: ErrInvalidRadius},
		{name: "resolver", options: GroupByOptions{GroupCount: 1, TopKPerGroup: 1}, want: ErrNilGroupResolver},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			{
				err := testCase.options.Validate()
				require.ErrorIs(t, err, testCase.want)
			}
		})
	}
}

func TestDenseFlatSearchGroups(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricIP)
	require.NoError(t, err)

	for key := uint64(0); key < 12; key++ {
		{
			err := index.Add(context.Background(), key, []float32{float32(key), 1})
			require.NoError(t, err)
		}
	}
	options := GroupByOptions{
		GroupCount:   3,
		TopKPerGroup: 2,
		Resolve: func(key uint64) (string, bool) {
			return strconv.FormatUint(key%3, 10), true
		},
	}
	got, err := index.SearchGroups(context.Background(), []float32{1, 0}, options)
	require.NoError(t, err)

	want := []GroupResult{
		{Value: "2", Results: []Result{{Key: 11, Score: 11}, {Key: 8, Score: 8}}},
		{Value: "1", Results: []Result{{Key: 10, Score: 10}, {Key: 7, Score: 7}}},
		{Value: "0", Results: []Result{{Key: 9, Score: 9}, {Key: 6, Score: 6}}},
	}
	require.Equal(t, want, got)
}

func TestDenseFlatSearchGroupsFilterRadiusAndMissingValue(t *testing.T) {
	index, err := NewDenseFlatIndex(1, MetricIP)
	require.NoError(t, err)

	for key := uint64(0); key < 12; key++ {
		{
			err := index.Add(context.Background(), key, []float32{float32(key)})
			require.NoError(t, err)
		}
	}
	got, err := index.SearchGroups(context.Background(), []float32{1}, GroupByOptions{
		GroupCount:   3,
		TopKPerGroup: 3,
		Radius:       8,
		Filter:       func(key uint64) bool { return key != 10 },
		Resolve: func(key uint64) (string, bool) {
			if key == 9 {
				return "", false
			}
			if key%3 == 2 {
				return "", true
			}
			return strconv.FormatUint(key%3, 10), true
		},
	})
	require.NoError(t, err)

	want := []GroupResult{{Value: "", Results: []Result{{Key: 11, Score: 11}, {Key: 8, Score: 8}}}}
	require.Equal(t, want, got)
}

func TestDenseFlatSearchGroupsUsesMetricOrdering(t *testing.T) {
	index, err := NewDenseFlatIndex(1, MetricL2)
	require.NoError(t, err)

	for key, value := range []float32{0, 1, 2, 3} {
		{
			err := index.Add(context.Background(), uint64(key), []float32{value})
			require.NoError(t, err)
		}
	}
	got, err := index.SearchGroups(context.Background(), []float32{0}, GroupByOptions{
		GroupCount:   2,
		TopKPerGroup: 2,
		Resolve: func(key uint64) (string, bool) {
			if key%2 == 0 {
				return "even", true
			}
			return "odd", true
		},
	})
	require.NoError(t, err)

	want := []GroupResult{
		{Value: "even", Results: []Result{{Key: 0, Score: 0}, {Key: 2, Score: 4}}},
		{Value: "odd", Results: []Result{{Key: 1, Score: 1}, {Key: 3, Score: 9}}},
	}
	require.Equal(t, want, got)
}

func TestScalarQuantizedFlatSearchGroupsMatchesScalarScan(t *testing.T) {
	candidates := []Candidate{
		{Key: 1, Vector: []float32{1.013, .031, -.077, .125}},
		{Key: 2, Vector: []float32{.511, 2.037, .291, -.417}},
		{Key: 3, Vector: []float32{3.117, -.271, .051, -.2}},
		{Key: 4, Vector: []float32{1.919, .813, -.613, .411}},
		{Key: 5, Vector: []float32{-.719, 1.117, .333, 2.019}},
		{Key: 6, Vector: []float32{2.717, -.919, .231, .777}},
	}
	query := []float32{.9, .02, -.04, .1}
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		t.Run(quantizationName(kind), func(t *testing.T) {
			var reformer DenseReformer
			if kind != QuantizationFP16 {
				rotator, err := NewFHTRotatorFromSigns(4, []byte{0x13, 0x57, 0x9b, 0xdf})
				require.NoError(t, err)

				reformer, err = NewRotationReformer(rotator)
				require.NoError(t, err)
			}
			index, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, kind, reformer, candidates)
			require.NoError(t, err)

			resolve := func(key uint64) (string, bool) {
				return strconv.FormatUint(key%3, 10), key != 5
			}
			options := GroupByOptions{
				GroupCount: 3, TopKPerGroup: 2,
				Radius: 10, Filter: func(key uint64) bool { return key != 4 }, Resolve: resolve,
			}
			got, err := index.SearchGroups(context.Background(), query, options)
			require.NoError(t, err)

			scanned, err := index.SearchWithOptions(context.Background(), query, SearchOptions{
				TopK: len(candidates), Radius: options.Radius, Filter: options.Filter,
			})
			require.NoError(t, err)

			accumulator := newGroupAccumulator(MetricL2, options.TopKPerGroup)
			for _, result := range scanned {
				if value, found := resolve(result.Key); found {
					accumulator.add(value, result)
				}
			}
			want := accumulator.finish(options.GroupCount)
			require.Equal(t, want, got)
		})
	}
}

func TestHNSWRaBitQSearchGroupsMatchesLinearCodeScan(t *testing.T) {
	candidates := hnswRaBitQCandidates(24, 64)
	options := DefaultHNSWRaBitQBuildOptions(MetricL2)
	options.TotalBits, options.Clusters = 5, 4
	options.M, options.EFConstruction = 6, 24
	builder, err := NewHNSWRaBitQBuilder(64, options)
	require.NoError(t, err)

	for _, candidate := range candidates {
		{
			err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	query := candidates[11].Vector
	resolve := func(key uint64) (string, bool) {
		return strconv.FormatUint(key%4, 10), key != candidates[3].Key
	}
	groupOptions := GroupByOptions{
		GroupCount: 4, TopKPerGroup: 3,
		Filter: func(key uint64) bool { return key%5 != 0 }, Resolve: resolve,
	}
	got, err := index.SearchGroups(context.Background(), query, groupOptions)
	require.NoError(t, err)

	scanned, err := index.SearchHNSWRaBitQ(context.Background(), query, HNSWRaBitQSearchOptions{
		SearchOptions: SearchOptions{TopK: len(candidates), Filter: groupOptions.Filter},
		EF:            len(candidates), Linear: true,
	})
	require.NoError(t, err)

	accumulator := newGroupAccumulator(MetricL2, groupOptions.TopKPerGroup)
	for _, result := range scanned {
		if value, found := resolve(result.Key); found {
			accumulator.add(value, result)
		}
	}
	want := accumulator.finish(groupOptions.GroupCount)
	require.Equal(t, want, got)
}

func TestSparseFlatSearchGroups(t *testing.T) {
	index, err := NewSparseFlatIndex(MetricIP)
	require.NoError(t, err)

	for key := uint64(0); key < 6; key++ {
		vector := SparseVector{Indices: []uint32{1}, Values: []float32{float32(key)}}
		{
			err := index.AddSparse(context.Background(), key, vector)
			require.NoError(t, err)
		}
	}
	got, err := index.SearchSparseGroups(context.Background(), SparseVector{Indices: []uint32{1}, Values: []float32{1}}, GroupByOptions{
		GroupCount:   2,
		TopKPerGroup: 2,
		Resolve: func(key uint64) (string, bool) {
			return strconv.FormatUint(key%2, 10), true
		},
	})
	require.NoError(t, err)

	want := []GroupResult{
		{Value: "1", Results: []Result{{Key: 5, Score: 5}, {Key: 3, Score: 3}}},
		{Value: "0", Results: []Result{{Key: 4, Score: 4}, {Key: 2, Score: 2}}},
	}
	require.Equal(t, want, got)
}

func TestQueryDenseGroupsMergesBeforeTruncating(t *testing.T) {
	first, err := NewDenseFlatIndex(1, MetricIP)
	require.NoError(t, err)

	second, _ := NewDenseFlatIndex(1, MetricIP)
	for key, value := range []float32{0, 1} {
		_ = first.Add(context.Background(), uint64(key), []float32{value})
	}
	for key, value := range []float32{10, 11} {
		_ = second.Add(context.Background(), uint64(key+2), []float32{value})
	}
	resolver := func(key uint64) (string, bool) {
		if key < 2 {
			return "low", true
		}
		return "high", true
	}
	options := GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: resolver}
	want := []GroupResult{{Value: "high", Results: []Result{{Key: 3, Score: 11}}}}
	got, err := QueryDenseGroups(context.Background(), MetricIP, []DenseGroupSearcher{first, second}, []float32{1}, options, 2)
	require.NoError(t, err)
	require.Equal(t, want, got)

	reversed, err := QueryDenseGroups(context.Background(), MetricIP, []DenseGroupSearcher{second, first}, []float32{1}, options, 1)
	require.NoError(t, err)
	require.Equal(t, want, reversed)
}

func TestQuerySparseGroupsAndEmptySegments(t *testing.T) {
	first, _ := NewSparseFlatIndex(MetricIP)
	second, _ := NewSparseFlatIndex(MetricIP)
	_ = first.AddSparse(context.Background(), 1, SparseVector{Indices: []uint32{1}, Values: []float32{1}})
	_ = second.AddSparse(context.Background(), 2, SparseVector{Indices: []uint32{1}, Values: []float32{2}})
	resolver := func(key uint64) (string, bool) { return strconv.FormatUint(key, 10), true }
	options := GroupByOptions{GroupCount: 2, TopKPerGroup: 1, Resolve: resolver}
	got, err := QuerySparseGroups(context.Background(), []SparseGroupSearcher{first, second}, SparseVector{Indices: []uint32{1}, Values: []float32{1}}, options, 2)
	require.NoError(t, err)

	want := []GroupResult{
		{Value: "2", Results: []Result{{Key: 2, Score: 2}}},
		{Value: "1", Results: []Result{{Key: 1, Score: 1}}},
	}
	require.Equal(t, want, got)

	empty, err := QuerySparseGroups(context.Background(), nil, SparseVector{}, options, 0)
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
}

func TestGroupByQueryValidationAndCancellation(t *testing.T) {
	index, err := NewDenseFlatIndex(1, MetricL2)
	require.NoError(t, err)

	_ = index.Add(context.Background(), 1, []float32{1})
	resolver := func(uint64) (string, bool) { return "group", true }
	options := GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: resolver}
	{
		_, err := QueryDenseGroups(context.Background(), MetricIP, []DenseGroupSearcher{index}, []float32{1}, options, 1)
		require.Error(t, err,
			"mismatched dense group metric succeeded")
	}
	{
		_, err := QueryDenseGroups(context.Background(), Metric(99), nil, []float32{1}, options, 1)
		require.Error(t, err,
			"invalid dense group metric succeeded")
	}
	{
		_, err := QueryDenseGroups(context.Background(), MetricL2, []DenseGroupSearcher{nil}, []float32{1}, options, 1)
		require.Error(t, err,
			"nil dense group searcher succeeded")
	}
	{
		_, err := QuerySparseGroups(context.Background(), []SparseGroupSearcher{nil}, SparseVector{}, options, 1)
		require.Error(t, err,
			"nil sparse group searcher succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := index.SearchGroups(canceled, []float32{1}, options)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := QueryDenseGroups(canceled, MetricL2, []DenseGroupSearcher{index}, []float32{1}, options, 1)
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestMergeGroupResultsRebuildsPerGroupTopK(t *testing.T) {
	batches := [][]GroupResult{
		{{Value: "a", Results: []Result{{Key: 3, Score: 3}, {Key: 1, Score: 1}}}},
		{{Value: "b", Results: []Result{{Key: 4, Score: 4}}}, {Value: "a", Results: []Result{{Key: 5, Score: 5}}}},
	}
	want := []GroupResult{
		{Value: "a", Results: []Result{{Key: 5, Score: 5}, {Key: 3, Score: 3}}},
		{Value: "b", Results: []Result{{Key: 4, Score: 4}}},
	}
	{
		got := MergeGroupResults(MetricIP, 2, 2, batches...)
		require.Equal(t, want, got)
	}
	{
		got := MergeGroupResults(MetricIP, 0, 2, batches...)
		require.NotNil(t, got)
		require.Len(t, got, 0)
	}
}
