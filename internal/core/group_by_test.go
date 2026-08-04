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
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

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
