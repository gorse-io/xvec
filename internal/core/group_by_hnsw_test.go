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
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDenseHNSWSearchGroupsExpandsForDistinctGroups(t *testing.T) {
	index := denseGroupHNSWFixture()
	options := HNSWGroupSearchOptions{
		GroupByOptions: GroupByOptions{
			GroupCount: 2, TopKPerGroup: 2, Radius: 200,
			Filter: func(key uint64) bool { return key != 20 },
			Resolve: func(key uint64) (string, bool) {
				if key < 20 {
					return "near", true
				}
				return "far", true
			},
		},
		EF: 2, PrefetchOffset: 1, PrefetchLines: 2,
	}
	got, err := index.SearchHNSWGroups(context.Background(), []float32{0}, options)
	require.NoError(t, err)

	want := []GroupResult{
		{Value: "near", Results: []Result{{Key: 10, Score: 0}, {Key: 11, Score: float32(.1) * float32(.1)}}},
		{Value: "far", Results: []Result{{Key: 21, Score: 121}}},
	}
	require.Equal(t, want, got)
}

func TestScalarQuantizedHNSWSearchGroupsExpandsForDistinctGroups(t *testing.T) {
	index, err := NewScalarQuantizedHNSWIndex(context.Background(), denseGroupHNSWFixture(), QuantizationInt8, nil)
	require.NoError(t, err)

	got, err := index.SearchHNSWGroups(context.Background(), []float32{0}, HNSWGroupSearchOptions{
		GroupByOptions: GroupByOptions{
			GroupCount: 2, TopKPerGroup: 1,
			Resolve: func(key uint64) (string, bool) {
				if key < 20 {
					return "near", true
				}
				return "far", true
			},
		},
		EF: 1,
	})
	require.NoError(t, err)
	require.Len(t, got, 2)
	require.True(t, got[0].Value == "near")
	require.True(t, got[1].Value == "far")
	require.Len(t, got[0].Results, 1)
	require.Len(t, got[1].Results, 1)
}

func TestSparseHNSWSearchGroupsExpandsForDistinctGroups(t *testing.T) {
	keys := []uint64{10, 11, 12, 13, 20, 21}
	index := &SparseHNSWIndex{
		options: DefaultSparseHNSWBuildOptions(),
		keys:    keys, offsets: []int{0, 1, 2, 3, 4, 5, 6},
		indices: []uint32{0, 0, 0, 0, 0, 0}, values: []float32{10, 9, 8, 7, 1, 0},
		positions: map[uint64]int{10: 0, 11: 1, 12: 2, 13: 3, 20: 4, 21: 5},
		levels:    []int{0, 0, 0, 0, 0, 0}, neighbors: groupHNSWChain(6),
		entryPoint: 0, maxLevel: 0,
	}
	got, err := index.SearchSparseHNSWGroups(context.Background(), SparseVector{
		Indices: []uint32{0}, Values: []float32{1},
	}, HNSWGroupSearchOptions{
		GroupByOptions: GroupByOptions{
			GroupCount: 2, TopKPerGroup: 2,
			Resolve: func(key uint64) (string, bool) {
				if key < 20 {
					return "hot", true
				}
				return "cold", true
			},
		},
		EF: 2, PrefetchOffset: 1, PrefetchLines: 2,
	})
	require.NoError(t, err)

	want := []GroupResult{
		{Value: "hot", Results: []Result{{Key: 10, Score: 10}, {Key: 11, Score: 9}}},
		{Value: "cold", Results: []Result{{Key: 20, Score: 1}}},
	}
	require.Equal(t, want, got)
}

func TestHNSWRaBitQSearchGroupsUsesGraphTraversal(t *testing.T) {
	ctx := context.Background()
	buildOptions := DefaultHNSWRaBitQBuildOptions(MetricL2)
	buildOptions.TotalBits = 5
	buildOptions.Clusters = 1
	buildOptions.SampleCount = 8
	buildOptions.MaxIterations = 2
	buildOptions.M = 2
	buildOptions.EFConstruction = 8
	builder, err := NewHNSWRaBitQBuilder(MinRaBitQDimension, buildOptions)
	require.NoError(t, err)

	for position := 0; position < 8; position++ {
		vector := make([]float32, MinRaBitQDimension)
		value := float32(position) / 10
		if position >= 6 {
			value = float32(position-5) * 10
		}
		for dimension := range vector {
			vector[dimension] = value + float32(dimension%3)/1000
		}
		{
			err := builder.Add(ctx, uint64(position+1), vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(ctx)
	require.NoError(t, err)

	groups, err := index.SearchHNSWRaBitQGroups(ctx, make([]float32, MinRaBitQDimension), HNSWGroupSearchOptions{
		GroupByOptions: GroupByOptions{
			GroupCount: 2, TopKPerGroup: 2,
			Resolve: func(key uint64) (string, bool) {
				if key <= 6 {
					return "near", true
				}
				return "far", true
			},
		},
		EF: 4,
	})
	require.NoError(t, err)
	require.Len(t, groups, 2)
	require.True(t, groups[0].Value == "near")
	require.True(t, groups[1].Value == "far")
}

func TestHNSWGroupSearchValidationAndCancellation(t *testing.T) {
	resolver := func(uint64) (string, bool) { return "group", true }
	{
		err := (HNSWGroupSearchOptions{
			GroupByOptions: GroupByOptions{GroupCount: maxPlatformInt(), TopKPerGroup: 2, Resolve: resolver}, EF: 1,
		}).Validate()
		require.ErrorIs(t, err, ErrGroupSizeOverflow)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := denseGroupHNSWFixture().SearchHNSWGroups(ctx, []float32{0}, HNSWGroupSearchOptions{
			GroupByOptions: GroupByOptions{GroupCount: 1, TopKPerGroup: 1, Resolve: resolver}, EF: 1,
		})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func denseGroupHNSWFixture() *HNSWIndex {
	keys := []uint64{10, 11, 12, 13, 20, 21}
	return &HNSWIndex{
		dimension: 1, options: DefaultHNSWBuildOptions(MetricL2),
		keys: keys, vectors: []float32{0, .1, .2, .3, 10, 11},
		positions: map[uint64]int{10: 0, 11: 1, 12: 2, 13: 3, 20: 4, 21: 5},
		levels:    []int{0, 0, 0, 0, 0, 0}, neighbors: groupHNSWChain(6),
		entryPoint: 0, maxLevel: 0,
	}
}

func groupHNSWChain(count int) [][][]int {
	neighbors := make([][][]int, count)
	for position := range neighbors {
		levelZero := make([]int, 0, 2)
		if position > 0 {
			levelZero = append(levelZero, position-1)
		}
		if position+1 < count {
			levelZero = append(levelZero, position+1)
		}
		neighbors[position] = [][]int{levelZero}
	}
	return neighbors
}
