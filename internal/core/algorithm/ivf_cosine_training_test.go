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
	"encoding/binary"
	"path/filepath"
	"testing"

	hashutil "github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/stretchr/testify/require"
)

func TestCosineIVFTrainsNormalizedCentroidsAndKeepsOriginals(t *testing.T) {
	t.Parallel()
	for _, fp16 := range []bool{false, true} {
		options := DefaultIVFBuildOptions(MetricCosine)
		options.NList = 1
		builder, err := newIVFBuilder(2, options, fp16)
		require.NoError(t, err)
		require.NoError(t, builder.Add(context.Background(), 1, []float32{3, 0}))
		require.NoError(t, builder.Add(context.Background(), 2, []float32{0, 4}))
		index, err := builder.Build(context.Background())
		require.NoError(t, err)
		require.True(t, index.cosineDotRouting)
		require.Equal(t, [][]float32{{0.5, 0.5}}, index.Centroids())
		original, ok := index.Vector(1)
		require.True(t, ok)
		require.Equal(t, []float32{3, 0}, original)
		got, err := index.Search(context.Background(), []float32{9, 0}, 2)
		require.NoError(t, err)
		want, err := TopK(context.Background(), MetricCosine, []float32{9, 0}, []Candidate{
			{Key: 1, Vector: []float32{3, 0}}, {Key: 2, Vector: []float32{0, 4}},
		}, 2)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
}

// The first centroid wins by dot product; the second wins by true cosine.
// This distinguishes the routing modes without relying on random training.
func cosineRoutingFixture(t *testing.T, dotRouting bool) *IVFIndex {
	t.Helper()
	options := DefaultIVFBuildOptions(MetricCosine)
	options.NList = 2
	index := &IVFIndex{
		dimension: 2, options: options, cosineDotRouting: dotRouting,
		keys: []uint64{10, 20}, vectors: []float32{1, 1, 0, 1},
		positions:       map[uint64]int{10: 0, 20: 1},
		lists:           []ivfList{{positions: []int{0}}, {positions: []int{1}}},
		listForPosition: []int{0, 1},
		model: &KMeansModel{metric: MetricCosine, dimension: 2,
			centroids: [][]float32{{0.9, 0}, {0.3, 0.3}}, counts: []int{1, 1}},
	}
	require.NoError(t, index.cacheCosineMagnitudes(context.Background()))
	return index
}

func TestCosineIVFRoutingPersistsAndAssignsIncrementalVectors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	for _, dotRouting := range []bool{false, true} {
		index := cosineRoutingFixture(t, dotRouting)
		expectedList := 1
		expectedVersion := uint16(1)
		if dotRouting {
			expectedList, expectedVersion = 0, 2
		}
		encoded, err := encodeIVFIndex(ctx, index)
		require.NoError(t, err)
		require.Equal(t, expectedVersion, binary.LittleEndian.Uint16(encoded[8:10]))
		reopened, err := decodeIVFIndex(ctx, encoded)
		require.NoError(t, err)
		require.Equal(t, dotRouting, reopened.cosineDotRouting)
		for _, current := range []*IVFIndex{index, reopened} {
			for _, query := range [][]float32{{1, 1}, {10, 10}} {
				lists, err := current.ProbedLists(ctx, query, 1)
				require.NoError(t, err)
				require.Equal(t, []int{expectedList}, lists)
				got, err := current.SearchIVF(ctx, query, IVFSearchOptions{SearchOptions: SearchOptions{TopK: 1}, NProbe: 1})
				require.NoError(t, err)
				require.Equal(t, current.keys[expectedList], got[0].Key)
			}
			require.NoError(t, current.Add(ctx, 30, []float32{10, 10}))
			list, ok := current.ListForKey(30)
			require.True(t, ok)
			require.Equal(t, expectedList, list)
			snapshot, err := current.persistenceSnapshot(ctx)
			require.NoError(t, err)
			require.Equal(t, dotRouting, snapshot.cosineDotRouting)
		}
	}
}

func TestCosineIVFRoutingSurvivesScalarQuantization(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	base := cosineRoutingFixture(t, true)
	quantized, err := NewScalarQuantizedIVFIndex(ctx, base, QuantizationInt8, nil)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "cosine.ivf")
	require.NoError(t, quantized.Save(ctx, path))
	reopened, err := OpenScalarQuantizedIVFIndex(ctx, path, QuantizationInt8, nil)
	require.NoError(t, err)
	for _, index := range []*ScalarQuantizedIVFIndex{quantized, reopened} {
		require.True(t, index.base.cosineDotRouting)
		results, err := index.SearchIVF(ctx, []float32{1, 1}, IVFSearchOptions{SearchOptions: SearchOptions{TopK: 1}, NProbe: 1})
		require.NoError(t, err)
		require.Len(t, results, 1)
		require.Equal(t, uint64(10), results[0].Key)
	}
}

func TestCosineIVFOnlineBootstrapNormalizesCentroids(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	options := DefaultIVFBuildOptions(MetricCosine)
	options.NList = 2
	builder, err := NewIVFBuilder(2, options)
	require.NoError(t, err)
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	encoded, err := encodeIVFIndex(ctx, index)
	require.NoError(t, err)
	index, err = decodeIVFIndex(ctx, encoded)
	require.NoError(t, err)
	require.True(t, index.cosineDotRouting)
	require.NoError(t, index.Add(ctx, 1, []float32{3, 0}))
	require.NoError(t, index.Add(ctx, 2, []float32{0, 4}))
	require.Equal(t, [][]float32{{1, 0}, {0, 1}}, index.Centroids())
	require.NoError(t, index.Add(ctx, 3, []float32{1, 2}))
	list, ok := index.ListForKey(3)
	require.True(t, ok)
	require.Equal(t, 1, list)
}

func TestIVFRejectsInvalidRoutingMetadata(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	encoded, err := encodeIVFIndex(ctx, cosineRoutingFixture(t, true))
	require.NoError(t, err)
	for _, change := range []func([]byte){
		func(b []byte) { b[51] = 2 },
		func(b []byte) { binary.LittleEndian.PutUint16(b[8:10], 1) },
		func(b []byte) { b[48] = byte(MetricL2) },
	} {
		corrupted := append([]byte(nil), encoded...)
		change(corrupted)
		binary.LittleEndian.PutUint32(corrupted[108:112], hashutil.CRC32C(corrupted[:108]))
		_, err := decodeIVFIndex(ctx, corrupted)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
}

func TestCosineIVFBuildAssignmentsMatchRouting(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	var baseline *IVFIndex
	for _, workers := range []int{1, 8} {
		options := DefaultIVFBuildOptions(MetricCosine)
		options.NList, options.Workers = 7, workers
		builder, err := NewIVFBuilder(3, options)
		require.NoError(t, err)
		for j := range 64 {
			require.NoError(t, builder.Add(ctx, uint64(j), []float32{
				float32(j%11) - 5, float32(j%7) - 3, float32(j%3) - 1,
			}))
		}
		index, err := builder.Build(ctx)
		require.NoError(t, err)
		for _, key := range index.keys {
			vector, ok := index.Vector(key)
			require.True(t, ok)
			probes, err := index.ProbedLists(ctx, vector, 1)
			require.NoError(t, err)
			list, ok := index.ListForKey(key)
			require.True(t, ok)
			require.Equal(t, list, probes[0])
		}
		if baseline != nil {
			require.Equal(t, baseline.Centroids(), index.Centroids())
			require.Equal(t, baseline.listForPosition, index.listForPosition)
		}
		baseline = index
	}
}
