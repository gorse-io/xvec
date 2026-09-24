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
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// Interleave lists as in a pre-packing file, including empty lists, zero
// vectors, ties, and non-unit vectors. No training changes are involved.
func interleavedCosineIVF(t *testing.T) *IVFIndex {
	t.Helper()
	index := cosineRoutingFixture(t, true)
	index.dimension = 17
	index.options.NList = 5
	index.keys, index.vectors = nil, nil
	index.positions = make(map[uint64]int)
	index.listForPosition = nil
	index.lists = make([]ivfList, 5)
	index.model = &KMeansModel{metric: MetricCosine, dimension: 17,
		centroids: make([][]float32, 5), counts: make([]int, 5)}
	rng := rand.New(rand.NewPCG(17, 23))
	for position := range 259 {
		key := uint64(1000 - position)
		vector := make([]float32, 17)
		for component := range vector {
			if position%7 != 0 {
				vector[component] = rng.Float32()*6 - 3
			}
		}
		cluster := position % 4
		index.keys = append(index.keys, key)
		index.vectors = append(index.vectors, vector...)
		index.positions[key] = position
		index.listForPosition = append(index.listForPosition, cluster)
		index.lists[cluster].positions = append(index.lists[cluster].positions, position)
		index.model.counts[cluster]++
		if position < 5 {
			index.model.centroids[position] = vector
		}
	}
	require.NoError(t, index.cacheCosineMagnitudes(context.Background()))
	return index
}

func requireSameCosineContents(t *testing.T, want, got *IVFIndex) {
	t.Helper()
	require.Equal(t, want.model, got.model)
	for cluster := range want.lists {
		before, err := want.List(cluster)
		require.NoError(t, err)
		after, err := got.List(cluster)
		require.NoError(t, err)
		require.Equal(t, before, after)
	}
	for _, key := range want.keys {
		before, ok := want.Vector(key)
		require.True(t, ok)
		after, ok := got.Vector(key)
		require.True(t, ok)
		for component := range before {
			require.Equal(t, math.Float32bits(before[component]), math.Float32bits(after[component]))
		}
		beforeList, ok := want.ListForKey(key)
		require.True(t, ok)
		afterList, ok := got.ListForKey(key)
		require.True(t, ok)
		require.Equal(t, beforeList, afterList)
	}
	ctx := context.Background()
	for _, key := range want.keys[:12] {
		query, _ := want.Vector(key)
		for _, nprobe := range []int{1, 3, 5} {
			for _, search := range []SearchOptions{
				{TopK: 1}, {TopK: 100}, {TopK: 300}, {TopK: 100, Radius: 0.6},
				{TopK: 100, Filter: func(key uint64) bool { return key%3 == 0 }},
			} {
				options := IVFSearchOptions{SearchOptions: search, NProbe: nprobe}
				before, err := want.SearchIVF(ctx, query, options)
				require.NoError(t, err)
				after, err := got.SearchIVF(ctx, query, options)
				require.NoError(t, err)
				require.Equal(t, before, after, "query=%d nprobe=%d", key, nprobe)
			}
		}
	}
}

func TestCosineIVFListPackingPreservesExactResults(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	original := interleavedCosineIVF(t)
	encoded, err := encodeIVFIndex(ctx, original)
	require.NoError(t, err)
	packed, err := decodeIVFIndex(ctx, encoded)
	require.NoError(t, err)
	require.NotEqual(t, original.keys, packed.keys)
	requireSameCosineContents(t, original, packed)

	// Already-packed reopen requires no permutation or new vector buffer.
	storage := &packed.vectors[0]
	require.NoError(t, packed.packCosineLists(ctx))
	require.Same(t, storage, &packed.vectors[0])

	// Append to earlier lists without overwriting subsequent packed lists.
	for _, key := range []uint64{2000, 2001, 2002} {
		vector, _ := original.Vector(3000 - key)
		require.NoError(t, original.Add(ctx, key, vector))
		require.NoError(t, packed.Add(ctx, key, vector))
	}
	requireSameCosineContents(t, original, packed)
	encoded, err = encodeIVFIndex(ctx, packed)
	require.NoError(t, err)
	reopened, err := decodeIVFIndex(ctx, encoded)
	require.NoError(t, err)
	requireSameCosineContents(t, original, reopened)

	quantized, err := NewScalarQuantizedIVFIndex(ctx, packed, QuantizationInt8, nil)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "packed.ivf")
	require.NoError(t, quantized.Save(ctx, path))
	quantizedReopened, err := OpenScalarQuantizedIVFIndex(ctx, path, QuantizationInt8, nil)
	require.NoError(t, err)
	query, _ := original.Vector(997)
	options := IVFSearchOptions{SearchOptions: SearchOptions{TopK: 100}, NProbe: 3}
	before, err := quantized.SearchIVF(ctx, query, options)
	require.NoError(t, err)
	after, err := quantizedReopened.SearchIVF(ctx, query, options)
	require.NoError(t, err)
	require.Equal(t, before, after)
}

type cancelPackingContext struct {
	context.Context
	remaining int
}

func (c *cancelPackingContext) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}

func TestCosineIVFListPackingCancellationIsAtomic(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	index := interleavedCosineIVF(t)
	before, err := index.persistenceSnapshot(ctx)
	require.NoError(t, err)
	// Cancel after copying, before publishing the new arrays.
	err = index.packCosineLists(&cancelPackingContext{Context: ctx, remaining: 3})
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, before.keys, index.keys)
	requireSameCosineContents(t, before, index)
	require.NoError(t, index.packCosineLists(ctx))
	require.NoError(t, index.cacheCosineMagnitudes(ctx))
	requireSameCosineContents(t, before, index)
}

func TestCosineIVFBuildPacksLists(t *testing.T) {
	t.Parallel()
	source := interleavedCosineIVF(t)
	candidates := make([]Candidate, 0, source.Len())
	for _, key := range source.keys {
		vector, ok := source.Vector(key)
		require.True(t, ok)
		candidates = append(candidates, Candidate{Key: key, Vector: vector})
	}
	index := buildSearchIVF(t, MetricCosine, candidates, 5)
	next := 0
	for _, list := range index.lists {
		previousInput := -1
		for _, position := range list.positions {
			require.Equal(t, next, position)
			key := index.keys[position]
			input := source.positions[key]
			require.Greater(t, input, previousInput, "preserve insertion order inside each list")
			previousInput = input
			vector, ok := index.Vector(key)
			require.True(t, ok)
			require.Equal(t, candidates[input].Vector, vector)
			next++
		}
	}
	require.Equal(t, source.Len(), next)
}

func TestCosineIVFListPackingLeavesLegacyAndFP16LayoutsAlone(t *testing.T) {
	t.Parallel()
	for _, fp16 := range []bool{false, true} {
		index := interleavedCosineIVF(t)
		index.cosineDotRouting = fp16
		index.fp16 = fp16
		keys := append([]uint64(nil), index.keys...)
		storage := &index.vectors[0]
		require.NoError(t, index.packCosineLists(context.Background()))
		require.Equal(t, keys, index.keys)
		require.Same(t, storage, &index.vectors[0])
	}
}
