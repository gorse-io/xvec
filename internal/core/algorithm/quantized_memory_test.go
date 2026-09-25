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
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestQuantizedHNSWSharedStorageAndSnapshotIsolation(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			candidates := quantizedIndexCandidates(80)
			base := buildDenseHNSWFromCandidates(t, candidates)
			index, err := NewScalarQuantizedHNSWIndex(ctx, base, kind, nil)
			require.NoError(t, err)
			require.Same(t, &index.base.vectors[0], &index.vectors.originals[0])
			require.Same(t, &index.base.keys[0], &index.vectors.keys[0])
			require.NotSame(t, &base.vectors[0], &index.vectors.originals[0])
			require.Same(t, index.vectors, index.FlatIndex().vectors)

			query := slices.Clone(candidates[7].Vector)
			options := SearchOptions{TopK: 20, Filter: func(key uint64) bool { return key%3 != 0 }}
			want := exactQuantizedResults(t, kind, MetricL2, nil, query, candidates, options)
			// Mutations of the live source graph and caller-owned inputs must not
			// affect either the immutable graph or its shared Flat view.
			original := slices.Clone(candidates[0].Vector)
			base.vectors[0] += 100
			candidates[0].Vector[0] += 200
			gotOriginal, found := index.Vector(candidates[0].Key)
			require.True(t, found)
			require.Equal(t, original, gotOriginal)
			gotOriginal[0] += 300
			again, _ := index.FlatIndex().Vector(candidates[0].Key)
			require.Equal(t, original, again)
			got, err := index.FlatIndex().SearchWithOptions(ctx, query, options)
			require.NoError(t, err)
			require.Equal(t, want, got)

			path := filepath.Join(t.TempDir(), "hnsw")
			require.NoError(t, index.Save(ctx, path))
			reopened, err := OpenScalarQuantizedHNSWIndex(ctx, path, kind, nil)
			require.NoError(t, err)
			require.Same(t, &reopened.base.vectors[0], &reopened.vectors.originals[0])
			require.Same(t, &reopened.base.keys[0], &reopened.vectors.keys[0])
			require.Equal(t, index.base.neighbors, reopened.base.neighbors)
			got, err = reopened.FlatIndex().SearchWithOptions(ctx, query, options)
			require.NoError(t, err)
			require.Equal(t, want, got)
			require.NoError(t, reopened.Save(ctx, filepath.Join(t.TempDir(), "resaved")))
		})
	}
}

func TestQuantizedHNSWBuilderTransfersOriginals(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			candidates := quantizedIndexCandidates(40)
			builder, err := NewHNSWBuilder(4, DefaultHNSWBuildOptions(MetricL2))
			require.NoError(t, err)
			require.NoError(t, builder.Reserve(len(candidates)))
			for _, c := range candidates {
				require.NoError(t, builder.Add(ctx, c.Key, c.Vector))
			}
			owned := &builder.vectors[0]
			index, err := builder.buildScalarQuantizedWithWorkers(ctx, 2, kind, nil)
			require.NoError(t, err)
			require.Same(t, owned, &index.base.vectors[0])
			require.Same(t, owned, &index.vectors.originals[0])
			require.Nil(t, builder.vectors)
			require.ErrorIs(t, builder.Add(ctx, 999, []float32{1, 2, 3, 4}), ErrBuilderClosed)
		})
	}
}

func TestQuantizedFlatStillOwnsCallerInput(t *testing.T) {
	candidates := quantizedIndexCandidates(4)
	original := slices.Clone(candidates[0].Vector)
	index, err := NewScalarQuantizedFlatIndex(context.Background(), 4, MetricL2, QuantizationInt4, nil, candidates)
	require.NoError(t, err)
	key := candidates[0].Key
	candidates[0].Vector[0] += 100
	candidates[0].Key = 9999
	got, found := index.Vector(key)
	require.True(t, found)
	require.Equal(t, original, got)
}

func TestQuantizedHNSWBorrowedOriginals(t *testing.T) {
	ctx := context.Background()
	for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			candidates := quantizedIndexCandidates(80)
			options := DefaultHNSWBuildOptions(MetricCosine)
			workers := 2
			if kind == QuantizationFP16 {
				workers = 1
			}
			index, err := BuildScalarQuantizedHNSWWithBorrowedVectors(ctx, 4, options, kind, nil, candidates, workers)
			require.NoError(t, err)
			require.Nil(t, index.base.vectors)
			require.Nil(t, index.vectors.originals)
			require.Same(t, &candidates[0].Vector[0], &index.base.vectorRows[0][0])
			query := slices.Clone(candidates[7].Vector)
			want, err := index.Search(ctx, query, 20)
			require.NoError(t, err)
			path := filepath.Join(t.TempDir(), "borrowed")
			require.NoError(t, index.Save(ctx, path))
			owned, err := OpenScalarQuantizedHNSWIndex(ctx, path, kind, nil)
			require.NoError(t, err)
			// External vectors may be supplied in a different key order.
			slices.Reverse(candidates)
			reopened, err := OpenScalarQuantizedHNSWIndexWithBorrowedVectors(ctx, path, kind, nil, candidates, false)
			require.NoError(t, err)
			require.Nil(t, reopened.base.vectors)
			require.Nil(t, reopened.vectors.originals)
			mapped, err := OpenScalarQuantizedHNSWIndexWithBorrowedVectors(ctx, path, kind, nil, candidates, true)
			require.NoError(t, err)
			for _, source := range []*ScalarQuantizedHNSWIndex{owned, reopened, mapped} {
				got, err := source.Search(ctx, query, 20)
				require.NoError(t, err)
				require.Equal(t, want, got)
				vector, found := source.FlatIndex().Vector(candidates[0].Key)
				require.True(t, found)
				require.Equal(t, candidates[0].Vector, vector)
				vector[0] += 100
				again, _ := source.Vector(candidates[0].Key)
				require.Equal(t, candidates[0].Vector, again)
			}
			resaved := filepath.Join(t.TempDir(), "resaved")
			require.NoError(t, reopened.Save(ctx, resaved))
			originalBytes, err := os.ReadFile(path)
			require.NoError(t, err)
			resavedBytes, err := os.ReadFile(resaved)
			require.NoError(t, err)
			require.Equal(t, originalBytes, resavedBytes)
			// Copying a graph with external rows must restore independent storage.
			cloned, err := NewScalarQuantizedHNSWIndex(ctx, reopened.base, kind, nil)
			require.NoError(t, err)
			require.Nil(t, cloned.base.vectorRows)
			key, original := candidates[0].Key, slices.Clone(candidates[0].Vector)
			candidates[0].Key, candidates[0].Vector = 99999, nil
			got, found := reopened.Vector(key)
			require.True(t, found)
			require.Equal(t, original, got, "candidate headers must not be retained")
		})
	}
}

func TestQuantizedHNSWBorrowedOriginalValidation(t *testing.T) {
	ctx := context.Background()
	candidates := quantizedIndexCandidates(8)
	index, err := BuildScalarQuantizedHNSWWithBorrowedVectors(ctx, 4, DefaultHNSWBuildOptions(MetricL2), QuantizationInt4, nil, candidates, 1)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "index")
	require.NoError(t, index.Save(ctx, path))
	for _, change := range []func([]Candidate) []Candidate{
		func(c []Candidate) []Candidate { return c[:len(c)-1] },
		func(c []Candidate) []Candidate { c[0].Key = 99999; return c },
		func(c []Candidate) []Candidate { c[0].Key = c[1].Key; return c },
		func(c []Candidate) []Candidate { c[0].Vector = c[0].Vector[:3]; return c },
		func(c []Candidate) []Candidate { c[0].Vector = []float32{100, 200, 300, 400}; return c },
	} {
		_, err := OpenScalarQuantizedHNSWIndexWithBorrowedVectors(ctx, path, QuantizationInt4, nil, change(slices.Clone(candidates)), false)
		require.Error(t, err)
	}
	_, err = OpenScalarQuantizedHNSWIndexWithBorrowedVectors(nil, path, QuantizationInt4, nil, candidates, false)
	require.Error(t, err)
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = OpenScalarQuantizedHNSWIndexWithBorrowedVectors(canceled, path, QuantizationInt4, nil, candidates, false)
	require.ErrorIs(t, err, context.Canceled)
	_, err = BuildScalarQuantizedHNSWWithBorrowedVectors(canceled, 4, DefaultHNSWBuildOptions(MetricL2), QuantizationInt4, nil, candidates, 1)
	require.ErrorIs(t, err, context.Canceled)
	_, err = BuildScalarQuantizedHNSWWithBorrowedVectors(ctx, 4, DefaultHNSWBuildOptions(MetricL2), QuantizationInt4, nil, append(slices.Clone(candidates), candidates[0]), 1)
	require.ErrorIs(t, err, ErrDuplicateKey)
	_, err = BuildScalarQuantizedHNSWWithBorrowedVectors(ctx, 4, DefaultHNSWBuildOptions(MetricL2), QuantizationInt4, nil, []Candidate{{Key: 1, Vector: []float32{1}}}, 1)
	require.Error(t, err)
	encoded, err := os.ReadFile(path)
	require.NoError(t, err)
	encoded[len(encoded)-1] ^= 1
	corrupt := filepath.Join(t.TempDir(), "corrupt")
	require.NoError(t, os.WriteFile(corrupt, encoded, 0o600))
	for _, useMmap := range []bool{false, true} {
		_, err := OpenScalarQuantizedHNSWIndexWithBorrowedVectors(ctx, corrupt, QuantizationInt4, nil, candidates, useMmap)
		require.ErrorIs(t, err, ErrHNSWChecksumMismatch)
		_, err = OpenScalarQuantizedHNSWIndexWithBorrowedVectors(ctx, filepath.Join(t.TempDir(), "missing"), QuantizationInt4, nil, candidates, useMmap)
		require.Error(t, err)
	}
	var empty *ScalarQuantizedHNSWIndex
	require.Nil(t, empty.FlatIndex())
}
