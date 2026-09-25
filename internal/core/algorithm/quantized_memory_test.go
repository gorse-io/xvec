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
	for _, kind := range []Quantization{QuantizationInt8, QuantizationInt4} {
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
