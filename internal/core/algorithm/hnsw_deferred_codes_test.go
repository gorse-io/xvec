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
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDeferredFP16FilteredScansAndGraphPublication(t *testing.T) {
	ctx := context.Background()
	candidates := hnswMemoryCandidates()
	originals := make(map[uint64][]byte, len(candidates))
	for _, c := range candidates {
		for _, v := range c.Vector {
			originals[c.Key] = binary.LittleEndian.AppendUint32(originals[c.Key], math.Float32bits(v))
		}
	}
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		for _, rotate := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/rotate=%v", metric, rotate), func(t *testing.T) {
				options := DefaultHNSWBuildOptions(metric)
				options.M, options.EFConstruction = 4, 16
				var reformer DenseReformer
				if rotate {
					rotator, err := NewFHTRotatorFromSigns(8, []byte{1, 22, 43, 64})
					require.NoError(t, err)
					reformer, err = NewRotationReformer(rotator)
					require.NoError(t, err)
				}
				eager, err := BuildScalarQuantizedHNSWWithBorrowedVectors(ctx, 8, options, QuantizationFP16, reformer, candidates, 1)
				require.NoError(t, err)
				path := filepath.Join(t.TempDir(), "index.hnsw")
				require.NoError(t, eager.Save(ctx, path))
				artifact, err := os.ReadFile(path)
				require.NoError(t, err)
				index, err := OpenScalarQuantizedHNSWIndexWithDeferredFP16Codes(ctx, path, reformer, originals, true)
				require.NoError(t, err)
				t.Cleanup(func() { require.NoError(t, index.Close()) })
				require.Nil(t, index.vectors.codes)
				require.Nil(t, index.vectors.mappedCodes)
				flat := index.FlatIndex()
				query := candidates[5].Vector
				keys := []uint64{candidates[1].Key, candidates[3].Key, candidates[5].Key, candidates[1].Key, math.MaxUint64}
				search := SearchOptions{TopK: 3, Filter: func(key uint64) bool { return key != candidates[3].Key }}
				want, err := eager.FlatIndex().SearchKeysWithOptions(ctx, query, keys, search)
				require.NoError(t, err)
				got, err := flat.SearchKeysWithOptions(ctx, query, keys, search)
				require.NoError(t, err)
				require.Equal(t, want, got)
				grouped := GroupByOptions{GroupCount: 3, TopKPerGroup: 2, Filter: search.Filter, Resolve: func(key uint64) (string, bool) { return fmt.Sprint(key % 4), true }}
				wantGroups, err := eager.FlatIndex().SearchGroups(ctx, query, grouped)
				require.NoError(t, err)
				gotGroups, err := flat.SearchGroups(ctx, query, grouped)
				require.NoError(t, err)
				require.Equal(t, wantGroups, gotGroups)
				require.Nil(t, index.vectors.codes, "Flat and group scans must not populate the whole code arena")
				canceled, cancel := context.WithCancel(ctx)
				cancel()
				_, err = index.Search(canceled, query, 4)
				require.ErrorIs(t, err, context.Canceled)
				require.Nil(t, index.vectors.codes)
				graphOptions := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 8}, EF: 32}
				wantGraph, err := eager.SearchHNSW(ctx, query, graphOptions)
				require.NoError(t, err)
				var wg sync.WaitGroup
				errs := make([]error, 16)
				results := make([][]Result, 16)
				for n := range results {
					wg.Go(func() {
						if n%2 == 0 {
							results[n], errs[n] = index.SearchHNSW(ctx, query, graphOptions)
						} else {
							results[n], errs[n] = flat.SearchKeysWithOptions(ctx, query, keys, search)
						}
					})
				}
				wg.Wait()
				for n := range results {
					require.NoError(t, errs[n])
					if n%2 == 0 {
						require.Equal(t, wantGraph, results[n])
					} else {
						require.Equal(t, want, results[n])
					}
				}
				require.Len(t, index.vectors.mappedCodes, len(candidates)*8*2)
				require.Equal(t, eager.vectors.codes, index.vectors.codes)
				saved := filepath.Join(t.TempDir(), "saved.hnsw")
				require.NoError(t, index.Save(ctx, saved))
				savedBytes, err := os.ReadFile(saved)
				require.NoError(t, err)
				require.Equal(t, artifact, savedBytes)
				require.NoError(t, index.Close())
				_, err = flat.SearchKeysWithOptions(ctx, query, keys, search)
				require.ErrorIs(t, err, ErrScalarQuantizedIndexClosed)
				_, err = index.Search(ctx, query, 4)
				require.ErrorIs(t, err, ErrScalarQuantizedIndexClosed)
			})
		}
	}
}

func TestDeferredFP16RejectsInvalidOriginalsAndArtifact(t *testing.T) {
	ctx := context.Background()
	candidates := hnswMemoryCandidates()
	options := DefaultHNSWBuildOptions(MetricL2)
	options.M, options.EFConstruction = 4, 16
	// FP32 can persist a value that overflows FP16; deferred open must still
	// reject it immediately, before any candidate is queried.
	candidates[0].Vector[0] = 1e10
	builder, err := NewHNSWBuilder(8, options)
	require.NoError(t, err)
	for _, c := range candidates {
		require.NoError(t, builder.Add(ctx, c.Key, c.Vector))
	}
	base, err := builder.Build(ctx)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "index.hnsw")
	require.NoError(t, base.Save(ctx, path))
	originals := make(map[uint64][]byte, len(candidates))
	for _, c := range candidates {
		for _, v := range c.Vector {
			originals[c.Key] = binary.LittleEndian.AppendUint32(originals[c.Key], math.Float32bits(v))
		}
	}
	_, err = OpenScalarQuantizedHNSWIndexWithDeferredFP16Codes(ctx, path, nil, originals, true)
	require.ErrorIs(t, err, ErrQuantizationOverflow)
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	data[len(data)-1] ^= 1
	require.NoError(t, os.WriteFile(path, data, 0600))
	_, err = OpenScalarQuantizedHNSWIndexWithDeferredFP16Codes(ctx, path, nil, originals, true)
	require.ErrorIs(t, err, ErrHNSWChecksumMismatch)
}
