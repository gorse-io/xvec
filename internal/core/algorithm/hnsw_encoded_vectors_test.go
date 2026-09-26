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
	"github.com/stretchr/testify/require"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func hnswMemoryCandidates() []Candidate {
	candidates := make([]Candidate, DefaultHNSWBruteForceThreshold+32)
	for n := range candidates {
		vector := make([]float32, 8)
		for d := range vector {
			vector[d] = float32(math.Sin(float64(n*7 + d)))
		}
		candidates[n] = Candidate{Key: uint64(n*3 + 1), Vector: vector}
	}
	return candidates
}

func TestHNSWEncodedOriginals(t *testing.T) {
	ctx := context.Background()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		for _, kind := range []Quantization{QuantizationFP16, QuantizationInt8, QuantizationInt4} {
			t.Run(fmt.Sprintf("%v/%v", metric, kind), func(t *testing.T) {
				candidates := hnswMemoryCandidates()
				options := DefaultHNSWBuildOptions(metric)
				options.M = 4
				options.EFConstruction = 16
				rotator, err := NewFHTRotatorFromSigns(8, []byte{1, 22, 43, 64})
				require.NoError(t, err)
				reformer, err := NewRotationReformer(rotator)
				require.NoError(t, err)
				original, err := BuildScalarQuantizedHNSWWithBorrowedVectors(ctx, 8, options, kind, reformer, candidates, 1)
				require.NoError(t, err)
				path := filepath.Join(t.TempDir(), "index.hnsw")
				require.NoError(t, original.Save(ctx, path))
				originals := make(map[uint64][]byte, len(candidates))
				for _, c := range candidates {
					for _, v := range c.Vector {
						originals[c.Key] = binary.LittleEndian.AppendUint32(originals[c.Key], math.Float32bits(v))
					}
				}
				for _, useMmap := range []bool{false, true} {
					index, err := OpenScalarQuantizedHNSWIndexWithEncodedVectors(ctx, path, kind, reformer, originals, useMmap)
					require.NoError(t, err)
					require.Empty(t, index.base.vectors)
					require.Nil(t, index.base.vectorRows)
					require.Nil(t, index.vectors.originals)
					require.NotNil(t, index.vectors.reader)
					require.Equal(t, original.vectors.codes, index.vectors.codes)
					require.Equal(t, original.base.neighbors, index.base.neighbors)
					assertHNSWGraphInvariants(t, index.base)
					key := candidates[3].Key
					got, ok := index.Vector(key)
					require.True(t, ok)
					require.Equal(t, candidates[3].Vector, got)
					got[0] += 100
					again, ok := index.Vector(key)
					require.True(t, ok)
					require.Equal(t, candidates[3].Vector, again)
					query := candidates[4].Vector
					for _, filter := range []CandidateFilter{nil, func(key uint64) bool { return key%2 == 0 }} {
						search := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 8, Filter: filter}, EF: 32, PrefetchOffset: 2, PrefetchLines: 2}
						want, err := original.SearchHNSW(ctx, query, search)
						require.NoError(t, err)
						actual, err := index.SearchHNSW(ctx, query, search)
						require.NoError(t, err)
						require.Equal(t, want, actual)
						want, err = original.SearchWithOptions(ctx, query, search.SearchOptions)
						require.NoError(t, err)
						actual, err = index.SearchWithOptions(ctx, query, search.SearchOptions)
						require.NoError(t, err)
						require.Equal(t, want, actual)
					}
					saved := filepath.Join(t.TempDir(), "saved.hnsw")
					require.NoError(t, index.Save(ctx, saved))
					oldBytes, err := os.ReadFile(path)
					require.NoError(t, err)
					newBytes, err := os.ReadFile(saved)
					require.NoError(t, err)
					require.Equal(t, oldBytes, newBytes)
					cloned, err := cloneHNSWIndex(ctx, index.base)
					require.NoError(t, err)
					require.Nil(t, cloned.encodedVectors)
					require.Equal(t, candidates[3].Vector, cloned.vectorAt(3))
				}
				originals[candidates[0].Key][0] ^= 1
				_, err = OpenScalarQuantizedHNSWIndexWithEncodedVectors(ctx, path, kind, reformer, originals, true)
				require.ErrorIs(t, err, ErrInvalidHNSWFile)
				delete(originals, candidates[0].Key)
				_, err = OpenScalarQuantizedHNSWIndexWithEncodedVectors(ctx, path, kind, reformer, originals, false)
				require.ErrorIs(t, err, ErrInvalidHNSWFile)
			})
		}
	}
}

func TestHNSWEncodedFP32ContiguousStorage(t *testing.T) {
	ctx := context.Background()
	candidates := hnswMemoryCandidates()
	options := DefaultHNSWBuildOptions(MetricCosine)
	options.M = 4
	options.EFConstruction = 16
	builder, err := NewHNSWBuilder(8, options)
	require.NoError(t, err)
	originals := make(map[uint64][]byte, len(candidates))
	for _, c := range candidates {
		require.NoError(t, builder.Add(ctx, c.Key, c.Vector))
		for _, v := range c.Vector {
			originals[c.Key] = binary.LittleEndian.AppendUint32(originals[c.Key], math.Float32bits(v))
		}
	}
	owned, err := builder.Build(ctx)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "index")
	require.NoError(t, owned.Save(ctx, path))
	for _, mmap := range []bool{false, true} {
		reopened, err := OpenHNSWIndexWithEncodedOriginals(ctx, path, originals, mmap)
		require.NoError(t, err)
		require.Nil(t, reopened.vectorRows)
		require.Nil(t, reopened.encodedVectors)
		require.Equal(t, owned.vectors, reopened.vectors)
		options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 8}, EF: 32, PrefetchOffset: 2, PrefetchLines: 2}
		want, err := owned.SearchHNSW(ctx, candidates[7].Vector, options)
		require.NoError(t, err)
		got, err := reopened.SearchHNSW(ctx, candidates[7].Vector, options)
		require.NoError(t, err)
		require.Equal(t, want, got)
		require.NoError(t, reopened.Add(ctx, 99999, make([]float32, 8)))
		require.Equal(t, candidates[0].Vector, reopened.vectorAt(0))
	}
	originals[candidates[0].Key][0] ^= 1
	_, err = OpenHNSWIndexWithEncodedOriginals(ctx, path, originals, true)
	require.ErrorIs(t, err, ErrInvalidHNSWFile)
}
