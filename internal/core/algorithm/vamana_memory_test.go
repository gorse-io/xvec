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
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestVamanaReservedStorageAndQuantizedTransfer(t *testing.T) {
	ctx := context.Background()
	for _, fp16 := range []bool{false, true} {
		t.Run(fmt.Sprint(fp16), func(t *testing.T) {
			builder, err := newVamanaBuilder(4, DefaultVamanaBuildOptions(MetricL2), fp16)
			require.NoError(t, err)
			require.ErrorIs(t, builder.Reserve(-1), ErrVamanaCapacity)
			require.ErrorIs(t, builder.Reserve(maxPlatformInt()), ErrVamanaCapacity)
			require.NoError(t, builder.Reserve(40))
			for _, candidate := range quantizedIndexCandidates(40) {
				require.NoError(t, builder.Add(ctx, candidate.Key, candidate.Vector))
			}
			require.NoError(t, builder.Reserve(1))
			if fp16 {
				owned := &builder.vectorsFP16[0]
				index, err := builder.BuildInterleavedWithWorkers(ctx, 2)
				require.NoError(t, err)
				require.Same(t, owned, &index.vectorsFP16[0])
			} else {
				owned := &builder.vectors[0]
				index, err := builder.BuildScalarQuantizedInterleavedWithWorkers(ctx, 2, QuantizationInt8, nil)
				require.NoError(t, err)
				require.Same(t, owned, &index.base.vectors[0])
				require.Same(t, owned, &index.vectors.originals[0])
				path := filepath.Join(t.TempDir(), "quantized")
				require.NoError(t, index.Save(ctx, path))
				reopened, err := OpenScalarQuantizedVamanaIndex(ctx, path, QuantizationInt8, nil)
				require.NoError(t, err)
				want, err := index.Search(ctx, []float32{1, 2, 3, 4}, 10)
				require.NoError(t, err)
				got, err := reopened.Search(ctx, []float32{1, 2, 3, 4}, 10)
				require.NoError(t, err)
				require.Equal(t, want, got)
			}
			require.ErrorIs(t, builder.Reserve(80), ErrBuilderClosed)
		})
	}
}

func TestVamanaStreamedSave(t *testing.T) {
	ctx := context.Background()
	for _, fp16 := range []bool{false, true} {
		t.Run(fmt.Sprint(fp16), func(t *testing.T) {
			builder, err := newVamanaBuilder(768, DefaultVamanaBuildOptions(MetricL2), fp16)
			require.NoError(t, err)
			require.NoError(t, builder.Reserve(48))
			for n := range 48 {
				vector := make([]float32, 768)
				vector[n] = float32(n + 1)
				require.NoError(t, builder.Add(ctx, uint64(n), vector))
			}
			index, err := builder.Build(ctx)
			require.NoError(t, err)
			encoded, err := encodeVamanaIndex(ctx, index)
			require.NoError(t, err)
			path := filepath.Join(t.TempDir(), "index")
			require.NoError(t, index.Save(ctx, path))
			got, err := os.ReadFile(path)
			require.NoError(t, err)
			require.Equal(t, encoded, got)
			// Checksums captured from the pre-streaming encoder at 5d9b8f5.
			wantHash := "00f9cc48998de8bf9f4546e7edaa25e8db9b1cc822200c0d0f2afcf16c770fa6"
			if fp16 {
				wantHash = "879bb875ea8b6016c242c8583ef8ec72d19ecc4ec485279260c1fcd533287b71"
			}
			require.Equal(t, wantHash, fmt.Sprintf("%x", sha256.Sum256(got)))
			reopened, err := OpenVamanaIndex(ctx, path)
			require.NoError(t, err)
			require.Equal(t, index.neighbors, reopened.neighbors)
			require.Equal(t, index.vectors, reopened.vectors)
			require.Equal(t, index.vectorsFP16, reopened.vectorsFP16)
			mapped, err := OpenVamanaIndexWithMmap(ctx, path, true)
			require.NoError(t, err)
			require.Equal(t, reopened.vectors, mapped.vectors)
			require.Equal(t, reopened.vectorsFP16, mapped.vectorsFP16)
			require.Equal(t, reopened.neighbors, mapped.neighbors)
			// The decoded index owns its data after the temporary mapping closes.
			require.NoError(t, mapped.Add(ctx, 1000, make([]float32, 768)))
			require.Equal(t, 49, mapped.Len())
			require.Equal(t, 48, reopened.Len())
			canceled, cancel := context.WithCancel(ctx)
			cancel()
			require.ErrorIs(t, index.Save(canceled, path), context.Canceled)
			unchanged, err := os.ReadFile(path)
			require.NoError(t, err)
			require.Equal(t, got, unchanged)
		})
	}
}

// BenchmarkVamanaSaveMemory isolates persistence allocations from graph build.
func BenchmarkVamanaSaveMemory(b *testing.B) {
	for _, fp16 := range []bool{false, true} {
		b.Run(fmt.Sprint(fp16), func(b *testing.B) {
			ctx := context.Background()
			builder, err := newVamanaBuilder(768, DefaultVamanaBuildOptions(MetricL2), fp16)
			if err != nil {
				b.Fatal(err)
			}
			for n := range 2048 {
				vector := make([]float32, 768)
				vector[n%768] = float32(n%100 + 1)
				if err := builder.Add(ctx, uint64(n), vector); err != nil {
					b.Fatal(err)
				}
			}
			index, err := builder.BuildInterleavedWithWorkers(ctx, 2)
			if err != nil {
				b.Fatal(err)
			}
			path := filepath.Join(b.TempDir(), "index")
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := index.Save(ctx, path); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestVamanaBorrowedRowsAndEncodedOriginals(t *testing.T) {
	ctx := context.Background()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		t.Run(fmt.Sprint(metric), func(t *testing.T) {
			candidates := quantizedIndexCandidates(1100)
			options := DefaultVamanaBuildOptions(metric)
			options.MaxDegree, options.SearchListSize = 8, 32
			builder, err := NewVamanaBuilder(4, options)
			require.NoError(t, err)
			require.NoError(t, builder.Reserve(len(candidates)))
			originals := make(map[uint64][]byte, len(candidates))
			for _, c := range candidates {
				require.NoError(t, builder.Add(ctx, c.Key, c.Vector))
				for _, value := range c.Vector {
					originals[c.Key] = binary.LittleEndian.AppendUint32(originals[c.Key], math.Float32bits(value))
				}
			}
			owned, err := builder.BuildInterleavedWithWorkers(ctx, 1)
			require.NoError(t, err)
			borrowed, err := BuildVamanaWithBorrowedVectors(ctx, 4, options, candidates, 1)
			require.NoError(t, err)
			require.Nil(t, borrowed.vectors)
			require.Same(t, &candidates[0].Vector[0], &borrowed.vectorRows[0][0])
			wantBytes, err := encodeVamanaIndex(ctx, owned)
			require.NoError(t, err)
			gotBytes, err := encodeVamanaIndex(ctx, borrowed)
			require.NoError(t, err)
			require.Equal(t, wantBytes, gotBytes)
			query := candidates[7].Vector
			want, err := owned.Search(ctx, query, 20)
			require.NoError(t, err)
			got, err := borrowed.Search(ctx, query, 20)
			require.NoError(t, err)
			require.Equal(t, want, got)
			path := filepath.Join(t.TempDir(), "index")
			require.NoError(t, borrowed.Save(ctx, path))
			require.NoError(t, borrowed.Add(ctx, 9999999, []float32{1, 2, 3, 4}))
			require.Nil(t, borrowed.vectorRows)
			require.Equal(t, candidates[0].Vector, borrowed.vectorAt(0))
			for _, kind := range []Quantization{QuantizationInt4, QuantizationInt8, QuantizationFP16} {
				for _, useMmap := range []bool{false, true} {
					t.Run(fmt.Sprintf("%v/%v", kind, useMmap), func(t *testing.T) {
						expected, err := NewScalarQuantizedVamanaIndex(ctx, owned, kind, nil)
						require.NoError(t, err)
						encoded, err := OpenScalarQuantizedVamanaIndexWithEncodedVectors(ctx, path, kind, nil, originals, useMmap)
						require.NoError(t, err)
						require.Nil(t, encoded.base.vectors)
						require.Nil(t, encoded.base.neighborDistances)
						require.Nil(t, encoded.vectors.originals)
						require.NotNil(t, encoded.vectors.reader)
						require.Same(t, &originals[candidates[0].Key][0], &encoded.base.encodedVectors[0][0])
						require.Same(t, encoded.vectors, encoded.FlatIndex().vectors)
						for _, filter := range []CandidateFilter{nil, func(key uint64) bool { return key%3 != 0 }} {
							search := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 20, Filter: filter}, EFSearch: 64}
							want, err := expected.SearchVamana(ctx, query, search)
							require.NoError(t, err)
							got, err := encoded.SearchVamana(ctx, query, search)
							require.NoError(t, err)
							require.Equal(t, want, got)
						}
						vector, found := encoded.Vector(candidates[0].Key)
						require.True(t, found)
						require.Equal(t, candidates[0].Vector, vector)
						vector[0]++
						again, _ := encoded.Vector(candidates[0].Key)
						require.Equal(t, candidates[0].Vector, again)
						saved := filepath.Join(t.TempDir(), "resaved")
						require.NoError(t, encoded.Save(ctx, saved))
						savedBytes, err := os.ReadFile(saved)
						require.NoError(t, err)
						require.Equal(t, wantBytes, savedBytes)
						// Copy-on-write rebuilds the omitted build caches and detaches originals.
						detached, err := cloneVamanaIndex(ctx, encoded.base)
						require.NoError(t, err)
						require.NoError(t, detached.Add(ctx, 9999999, []float32{1, 2, 3, 4}))
						require.Nil(t, detached.encodedVectors)
						require.NoError(t, validateVamanaIndex(ctx, detached))
					})
				}
			}
			key := candidates[0].Key
			original := originals[key]
			originals[key] = slices.Clone(original)
			originals[key][0] ^= 1
			_, err = OpenScalarQuantizedVamanaIndexWithEncodedVectors(ctx, path, QuantizationInt4, nil, originals, true)
			require.ErrorIs(t, err, ErrInvalidVamanaFile)
			delete(originals, key)
			_, err = OpenScalarQuantizedVamanaIndexWithEncodedVectors(ctx, path, QuantizationInt4, nil, originals, false)
			require.ErrorIs(t, err, ErrInvalidVamanaFile)
		})
	}
}

func TestVamanaBorrowedRowsValidation(t *testing.T) {
	ctx := context.Background()
	options := DefaultVamanaBuildOptions(MetricL2)
	_, err := BuildVamanaWithBorrowedVectors(nil, 4, options, nil, 1)
	require.Error(t, err)
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = BuildVamanaWithBorrowedVectors(canceled, 4, options, nil, 1)
	require.ErrorIs(t, err, context.Canceled)
	candidates := quantizedIndexCandidates(2)
	candidates[1].Key = candidates[0].Key
	_, err = BuildVamanaWithBorrowedVectors(ctx, 4, options, candidates, 1)
	require.ErrorIs(t, err, ErrDuplicateKey)
	_, err = BuildVamanaWithBorrowedVectors(ctx, 3, options, candidates[:1], 1)
	require.Error(t, err)
	_, err = BuildVamanaWithBorrowedVectors(ctx, 4, options, nil, 0)
	require.ErrorIs(t, err, ErrInvalidVamanaWorkers)
	_, err = BuildScalarQuantizedVamanaWithBorrowedVectors(ctx, 4, options, nil, 1, Quantization(255), nil)
	require.ErrorIs(t, err, ErrInvalidQuantization)
}
