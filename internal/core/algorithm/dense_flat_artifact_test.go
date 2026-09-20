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
	"errors"
	"fmt"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestDenseFlatArtifactRoundTrip(t *testing.T) {
	ctx := context.Background()
	index, err := NewDenseFlatIndex(2, MetricL2)
	require.NoError(t, err)
	require.NoError(t, index.Add(ctx, 7, []float32{1, 2}))
	require.NoError(t, index.Add(ctx, 9, []float32{3, 4}))

	path := filepath.Join(t.TempDir(), "flat.xvec")
	require.NoError(t, index.Save(ctx, path))
	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())

	reopened, err := OpenDenseFlatIndexWithMmap(ctx, path, true)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, reopened.Close()) })
	require.Equal(t, 2, reopened.Dimension())
	require.Equal(t, MetricL2, reopened.Metric())
	require.Equal(t, 2, reopened.Len())
	require.Equal(t, []float32{3, 4}, mustDenseVector(t, reopened, 9))

	results, err := reopened.Search(ctx, []float32{1, 2}, 2)
	require.NoError(t, err)
	require.Equal(t, []Result{{Key: 7, Score: 0}, {Key: 9, Score: 8}}, results)
}

func TestDenseFlatArtifactRoundTripsMetricsAndStorageModes(t *testing.T) {
	ctx := context.Background()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine} {
		for _, useMmap := range []bool{false, true} {
			t.Run(fmt.Sprintf("metric=%d/mmap=%t", metric, useMmap), func(t *testing.T) {
				index, err := NewDenseFlatIndex(2, metric)
				require.NoError(t, err)
				for _, candidate := range []Candidate{
					{Key: 10, Vector: []float32{1, 0}},
					{Key: 20, Vector: []float32{0, 2}},
					{Key: 30, Vector: []float32{1, 1}},
				} {
					require.NoError(t, index.Add(ctx, candidate.Key, candidate.Vector))
				}
				path := filepath.Join(t.TempDir(), "flat.xvec")
				require.NoError(t, index.Save(ctx, path))

				reopened, err := OpenDenseFlatIndexWithMmap(ctx, path, useMmap)
				require.NoError(t, err)
				t.Cleanup(func() { require.NoError(t, reopened.Close()) })
				require.Equal(t, useMmap, reopened.mapping != nil)
				require.NotEmpty(t, reopened.backing)
				layout := decodeLayoutForTest(t, reopened.backing)
				require.Equal(t, unsafe.Pointer(&reopened.backing[layout.keysOffset]), unsafe.Pointer(&reopened.keys[0]))
				require.Equal(t, unsafe.Pointer(&reopened.backing[layout.vectorsOffset]), unsafe.Pointer(&reopened.vectors[0]))
				if metric == MetricCosine {
					require.Equal(t, unsafe.Pointer(&reopened.backing[layout.magnitudesOffset]), unsafe.Pointer(&reopened.magnitudes[0]))
				}

				want, err := index.Search(ctx, []float32{1, 0}, 3)
				require.NoError(t, err)
				got, err := reopened.Search(ctx, []float32{1, 0}, 3)
				require.NoError(t, err)
				require.Equal(t, want, got)

				options := SearchOptions{TopK: 2, Filter: func(key uint64) bool { return key != 20 }}
				wantOptions, err := index.SearchWithOptions(ctx, []float32{1, 0}, options)
				require.NoError(t, err)
				gotOptions, err := reopened.SearchWithOptions(ctx, []float32{1, 0}, options)
				require.NoError(t, err)
				require.Equal(t, wantOptions, gotOptions)

				groups := GroupByOptions{
					GroupCount: 2, TopKPerGroup: 2,
					Resolve: func(key uint64) (string, bool) {
						if key == 20 {
							return "b", true
						}
						return "a", true
					},
				}
				wantGroups, err := index.SearchGroups(ctx, []float32{1, 0}, groups)
				require.NoError(t, err)
				gotGroups, err := reopened.SearchGroups(ctx, []float32{1, 0}, groups)
				require.NoError(t, err)
				require.Equal(t, wantGroups, gotGroups)
			})
		}
	}
}

func TestDenseFlatArtifactEmptyRoundTrip(t *testing.T) {
	index, err := NewDenseFlatIndex(3, MetricCosine)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "empty.xvec")
	require.NoError(t, index.Save(context.Background(), path))
	reopened, err := OpenDenseFlatIndexWithMmap(context.Background(), path, true)
	require.NoError(t, err)
	require.Zero(t, reopened.Len())
	require.NoError(t, reopened.Close())
	require.NoError(t, reopened.Close())
}

func TestDenseFlatArtifactRejectsCorruptionAndInvalidHeaders(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricL2)
	require.NoError(t, err)
	require.NoError(t, index.Add(context.Background(), 1, []float32{1, 2}))
	require.NoError(t, index.Add(context.Background(), 2, []float32{3, 4}))
	path := filepath.Join(t.TempDir(), "valid.xvec")
	require.NoError(t, index.Save(context.Background(), path))
	original, err := os.ReadFile(path)
	require.NoError(t, err)

	tests := []struct {
		name   string
		mutate func([]byte) []byte
	}{
		{"truncated_header", func(data []byte) []byte { return data[:32] }},
		{"truncated_payload", func(data []byte) []byte { return data[:len(data)-1] }},
		{"bad_magic", func(data []byte) []byte { data[0] ^= 0xff; return data }},
		{"bad_crc", func(data []byte) []byte { data[len(data)-1] ^= 0xff; return data }},
		{"zero_dimension", func(data []byte) []byte { binary.LittleEndian.PutUint64(data[16:24], 0); return data }},
		{"wide_invalid_metric", func(data []byte) []byte { binary.LittleEndian.PutUint32(data[24:28], 0x101); return data }},
		{"overflow_count", func(data []byte) []byte { binary.LittleEndian.PutUint64(data[32:40], ^uint64(0)); return data }},
		{"unaligned_offset", func(data []byte) []byte { binary.LittleEndian.PutUint64(data[40:48], 129); return data }},
		{"duplicate_key", func(data []byte) []byte {
			copy(data[136:144], data[128:136])
			rewriteDenseFlatChecksum(data)
			return data
		}},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			data := testCase.mutate(append([]byte(nil), original...))
			corrupt := filepath.Join(t.TempDir(), "corrupt.xvec")
			require.NoError(t, os.WriteFile(corrupt, data, 0o600))
			_, err := OpenDenseFlatIndexWithMmap(context.Background(), corrupt, true)
			require.Error(t, err)
		})
	}
}

func TestDenseFlatArtifactRejectsIncorrectCosineMagnitude(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricCosine)
	require.NoError(t, err)
	require.NoError(t, index.Add(context.Background(), 1, []float32{3, 4}))
	path := filepath.Join(t.TempDir(), "cosine.xvec")
	require.NoError(t, index.Save(context.Background(), path))

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	layout := decodeLayoutForTest(t, data)
	binary.LittleEndian.PutUint32(data[layout.magnitudesOffset:], math.Float32bits(123))
	rewriteDenseFlatChecksum(data)
	require.NoError(t, os.WriteFile(path, data, 0o600))

	_, err = OpenDenseFlatIndexWithMmap(context.Background(), path, true)
	require.Error(t, err)
}

func TestDenseFlatArtifactReadOnlyCloseAndCancellation(t *testing.T) {
	ctx := context.Background()
	index, err := NewDenseFlatIndex(8, MetricL2)
	require.NoError(t, err)
	for key := range 512 {
		vector := make([]float32, 8)
		vector[0] = float32(key)
		require.NoError(t, index.Add(ctx, uint64(key), vector))
	}
	path := filepath.Join(t.TempDir(), "flat.xvec")
	require.NoError(t, index.Save(ctx, path))
	reopened, err := OpenDenseFlatIndexWithMmap(ctx, path, true)
	require.NoError(t, err)
	require.ErrorIs(t, reopened.Add(ctx, 999, make([]float32, 8)), ErrDenseFlatReadOnly)
	require.ErrorIs(t, reopened.Reserve(1024), ErrDenseFlatReadOnly)

	var wait sync.WaitGroup
	searchErrors := make(chan error, 16)
	for range 16 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, searchErr := reopened.Search(ctx, make([]float32, 8), 10)
			searchErrors <- searchErr
		}()
	}
	require.NoError(t, reopened.Close())
	wait.Wait()
	close(searchErrors)
	for searchErr := range searchErrors {
		require.True(t, searchErr == nil || errors.Is(searchErr, ErrDenseFlatClosed))
	}
	require.NoError(t, reopened.Close())
	_, err = reopened.Search(ctx, make([]float32, 8), 1)
	require.ErrorIs(t, err, ErrDenseFlatClosed)
	require.NoError(t, os.Remove(path))

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	require.ErrorIs(t, index.Save(canceled, filepath.Join(t.TempDir(), "cancel.xvec")), context.Canceled)
	_, err = OpenDenseFlatIndexWithMmap(canceled, path, true)
	require.ErrorIs(t, err, context.Canceled)
}

func TestDenseFlatClosePreventsInMemoryMutation(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricL2)
	require.NoError(t, err)
	require.NoError(t, index.Close())
	require.ErrorIs(t, index.Add(context.Background(), 1, []float32{1, 2}), ErrDenseFlatClosed)
	require.ErrorIs(t, index.Reserve(10), ErrDenseFlatClosed)
}

func TestDenseFlatCloseInvalidatesIndexWhenUnmapFails(t *testing.T) {
	index, err := NewDenseFlatIndex(2, MetricL2)
	require.NoError(t, err)
	require.NoError(t, index.Add(context.Background(), 7, []float32{1, 2}))
	index.backing = []byte{1}

	unmapErr := errors.New("unmap failed")
	unmapCalls := 0
	index.unmap = func() error {
		unmapCalls++
		return unmapErr
	}

	require.ErrorIs(t, index.Close(), unmapErr)
	require.True(t, index.closed)
	require.Nil(t, index.keys)
	require.Nil(t, index.vectors)
	require.Nil(t, index.magnitudes)
	require.Nil(t, index.positions)
	require.Nil(t, index.backing)
	require.Nil(t, index.mapping)
	require.Nil(t, index.unmap)

	_, err = index.Search(context.Background(), []float32{1, 2}, 1)
	require.ErrorIs(t, err, ErrDenseFlatClosed)
	_, found := index.Vector(7)
	require.False(t, found)

	require.NoError(t, index.Close())
	require.Equal(t, 1, unmapCalls)
}

func mustDenseVector(t *testing.T, index *DenseFlatIndex, key uint64) []float32 {
	t.Helper()
	vector, ok := index.Vector(key)
	require.True(t, ok)
	return vector
}

func decodeLayoutForTest(t *testing.T, data []byte) denseFlatArtifactLayout {
	t.Helper()
	layout, err := decodeDenseFlatArtifactHeader(data[:denseFlatArtifactHeaderSize], int64(len(data)))
	require.NoError(t, err)
	return layout
}

func rewriteDenseFlatChecksum(data []byte) {
	for offset := denseFlatArtifactCRCOffset; offset < denseFlatArtifactCRCOffset+4; offset++ {
		data[offset] = 0
	}
	binary.LittleEndian.PutUint32(data[denseFlatArtifactCRCOffset:], crc32.Checksum(data, denseFlatCRCTable))
}
