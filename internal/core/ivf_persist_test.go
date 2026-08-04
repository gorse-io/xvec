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
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestIVFPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricCosine, 3)
	path := filepath.Join(t.TempDir(), "vectors.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())

	opened, err := OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameIVFIndex(t, opened, index)
	query := []float32{0.25, 0.5, 0.75}
	want, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: index.Len()},
		NProbe:        index.NList(),
	})
	require.NoError(t, err)

	got, err := opened.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: opened.Len()},
		NProbe:        opened.NList(),
	})
	require.NoError(t, err)
	require.Equal(t, want, got)

	replacement := persistedIVFIndex(t, MetricIP, 2)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err = OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameIVFIndex(t, opened, replacement)
}

func TestIVFPersistenceEmpty(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricMIPSL2)
	options.Seed = 17
	builder, err := NewIVFBuilder(7, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "empty.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenIVFIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameIVFIndex(t, opened, index)
}

func TestIVFPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricL2, 2)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.ivf")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	original, err := os.ReadFile(path)
	require.NoError(t, err)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		err := index.Save(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}

	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.True(t, slices.Equal(after, original),
		"canceled replacement changed published IVF file")
	{
		_, err := OpenIVFIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenIVFIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
	{
		_, err := OpenIVFIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
	{
		_, err := OpenIVFIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	invalid := &IVFIndex{dimension: 3, options: DefaultIVFBuildOptions(MetricL2), keys: []uint64{1}}
	{
		err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.ivf"))
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
}

func TestIVFPersistenceDetectsTruncationAndCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeIVFIndex(context.Background(), persistedIVFIndex(t, MetricL2, 2))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, ivfHeaderSize - 1, ivfHeaderSize, len(valid) - 1} {
		{
			_, err := decodeIVFIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidIVFFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeIVFIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeIVFIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], ivfFileVersion+1)
	{
		_, err := decodeIVFIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedIVFVersion)
	}

	badHeaderCRC := slices.Clone(valid)
	badHeaderCRC[55] ^= 1
	{
		_, err := decodeIVFIndex(context.Background(), badHeaderCRC)
		require.ErrorIs(t, err, ErrIVFChecksumMismatch)
	}

	badPayloadCRC := slices.Clone(valid)
	badPayloadCRC[len(badPayloadCRC)-1] ^= 1
	{
		_, err := decodeIVFIndex(context.Background(), badPayloadCRC)
		require.ErrorIs(t, err, ErrIVFChecksumMismatch)
	}
}

func TestIVFPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeIVFIndex(context.Background(), persistedIVFIndex(t, MetricL2, 2))
	require.NoError(t, err)

	dimension := int(binary.LittleEndian.Uint32(valid[40:44]))
	nlist := int(binary.LittleEndian.Uint32(valid[44:48]))
	firstRecord := ivfHeaderSize + nlist*dimension*4
	recordSize := ivfRecordOverhead + dimension*4

	tests := []struct {
		name   string
		mutate func([]byte)
	}{
		{
			name: "duplicate key",
			mutate: func(data []byte) {
				copy(data[firstRecord+recordSize:firstRecord+recordSize+8], data[firstRecord:firstRecord+8])
			},
		},
		{
			name: "list out of range",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[firstRecord+8+dimension*4:firstRecord+recordSize], uint32(nlist))
			},
		},
		{
			name: "non-finite centroid",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[ivfHeaderSize:ivfHeaderSize+4], math.Float32bits(float32(math.NaN())))
			},
		},
		{
			name: "non-finite vector",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[firstRecord+8:firstRecord+12], math.Float32bits(float32(math.Inf(1))))
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			rechecksumIVF(encoded)
			{
				_, err := decodeIVFIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidIVFFile)
			}
		})
	}

	badOptions := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badOptions[52:56], 0)
	binary.LittleEndian.PutUint32(badOptions[108:112], ailego.CRC32C(badOptions[:108]))
	{
		_, err := decodeIVFIndex(context.Background(), badOptions)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}

	badLength := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badLength[16:24], uint64(len(badLength)+1))
	binary.LittleEndian.PutUint32(badLength[108:112], ailego.CRC32C(badLength[:108]))
	{
		_, err := decodeIVFIndex(context.Background(), badLength)
		require.ErrorIs(t, err, ErrInvalidIVFFile)
	}
}

func FuzzDecodeIVFIndex(f *testing.F) {
	index := persistedIVFIndex(f, MetricL2, 2)
	valid, err := encodeIVFIndex(context.Background(), index)
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("ZVECIVF"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeIVFIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateIVFIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func persistedIVFIndex(t testing.TB, metric Metric, nlist int) *IVFIndex {
	t.Helper()
	options := DefaultIVFBuildOptions(metric)
	options.NList = nlist
	options.NIterations = 7
	options.Tolerance = 1e-8
	options.Workers = 2
	options.Seed = 0x123456789abcdef0
	builder, err := NewIVFBuilder(3, options)
	require.NoError(t, err)

	for _, candidate := range []Candidate{
		{Key: 41, Vector: []float32{1, 0, 0}},
		{Key: 7, Vector: []float32{0, 1, 0}},
		{Key: 99, Vector: []float32{0, 0, 1}},
		{Key: 5, Vector: []float32{1, 1, 0}},
		{Key: 123, Vector: []float32{0.5, 0.25, 0.75}},
	} {
		{
			err := builder.Add(context.Background(), candidate.Key, candidate.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

func assertSameIVFIndex(t testing.TB, got, want *IVFIndex) {
	t.Helper()
	require.Equal(t, want.Dimension(), got.Dimension())
	require.Equal(t, want.Metric(), got.Metric())
	require.Equal(t, want.Len(), got.Len())
	require.Equal(t, want.NList(), got.NList())
	require.Equal(t, want.BuildOptions(), got.BuildOptions())
	require.Equal(t, want.TrainingCost(), got.TrainingCost())
	require.Equal(t, want.TrainingIterations(), got.TrainingIterations())
	require.Equal(t, want.TrainingConverged(), got.TrainingConverged())
	require.Equal(t, want.Centroids(), got.Centroids())
	require.True(t, slices.Equal(got.keys, want.keys))

	for _, key := range want.keys {
		gotVector, gotOK := got.Vector(key)
		wantVector, wantOK := want.Vector(key)
		gotList, gotListOK := got.ListForKey(key)
		wantList, wantListOK := want.ListForKey(key)
		require.Equal(t, wantOK, gotOK)
		require.True(t, slices.Equal(gotVector, wantVector))
		require.Equal(t, wantListOK, gotListOK)
		require.Equal(t, wantList, gotList)
	}
}

func rechecksumIVF(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[96:100], ailego.CRC32C(encoded[ivfHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
