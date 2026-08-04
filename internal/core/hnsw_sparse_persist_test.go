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

func TestSparseHNSWPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedSparseHNSWIndex(t, 160)
	path := filepath.Join(t.TempDir(), "vectors.shnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, opened, index)
	query := SparseVector{Indices: []uint32{3, 107, 211}, Values: []float32{1, 2, 3}}
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 25}, EF: 80}
	want, err := index.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)

	got, err := opened.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)
	require.Equal(t, want, got)

	replacement := persistedSparseHNSWIndex(t, 40)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err = OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, opened, replacement)
}

func TestSparseHNSWPersistenceLargeGraphSearch(t *testing.T) {
	index := persistedSparseHNSWIndex(t, DefaultHNSWBruteForceThreshold+100)
	path := filepath.Join(t.TempDir(), "large.shnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	query := sparseHNSWBuildInputs(index.Len())[713].vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EF: 120}
	want, err := index.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)

	got, err := opened.SearchSparseHNSW(context.Background(), query, options)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestSparseHNSWPersistenceEmpty(t *testing.T) {
	t.Parallel()
	builder, err := NewSparseHNSWBuilder(DefaultSparseHNSWBuildOptions())
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "empty.shnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameSparseHNSWIndex(t, opened, index)
}

func TestSparseHNSWPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedSparseHNSWIndex(t, 32)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.shnsw")
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

	after, _ := os.ReadFile(path)
	require.True(t, slices.Equal(after, original),
		"canceled replacement changed published file")
	{
		_, err := OpenSparseHNSWIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenSparseHNSWIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}
	{
		_, err := OpenSparseHNSWIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}
	{
		_, err := OpenSparseHNSWIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	var nilIndex *SparseHNSWIndex
	{
		err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil.shnsw"))
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}

	invalid, err := cloneSparseHNSWIndex(context.Background(), index)
	require.NoError(t, err)

	invalid.offsets[1] = len(invalid.indices) + 1
	{
		err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.shnsw"))
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}
}

func TestSparseHNSWPersistenceDetectsCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(t, 32))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, sparseHNSWHeaderSize - 1, sparseHNSWHeaderSize, len(valid) - 1} {
		{
			_, err := decodeSparseHNSWIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeSparseHNSWIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], sparseHNSWFileVersion+1)
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedSparseHNSWVersion)
	}

	badHeader := slices.Clone(valid)
	badHeader[48] ^= 1
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badHeader)
		require.ErrorIs(t, err, ErrSparseHNSWChecksumMismatch)
	}

	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	{
		_, err := decodeSparseHNSWIndex(context.Background(), badPayload)
		require.ErrorIs(t, err, ErrSparseHNSWChecksumMismatch)
	}
}

func TestSparseHNSWPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(t, 32))
	require.NoError(t, err)

	records := parseSparseHNSWRecordOffsets(t, valid)
	require.True(t, len(records) >= 2,
		"fixture lacks sparse elements or graph edges")
	require.True(t, len(records[0].coordinates) >= 2,
		"fixture lacks sparse elements or graph edges")
	require.True(t, records[0].neighbor >= 0,
		"fixture lacks sparse elements or graph edges")

	tests := []struct {
		name   string
		mutate func([]byte)
	}{
		{"duplicate key", func(data []byte) { copy(data[records[1].key:records[1].key+8], data[records[0].key:records[0].key+8]) }},
		{"coordinate order", func(data []byte) {
			copy(data[records[0].coordinates[1]:records[0].coordinates[1]+4], data[records[0].coordinates[0]:records[0].coordinates[0]+4])
		}},
		{"non-finite value", func(data []byte) {
			binary.LittleEndian.PutUint32(data[records[0].coordinates[0]+4:records[0].coordinates[0]+8], math.Float32bits(float32(math.NaN())))
		}},
		{"invalid level", func(data []byte) {
			binary.LittleEndian.PutUint32(data[records[0].level:records[0].level+4], MaxHNSWLevel+1)
		}},
		{"neighbor out of range", func(data []byte) {
			binary.LittleEndian.PutUint32(data[records[0].neighbor:records[0].neighbor+4], uint32(len(records)))
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			rechecksumSparseHNSW(encoded)
			{
				_, err := decodeSparseHNSWIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
			}
		})
	}
	headerTests := []func([]byte){
		func(data []byte) { binary.LittleEndian.PutUint32(data[48:52], 0) },
		func(data []byte) { binary.LittleEndian.PutUint64(data[60:68], uint64(len(records))) },
		func(data []byte) { data[92] = 1 },
		func(data []byte) { binary.LittleEndian.PutUint64(data[16:24], uint64(len(data)+1)) },
	}
	for _, mutate := range headerTests {
		encoded := slices.Clone(valid)
		mutate(encoded)
		binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
		{
			_, err := decodeSparseHNSWIndex(context.Background(), encoded)
			require.ErrorIs(t, err, ErrInvalidSparseHNSWFile)
		}
	}
}

func FuzzDecodeSparseHNSWIndex(f *testing.F) {
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(f, 12))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("ZVSPHNSW"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeSparseHNSWIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateSparseHNSWIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func persistedSparseHNSWIndex(t testing.TB, count int) *SparseHNSWIndex {
	t.Helper()
	options := DefaultSparseHNSWBuildOptions()
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x123456789abcdef0
	return buildSparseHNSW(t, options, sparseHNSWBuildInputs(count))
}

func assertSameSparseHNSWIndex(t testing.TB, got, want *SparseHNSWIndex) {
	t.Helper()
	require.Equal(t, want.Metric(), got.Metric(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.Len(), got.Len(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.BuildOptions(), got.BuildOptions(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.entryPoint, got.entryPoint,

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.MaxLevel(), got.MaxLevel(),

		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.levelRNGState, got.levelRNGState,

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.keys, want.keys),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.offsets, want.offsets),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.indices, want.indices),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.values, want.values),

		"reopened sparse HNSW metadata differs")
	require.True(t, slices.Equal(got.levels, want.levels),
		"reopened sparse HNSW metadata differs")
	require.Equal(t, want.neighbors, got.neighbors,
		"reopened sparse HNSW metadata differs")
}

type sparseHNSWRecordOffset struct {
	key         int
	level       int
	coordinates []int
	neighbor    int
}

func parseSparseHNSWRecordOffsets(t testing.TB, encoded []byte) []sparseHNSWRecordOffset {
	t.Helper()
	count := int(binary.LittleEndian.Uint64(encoded[32:40]))
	offset := sparseHNSWHeaderSize
	records := make([]sparseHNSWRecordOffset, count)
	for position := range records {
		records[position].neighbor = -1
		records[position].key = offset
		offset += 8
		records[position].level = offset
		level := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		offset += 4
		nonzero := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		offset += 4
		for range nonzero {
			records[position].coordinates = append(records[position].coordinates, offset)
			offset += sparseHNSWElementBytes
		}
		for currentLevel := 0; currentLevel <= level; currentLevel++ {
			degree := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
			offset += 4
			if currentLevel == 0 && degree != 0 {
				records[position].neighbor = offset
			}
			offset += degree * 4
		}
	}
	require.Len(t, encoded, offset)

	return records
}

func rechecksumSparseHNSW(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[88:92], ailego.CRC32C(encoded[sparseHNSWHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
