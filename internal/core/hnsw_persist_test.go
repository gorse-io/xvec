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

func TestHNSWPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricCosine, 160)
	path := filepath.Join(t.TempDir(), "vectors.hnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	info, err := os.Stat(path)
	require.NoError(t, err)
	assertPrivateFileMode(t, info.Mode())

	opened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, opened, index)
	query := []float32{7.25, 13.5, 1.25}
	want, err := index.SearchHNSW(context.Background(), query, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 80,
	})
	require.NoError(t, err)

	got, err := opened.SearchHNSW(context.Background(), query, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 80,
	})
	require.NoError(t, err)
	require.Equal(t, want, got)

	replacement := persistedHNSWIndex(t, MetricIP, 40)
	{
		err := replacement.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err = OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, opened, replacement)
}

func TestHNSWPersistenceLargeGraphSearch(t *testing.T) {
	index := persistedHNSWIndex(t, MetricL2, DefaultHNSWBruteForceThreshold+100)
	path := filepath.Join(t.TempDir(), "large.hnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	query := hnswBuildInputs(index.Len())[713].Vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EF: 120}
	want, err := index.SearchHNSW(context.Background(), query, options)
	require.NoError(t, err)

	got, err := opened.SearchHNSW(context.Background(), query, options)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestHNSWPersistenceEmpty(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricMIPSL2)
	options.M = 4
	options.EFConstruction = 12
	options.Seed = 17
	builder, err := NewHNSWBuilder(7, options)
	require.NoError(t, err)

	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	path := filepath.Join(t.TempDir(), "empty.hnsw")
	{
		err := index.Save(context.Background(), path)
		require.NoError(t, err)
	}

	opened, err := OpenHNSWIndex(context.Background(), path)
	require.NoError(t, err)

	assertSameHNSWIndex(t, opened, index)
}

func TestHNSWPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricL2, 32)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.hnsw")
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
		"canceled replacement changed published HNSW file")
	{
		_, err := OpenHNSWIndex(canceled, path)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := encodeHNSWIndex(canceled, index)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := decodeHNSWIndex(canceled, original)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		err := index.Save(nil, path)
		require.Error(t, err,
			"nil Save context succeeded")
	}
	{
		_, err := OpenHNSWIndex(nil, path)
		require.Error(t, err,
			"nil Open context succeeded")
	}
	{
		_, err := encodeHNSWIndex(nil, index)
		require.Error(t, err,
			"nil encode context succeeded")
	}
	{
		_, err := decodeHNSWIndex(nil, original)
		require.Error(t, err,
			"nil decode context succeeded")
	}
	{
		err := index.Save(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}
	{
		_, err := OpenHNSWIndex(context.Background(), "")
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}
	{
		_, err := OpenHNSWIndex(context.Background(), filepath.Join(dir, "missing"))
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	var nilIndex *HNSWIndex
	{
		err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil.hnsw"))
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}

	invalid := &HNSWIndex{
		dimension: 3,
		options:   DefaultHNSWBuildOptions(MetricL2),
		keys:      []uint64{1},
	}
	{
		err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.hnsw"))
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}
}

func TestHNSWPersistenceDetectsTruncationAndCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(t, MetricL2, 32))
	require.NoError(t, err)

	for _, cut := range []int{0, 1, hnswHeaderSize - 1, hnswHeaderSize, len(valid) - 1} {
		{
			_, err := decodeHNSWIndex(context.Background(), valid[:cut])
			require.ErrorIs(t, err, ErrInvalidHNSWFile)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	{
		_, err := decodeHNSWIndex(context.Background(), trailing)
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}

	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	{
		_, err := decodeHNSWIndex(context.Background(), badMagic)
		require.ErrorIs(t, err, ErrInvalidHNSWFile)
	}

	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], hnswFileVersion+1)
	{
		_, err := decodeHNSWIndex(context.Background(), badVersion)
		require.ErrorIs(t, err, ErrUnsupportedHNSWVersion)
	}

	badHeaderCRC := slices.Clone(valid)
	badHeaderCRC[44] ^= 1
	{
		_, err := decodeHNSWIndex(context.Background(), badHeaderCRC)
		require.ErrorIs(t, err, ErrHNSWChecksumMismatch)
	}

	badPayloadCRC := slices.Clone(valid)
	badPayloadCRC[len(badPayloadCRC)-1] ^= 1
	{
		_, err := decodeHNSWIndex(context.Background(), badPayloadCRC)
		require.ErrorIs(t, err, ErrHNSWChecksumMismatch)
	}
}

func TestHNSWPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(t, MetricL2, 32))
	require.NoError(t, err)

	records := parseHNSWRecordOffsets(t, valid)
	require.True(t, len(records) >= 2,
		"persistence fixture lacks required graph edges")
	require.True(t, len(records[0].neighborOffsets) >= 2,
		"persistence fixture lacks required graph edges")

	upperNeighborOffset := -1
	for _, record := range records {
		if len(record.upperNeighborOffsets) != 0 {
			upperNeighborOffset = record.upperNeighborOffsets[0]
			break
		}
	}
	lowLevelPosition := -1
	for position, record := range records {
		if record.maxLevel == 0 {
			lowLevelPosition = position
			break
		}
	}
	require.True(t, upperNeighborOffset >= 0,
		"persistence fixture lacks an upper edge or level-zero target")
	require.True(t, lowLevelPosition >= 0,
		"persistence fixture lacks an upper edge or level-zero target")

	tests := []struct {
		name   string
		mutate func([]byte)
	}{
		{
			name: "duplicate key",
			mutate: func(data []byte) {
				copy(data[records[1].key:records[1].key+8], data[records[0].key:records[0].key+8])
			},
		},
		{
			name: "non-finite vector",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[records[0].vector:records[0].vector+4], math.Float32bits(float32(math.NaN())))
			},
		},
		{
			name: "invalid level",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[records[0].level:records[0].level+4], MaxHNSWLevel+1)
			},
		},
		{
			name: "neighbor out of range",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[records[0].neighborOffsets[0]:records[0].neighborOffsets[0]+4], uint32(len(records)))
			},
		},
		{
			name: "duplicate neighbor",
			mutate: func(data []byte) {
				copy(data[records[0].neighborOffsets[1]:records[0].neighborOffsets[1]+4], data[records[0].neighborOffsets[0]:records[0].neighborOffsets[0]+4])
			},
		},
		{
			name: "neighbor lacks level",
			mutate: func(data []byte) {
				binary.LittleEndian.PutUint32(data[upperNeighborOffset:upperNeighborOffset+4], uint32(lowLevelPosition))
			},
		},
		{
			name: "degree exceeds limit",
			mutate: func(data []byte) {
				m := binary.LittleEndian.Uint32(data[44:48])
				binary.LittleEndian.PutUint32(data[records[0].degree:records[0].degree+4], m*2+1)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			rechecksumHNSW(encoded)
			{
				_, err := decodeHNSWIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidHNSWFile)
			}
		})
	}

	headerTests := []struct {
		name   string
		mutate func([]byte)
	}{
		{"invalid options", func(data []byte) { binary.LittleEndian.PutUint32(data[44:48], 0) }},
		{"entry out of range", func(data []byte) { binary.LittleEndian.PutUint64(data[56:64], uint64(len(records))) }},
		{"maximum level mismatch", func(data []byte) { binary.LittleEndian.PutUint32(data[64:68], MaxHNSWLevel) }},
		{"reserved field", func(data []byte) { data[88] = 1 }},
		{"file length", func(data []byte) { binary.LittleEndian.PutUint64(data[16:24], uint64(len(data)+1)) }},
	}
	for _, test := range headerTests {
		t.Run(test.name, func(t *testing.T) {
			encoded := slices.Clone(valid)
			test.mutate(encoded)
			binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
			{
				_, err := decodeHNSWIndex(context.Background(), encoded)
				require.ErrorIs(t, err, ErrInvalidHNSWFile)
			}
		})
	}
}

func FuzzDecodeHNSWIndex(f *testing.F) {
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(f, MetricL2, 12))
	require.NoError(f, err)

	f.Add(valid)
	f.Add([]byte("ZVECHNSW"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeHNSWIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		{
			err := validateHNSWIndex(context.Background(), index)
			require.NoError(t, err)
		}
	})
}

func persistedHNSWIndex(t testing.TB, metric Metric, count int) *HNSWIndex {
	t.Helper()
	options := DefaultHNSWBuildOptions(metric)
	options.M = 8
	options.EFConstruction = 40
	options.Seed = 0x123456789abcdef0
	builder, err := NewHNSWBuilder(3, options)
	require.NoError(t, err)

	for _, input := range hnswBuildInputs(count) {
		{
			err := builder.Add(context.Background(), input.Key, input.Vector)
			require.NoError(t, err)
		}
	}
	index, err := builder.Build(context.Background())
	require.NoError(t, err)

	return index
}

func assertSameHNSWIndex(t testing.TB, got, want *HNSWIndex) {
	t.Helper()
	require.Equal(t, want.Dimension(), got.Dimension())
	require.Equal(t, want.Metric(), got.Metric())
	require.Equal(t, want.Len(), got.Len())
	require.Equal(t, want.BuildOptions(), got.BuildOptions())
	require.Equal(t, want.entryPoint, got.entryPoint)
	require.Equal(t, want.MaxLevel(), got.MaxLevel())
	require.Equal(t, want.levelRNGState, got.levelRNGState)
	require.True(t, slices.Equal(got.keys, want.keys))
	require.True(t, slices.Equal(got.levels, want.levels))
	require.Equal(t, want.neighbors, got.neighbors)

	for _, key := range want.keys {
		gotVector, gotOK := got.Vector(key)
		wantVector, wantOK := want.Vector(key)
		require.Equal(t, wantOK, gotOK)
		require.True(t, slices.Equal(gotVector, wantVector))
	}
}

type hnswRecordOffset struct {
	key                  int
	level                int
	maxLevel             int
	vector               int
	degree               int
	neighborOffsets      []int
	upperNeighborOffsets []int
}

func parseHNSWRecordOffsets(t testing.TB, encoded []byte) []hnswRecordOffset {
	t.Helper()
	count := int(binary.LittleEndian.Uint64(encoded[32:40]))
	dimension := int(binary.LittleEndian.Uint32(encoded[40:44]))
	offset := hnswHeaderSize
	records := make([]hnswRecordOffset, count)
	for position := range records {
		records[position].key = offset
		offset += 8
		records[position].level = offset
		level := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		records[position].maxLevel = level
		offset += 4
		records[position].vector = offset
		offset += dimension * 4
		for currentLevel := 0; currentLevel <= level; currentLevel++ {
			degreeOffset := offset
			degree := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
			offset += 4
			if currentLevel == 0 {
				records[position].degree = degreeOffset
				for neighborIndex := 0; neighborIndex < degree; neighborIndex++ {
					records[position].neighborOffsets = append(records[position].neighborOffsets, offset)
					offset += 4
				}
			} else {
				for neighborIndex := 0; neighborIndex < degree; neighborIndex++ {
					records[position].upperNeighborOffsets = append(records[position].upperNeighborOffsets, offset)
					offset += 4
				}
			}
		}
	}
	require.Len(t, encoded, offset)

	return records
}

func rechecksumHNSW(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[84:88], ailego.CRC32C(encoded[hnswHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
