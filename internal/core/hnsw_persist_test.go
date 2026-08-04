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
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestHNSWPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricCosine, 160)
	path := filepath.Join(t.TempDir(), "vectors.hnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	assertPrivateFileMode(t, info.Mode())

	opened, err := OpenHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWIndex(t, opened, index)
	query := []float32{7.25, 13.5, 1.25}
	want, err := index.SearchHNSW(context.Background(), query, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchHNSW(context.Background(), query, HNSWSearchOptions{
		SearchOptions: SearchOptions{TopK: 25}, EF: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened search = %#v, want %#v", got, want)
	}

	replacement := persistedHNSWIndex(t, MetricIP, 40)
	if err := replacement.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err = OpenHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWIndex(t, opened, replacement)
}

func TestHNSWPersistenceLargeGraphSearch(t *testing.T) {
	index := persistedHNSWIndex(t, MetricL2, DefaultHNSWBruteForceThreshold+100)
	path := filepath.Join(t.TempDir(), "large.hnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err := OpenHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	query := hnswBuildInputs(index.Len())[713].Vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EF: 120}
	want, err := index.SearchHNSW(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchHNSW(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("large reopened search = %#v, want %#v", got, want)
	}
}

func TestHNSWPersistenceEmpty(t *testing.T) {
	t.Parallel()
	options := DefaultHNSWBuildOptions(MetricMIPSL2)
	options.M = 4
	options.EFConstruction = 12
	options.Seed = 17
	builder, err := NewHNSWBuilder(7, options)
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "empty.hnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err := OpenHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWIndex(t, opened, index)
}

func TestHNSWPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedHNSWIndex(t, MetricL2, 32)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.hnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := index.Save(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Save error = %v", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after, original) {
		t.Fatal("canceled replacement changed published HNSW file")
	}
	if _, err := OpenHNSWIndex(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open error = %v", err)
	}
	if _, err := encodeHNSWIndex(canceled, index); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled encode error = %v", err)
	}
	if _, err := decodeHNSWIndex(canceled, original); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled decode error = %v", err)
	}
	if err := index.Save(nil, path); err == nil {
		t.Fatal("nil Save context succeeded")
	}
	if _, err := OpenHNSWIndex(nil, path); err == nil {
		t.Fatal("nil Open context succeeded")
	}
	if _, err := encodeHNSWIndex(nil, index); err == nil {
		t.Fatal("nil encode context succeeded")
	}
	if _, err := decodeHNSWIndex(nil, original); err == nil {
		t.Fatal("nil decode context succeeded")
	}
	if err := index.Save(context.Background(), ""); !errors.Is(err, ErrInvalidHNSWFile) {
		t.Fatalf("empty Save path error = %v", err)
	}
	if _, err := OpenHNSWIndex(context.Background(), ""); !errors.Is(err, ErrInvalidHNSWFile) {
		t.Fatalf("empty Open path error = %v", err)
	}
	if _, err := OpenHNSWIndex(context.Background(), filepath.Join(dir, "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing Open error = %v", err)
	}
	var nilIndex *HNSWIndex
	if err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil.hnsw")); !errors.Is(err, ErrInvalidHNSWFile) {
		t.Fatalf("nil index Save error = %v", err)
	}
	invalid := &HNSWIndex{
		dimension: 3,
		options:   DefaultHNSWBuildOptions(MetricL2),
		keys:      []uint64{1},
	}
	if err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.hnsw")); !errors.Is(err, ErrInvalidHNSWFile) {
		t.Fatalf("invalid index Save error = %v", err)
	}
}

func TestHNSWPersistenceDetectsTruncationAndCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(t, MetricL2, 32))
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{0, 1, hnswHeaderSize - 1, hnswHeaderSize, len(valid) - 1} {
		if _, err := decodeHNSWIndex(context.Background(), valid[:cut]); !errors.Is(err, ErrInvalidHNSWFile) {
			t.Fatalf("cut %d error = %v", cut, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	if _, err := decodeHNSWIndex(context.Background(), trailing); !errors.Is(err, ErrInvalidHNSWFile) {
		t.Fatalf("trailing data error = %v", err)
	}
	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	if _, err := decodeHNSWIndex(context.Background(), badMagic); !errors.Is(err, ErrInvalidHNSWFile) {
		t.Fatalf("magic error = %v", err)
	}
	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], hnswFileVersion+1)
	if _, err := decodeHNSWIndex(context.Background(), badVersion); !errors.Is(err, ErrUnsupportedHNSWVersion) {
		t.Fatalf("version error = %v", err)
	}
	badHeaderCRC := slices.Clone(valid)
	badHeaderCRC[44] ^= 1
	if _, err := decodeHNSWIndex(context.Background(), badHeaderCRC); !errors.Is(err, ErrHNSWChecksumMismatch) {
		t.Fatalf("header checksum error = %v", err)
	}
	badPayloadCRC := slices.Clone(valid)
	badPayloadCRC[len(badPayloadCRC)-1] ^= 1
	if _, err := decodeHNSWIndex(context.Background(), badPayloadCRC); !errors.Is(err, ErrHNSWChecksumMismatch) {
		t.Fatalf("payload checksum error = %v", err)
	}
}

func TestHNSWPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(t, MetricL2, 32))
	if err != nil {
		t.Fatal(err)
	}
	records := parseHNSWRecordOffsets(t, valid)
	if len(records) < 2 || len(records[0].neighborOffsets) < 2 {
		t.Fatal("persistence fixture lacks required graph edges")
	}
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
	if upperNeighborOffset < 0 || lowLevelPosition < 0 {
		t.Fatal("persistence fixture lacks an upper edge or level-zero target")
	}

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
			if _, err := decodeHNSWIndex(context.Background(), encoded); !errors.Is(err, ErrInvalidHNSWFile) {
				t.Fatalf("error = %v", err)
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
			if _, err := decodeHNSWIndex(context.Background(), encoded); !errors.Is(err, ErrInvalidHNSWFile) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func FuzzDecodeHNSWIndex(f *testing.F) {
	valid, err := encodeHNSWIndex(context.Background(), persistedHNSWIndex(f, MetricL2, 12))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte("ZVECHNSW"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeHNSWIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		if err := validateHNSWIndex(context.Background(), index); err != nil {
			t.Fatalf("decoded invalid index: %v", err)
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
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range hnswBuildInputs(count) {
		if err := builder.Add(context.Background(), input.Key, input.Vector); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return index
}

func assertSameHNSWIndex(t testing.TB, got, want *HNSWIndex) {
	t.Helper()
	if got.Dimension() != want.Dimension() || got.Metric() != want.Metric() || got.Len() != want.Len() ||
		got.BuildOptions() != want.BuildOptions() || got.entryPoint != want.entryPoint ||
		got.MaxLevel() != want.MaxLevel() || got.levelRNGState != want.levelRNGState ||
		!slices.Equal(got.keys, want.keys) || !slices.Equal(got.levels, want.levels) ||
		!reflect.DeepEqual(got.neighbors, want.neighbors) {
		t.Fatalf("reopened HNSW metadata differs\ngot:  %#v\nwant: %#v", got, want)
	}
	for _, key := range want.keys {
		gotVector, gotOK := got.Vector(key)
		wantVector, wantOK := want.Vector(key)
		if gotOK != wantOK || !slices.Equal(gotVector, wantVector) {
			t.Fatalf("key %d differs after reopen", key)
		}
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
	if offset != len(encoded) {
		t.Fatalf("fixture parse ended at %d, want %d", offset, len(encoded))
	}
	return records
}

func rechecksumHNSW(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[84:88], ailego.CRC32C(encoded[hnswHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
