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

func TestSparseHNSWPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedSparseHNSWIndex(t, 160)
	path := filepath.Join(t.TempDir(), "vectors.shnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameSparseHNSWIndex(t, opened, index)
	query := SparseVector{Indices: []uint32{3, 107, 211}, Values: []float32{1, 2, 3}}
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 25}, EF: 80}
	want, err := index.SearchSparseHNSW(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchSparseHNSW(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened search = %#v, want %#v", got, want)
	}

	replacement := persistedSparseHNSWIndex(t, 40)
	if err := replacement.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err = OpenSparseHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameSparseHNSWIndex(t, opened, replacement)
}

func TestSparseHNSWPersistenceLargeGraphSearch(t *testing.T) {
	index := persistedSparseHNSWIndex(t, DefaultHNSWBruteForceThreshold+100)
	path := filepath.Join(t.TempDir(), "large.shnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	query := sparseHNSWBuildInputs(index.Len())[713].vector
	options := HNSWSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EF: 120}
	want, err := index.SearchSparseHNSW(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchSparseHNSW(context.Background(), query, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("large reopened search = %#v, want %#v", got, want)
	}
}

func TestSparseHNSWPersistenceEmpty(t *testing.T) {
	t.Parallel()
	builder, err := NewSparseHNSWBuilder(DefaultSparseHNSWBuildOptions())
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "empty.shnsw")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err := OpenSparseHNSWIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameSparseHNSWIndex(t, opened, index)
}

func TestSparseHNSWPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedSparseHNSWIndex(t, 32)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.shnsw")
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
	after, _ := os.ReadFile(path)
	if !slices.Equal(after, original) {
		t.Fatal("canceled replacement changed published file")
	}
	if _, err := OpenSparseHNSWIndex(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open error = %v", err)
	}
	if err := index.Save(nil, path); err == nil {
		t.Fatal("nil Save context succeeded")
	}
	if _, err := OpenSparseHNSWIndex(nil, path); err == nil {
		t.Fatal("nil Open context succeeded")
	}
	if err := index.Save(context.Background(), ""); !errors.Is(err, ErrInvalidSparseHNSWFile) {
		t.Fatalf("empty Save path error = %v", err)
	}
	if _, err := OpenSparseHNSWIndex(context.Background(), ""); !errors.Is(err, ErrInvalidSparseHNSWFile) {
		t.Fatalf("empty Open path error = %v", err)
	}
	if _, err := OpenSparseHNSWIndex(context.Background(), filepath.Join(dir, "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing Open error = %v", err)
	}
	var nilIndex *SparseHNSWIndex
	if err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil.shnsw")); !errors.Is(err, ErrInvalidSparseHNSWFile) {
		t.Fatalf("nil index Save error = %v", err)
	}
	invalid, err := cloneSparseHNSWIndex(context.Background(), index)
	if err != nil {
		t.Fatal(err)
	}
	invalid.offsets[1] = len(invalid.indices) + 1
	if err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.shnsw")); !errors.Is(err, ErrInvalidSparseHNSWFile) {
		t.Fatalf("invalid offsets Save error = %v", err)
	}
}

func TestSparseHNSWPersistenceDetectsCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(t, 32))
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{0, 1, sparseHNSWHeaderSize - 1, sparseHNSWHeaderSize, len(valid) - 1} {
		if _, err := decodeSparseHNSWIndex(context.Background(), valid[:cut]); !errors.Is(err, ErrInvalidSparseHNSWFile) {
			t.Fatalf("cut %d error = %v", cut, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	if _, err := decodeSparseHNSWIndex(context.Background(), trailing); !errors.Is(err, ErrInvalidSparseHNSWFile) {
		t.Fatalf("trailing error = %v", err)
	}
	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	if _, err := decodeSparseHNSWIndex(context.Background(), badMagic); !errors.Is(err, ErrInvalidSparseHNSWFile) {
		t.Fatalf("magic error = %v", err)
	}
	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], sparseHNSWFileVersion+1)
	if _, err := decodeSparseHNSWIndex(context.Background(), badVersion); !errors.Is(err, ErrUnsupportedSparseHNSWVersion) {
		t.Fatalf("version error = %v", err)
	}
	badHeader := slices.Clone(valid)
	badHeader[48] ^= 1
	if _, err := decodeSparseHNSWIndex(context.Background(), badHeader); !errors.Is(err, ErrSparseHNSWChecksumMismatch) {
		t.Fatalf("header checksum error = %v", err)
	}
	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	if _, err := decodeSparseHNSWIndex(context.Background(), badPayload); !errors.Is(err, ErrSparseHNSWChecksumMismatch) {
		t.Fatalf("payload checksum error = %v", err)
	}
}

func TestSparseHNSWPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(t, 32))
	if err != nil {
		t.Fatal(err)
	}
	records := parseSparseHNSWRecordOffsets(t, valid)
	if len(records) < 2 || len(records[0].coordinates) < 2 || records[0].neighbor < 0 {
		t.Fatal("fixture lacks sparse elements or graph edges")
	}
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
			if _, err := decodeSparseHNSWIndex(context.Background(), encoded); !errors.Is(err, ErrInvalidSparseHNSWFile) {
				t.Fatalf("error = %v", err)
			}
		})
	}
	headerTests := []func([]byte){
		func(data []byte) { binary.LittleEndian.PutUint32(data[48:52], 0) },
		func(data []byte) { binary.LittleEndian.PutUint64(data[60:68], uint64(len(records))) },
		func(data []byte) { data[92] = 1 },
		func(data []byte) { binary.LittleEndian.PutUint64(data[16:24], uint64(len(data)+1)) },
	}
	for index, mutate := range headerTests {
		encoded := slices.Clone(valid)
		mutate(encoded)
		binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
		if _, err := decodeSparseHNSWIndex(context.Background(), encoded); !errors.Is(err, ErrInvalidSparseHNSWFile) {
			t.Fatalf("header mutation %d error = %v", index, err)
		}
	}
}

func FuzzDecodeSparseHNSWIndex(f *testing.F) {
	valid, err := encodeSparseHNSWIndex(context.Background(), persistedSparseHNSWIndex(f, 12))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte("ZVSPHNSW"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeSparseHNSWIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		if err := validateSparseHNSWIndex(context.Background(), index); err != nil {
			t.Fatalf("decoded invalid index: %v", err)
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
	if got.Metric() != want.Metric() || got.Len() != want.Len() || got.BuildOptions() != want.BuildOptions() ||
		got.entryPoint != want.entryPoint || got.MaxLevel() != want.MaxLevel() || got.levelRNGState != want.levelRNGState ||
		!slices.Equal(got.keys, want.keys) || !slices.Equal(got.offsets, want.offsets) ||
		!slices.Equal(got.indices, want.indices) || !slices.Equal(got.values, want.values) ||
		!slices.Equal(got.levels, want.levels) || !reflect.DeepEqual(got.neighbors, want.neighbors) {
		t.Fatalf("reopened sparse HNSW metadata differs")
	}
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
	if offset != len(encoded) {
		t.Fatalf("fixture parse ended at %d, want %d", offset, len(encoded))
	}
	return records
}

func rechecksumSparseHNSW(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[88:92], ailego.CRC32C(encoded[sparseHNSWHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
