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

func TestIVFPersistenceRoundTripAndReplace(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricCosine, 3)
	path := filepath.Join(t.TempDir(), "vectors.ivf")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o077 != 0 {
		t.Fatalf("artifact mode = %o, want private", info.Mode().Perm())
	}

	opened, err := OpenIVFIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameIVFIndex(t, opened, index)
	query := []float32{0.25, 0.5, 0.75}
	want, err := index.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: index.Len()},
		NProbe:        index.NList(),
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchIVF(context.Background(), query, IVFSearchOptions{
		SearchOptions: SearchOptions{TopK: opened.Len()},
		NProbe:        opened.NList(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened search = %#v, want %#v", got, want)
	}

	replacement := persistedIVFIndex(t, MetricIP, 2)
	if err := replacement.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err = OpenIVFIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameIVFIndex(t, opened, replacement)
}

func TestIVFPersistenceEmpty(t *testing.T) {
	t.Parallel()
	options := DefaultIVFBuildOptions(MetricMIPSL2)
	options.Seed = 17
	builder, err := NewIVFBuilder(7, options)
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "empty.ivf")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err := OpenIVFIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameIVFIndex(t, opened, index)
}

func TestIVFPersistenceCancellationAndErrors(t *testing.T) {
	t.Parallel()
	index := persistedIVFIndex(t, MetricL2, 2)
	dir := t.TempDir()
	path := filepath.Join(dir, "index.ivf")
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
		t.Fatal("canceled replacement changed published IVF file")
	}
	if _, err := OpenIVFIndex(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open error = %v", err)
	}
	if err := index.Save(nil, path); err == nil {
		t.Fatal("nil Save context succeeded")
	}
	if _, err := OpenIVFIndex(nil, path); err == nil {
		t.Fatal("nil Open context succeeded")
	}
	if err := index.Save(context.Background(), ""); !errors.Is(err, ErrInvalidIVFFile) {
		t.Fatalf("empty Save path error = %v", err)
	}
	if _, err := OpenIVFIndex(context.Background(), ""); !errors.Is(err, ErrInvalidIVFFile) {
		t.Fatalf("empty Open path error = %v", err)
	}
	if _, err := OpenIVFIndex(context.Background(), filepath.Join(dir, "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing Open error = %v", err)
	}
	invalid := &IVFIndex{dimension: 3, options: DefaultIVFBuildOptions(MetricL2), keys: []uint64{1}}
	if err := invalid.Save(context.Background(), filepath.Join(dir, "invalid.ivf")); !errors.Is(err, ErrInvalidIVFFile) {
		t.Fatalf("invalid index Save error = %v", err)
	}
}

func TestIVFPersistenceDetectsTruncationAndCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeIVFIndex(context.Background(), persistedIVFIndex(t, MetricL2, 2))
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{0, 1, ivfHeaderSize - 1, ivfHeaderSize, len(valid) - 1} {
		if _, err := decodeIVFIndex(context.Background(), valid[:cut]); !errors.Is(err, ErrInvalidIVFFile) {
			t.Fatalf("cut %d error = %v", cut, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	if _, err := decodeIVFIndex(context.Background(), trailing); !errors.Is(err, ErrInvalidIVFFile) {
		t.Fatalf("trailing data error = %v", err)
	}
	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	if _, err := decodeIVFIndex(context.Background(), badMagic); !errors.Is(err, ErrInvalidIVFFile) {
		t.Fatalf("magic error = %v", err)
	}
	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], ivfFileVersion+1)
	if _, err := decodeIVFIndex(context.Background(), badVersion); !errors.Is(err, ErrUnsupportedIVFVersion) {
		t.Fatalf("version error = %v", err)
	}
	badHeaderCRC := slices.Clone(valid)
	badHeaderCRC[55] ^= 1
	if _, err := decodeIVFIndex(context.Background(), badHeaderCRC); !errors.Is(err, ErrIVFChecksumMismatch) {
		t.Fatalf("header checksum error = %v", err)
	}
	badPayloadCRC := slices.Clone(valid)
	badPayloadCRC[len(badPayloadCRC)-1] ^= 1
	if _, err := decodeIVFIndex(context.Background(), badPayloadCRC); !errors.Is(err, ErrIVFChecksumMismatch) {
		t.Fatalf("payload checksum error = %v", err)
	}
}

func TestIVFPersistenceRejectsSemanticCorruption(t *testing.T) {
	t.Parallel()
	valid, err := encodeIVFIndex(context.Background(), persistedIVFIndex(t, MetricL2, 2))
	if err != nil {
		t.Fatal(err)
	}
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
			if _, err := decodeIVFIndex(context.Background(), encoded); !errors.Is(err, ErrInvalidIVFFile) {
				t.Fatalf("error = %v", err)
			}
		})
	}

	badOptions := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badOptions[52:56], 0)
	binary.LittleEndian.PutUint32(badOptions[108:112], ailego.CRC32C(badOptions[:108]))
	if _, err := decodeIVFIndex(context.Background(), badOptions); !errors.Is(err, ErrInvalidIVFFile) {
		t.Fatalf("invalid options error = %v", err)
	}
	badLength := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badLength[16:24], uint64(len(badLength)+1))
	binary.LittleEndian.PutUint32(badLength[108:112], ailego.CRC32C(badLength[:108]))
	if _, err := decodeIVFIndex(context.Background(), badLength); !errors.Is(err, ErrInvalidIVFFile) {
		t.Fatalf("invalid length error = %v", err)
	}
}

func FuzzDecodeIVFIndex(f *testing.F) {
	index := persistedIVFIndex(f, MetricL2, 2)
	valid, err := encodeIVFIndex(context.Background(), index)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte("ZVECIVF"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeIVFIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		if err := validateIVFIndex(context.Background(), index); err != nil {
			t.Fatalf("decoded invalid index: %v", err)
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
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range []Candidate{
		{Key: 41, Vector: []float32{1, 0, 0}},
		{Key: 7, Vector: []float32{0, 1, 0}},
		{Key: 99, Vector: []float32{0, 0, 1}},
		{Key: 5, Vector: []float32{1, 1, 0}},
		{Key: 123, Vector: []float32{0.5, 0.25, 0.75}},
	} {
		if err := builder.Add(context.Background(), candidate.Key, candidate.Vector); err != nil {
			t.Fatal(err)
		}
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return index
}

func assertSameIVFIndex(t testing.TB, got, want *IVFIndex) {
	t.Helper()
	if got.Dimension() != want.Dimension() || got.Metric() != want.Metric() || got.Len() != want.Len() ||
		got.NList() != want.NList() || got.BuildOptions() != want.BuildOptions() ||
		got.TrainingCost() != want.TrainingCost() || got.TrainingIterations() != want.TrainingIterations() ||
		got.TrainingConverged() != want.TrainingConverged() ||
		!reflect.DeepEqual(got.Centroids(), want.Centroids()) || !slices.Equal(got.keys, want.keys) {
		t.Fatalf("reopened IVF metadata differs\ngot:  %#v\nwant: %#v", got, want)
	}
	for _, key := range want.keys {
		gotVector, gotOK := got.Vector(key)
		wantVector, wantOK := want.Vector(key)
		gotList, gotListOK := got.ListForKey(key)
		wantList, wantListOK := want.ListForKey(key)
		if gotOK != wantOK || !slices.Equal(gotVector, wantVector) || gotListOK != wantListOK || gotList != wantList {
			t.Fatalf("key %d differs after reopen", key)
		}
	}
}

func rechecksumIVF(encoded []byte) {
	binary.LittleEndian.PutUint32(encoded[96:100], ailego.CRC32C(encoded[ivfHeaderSize:]))
	binary.LittleEndian.PutUint32(encoded[108:112], ailego.CRC32C(encoded[:108]))
}
