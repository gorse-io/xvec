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
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestHNSWRaBitQPersistenceRoundTripReplaceAndIncrement(t *testing.T) {
	index := buildHNSWRaBitQ(t, hnswRaBitQCandidates(180, 70), hnswRaBitQTestOptions(MetricCosine))
	path := filepath.Join(t.TempDir(), "vectors.hrbtq")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	assertPrivateFileMode(t, info.Mode())
	opened, err := OpenHNSWRaBitQIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWRaBitQIndex(t, opened, index)
	query := hnswRaBitQCandidates(1, 70)[0].Vector
	options := HNSWRaBitQSearchOptions{SearchOptions: SearchOptions{TopK: 25}, EF: 80}
	for _, refine := range []bool{false, true} {
		options.Refine = refine
		want, err := index.SearchHNSWRaBitQ(context.Background(), query, options)
		if err != nil {
			t.Fatal(err)
		}
		got, err := opened.SearchHNSWRaBitQ(context.Background(), query, options)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("refine %v reopened search = %#v, want %#v", refine, got, want)
		}
	}

	next := hnswRaBitQCandidates(1, 70)[0]
	next.Key = 999999
	if err := opened.Add(context.Background(), next.Key, next.Vector); err != nil {
		t.Fatal(err)
	}
	if err := opened.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenHNSWRaBitQIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWRaBitQIndex(t, reopened, opened)

	replacement := buildHNSWRaBitQ(t, hnswRaBitQCandidates(40, 64), hnswRaBitQTestOptions(MetricIP))
	if err := replacement.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err = OpenHNSWRaBitQIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWRaBitQIndex(t, reopened, replacement)
}

func TestHNSWRaBitQPersistenceEmptyCancellationAndErrors(t *testing.T) {
	builder, err := NewHNSWRaBitQBuilder(64, hnswRaBitQTestOptions(MetricL2))
	if err != nil {
		t.Fatal(err)
	}
	index, err := builder.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.hrbtq")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := OpenHNSWRaBitQIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameHNSWRaBitQIndex(t, opened, index)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := index.Save(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Save error = %v", err)
	}
	after, _ := os.ReadFile(path)
	if !slices.Equal(after, original) {
		t.Fatal("canceled Save changed published artifact")
	}
	if _, err := OpenHNSWRaBitQIndex(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open error = %v", err)
	}
	if _, err := encodeHNSWRaBitQIndex(canceled, index); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled encode error = %v", err)
	}
	if _, err := decodeHNSWRaBitQIndex(canceled, original); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled decode error = %v", err)
	}
	if err := index.Save(nil, path); err == nil {
		t.Fatal("nil Save context succeeded")
	}
	if _, err := OpenHNSWRaBitQIndex(nil, path); err == nil {
		t.Fatal("nil Open context succeeded")
	}
	if err := index.Save(context.Background(), ""); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("empty Save path error = %v", err)
	}
	if _, err := OpenHNSWRaBitQIndex(context.Background(), ""); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("empty Open path error = %v", err)
	}
	if _, err := OpenHNSWRaBitQIndex(context.Background(), filepath.Join(dir, "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing Open error = %v", err)
	}
	var nilIndex *HNSWRaBitQIndex
	if err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil")); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("nil Save error = %v", err)
	}
}

func TestHNSWRaBitQPersistenceDetectsCorruption(t *testing.T) {
	valid, err := encodeHNSWRaBitQIndex(context.Background(), buildHNSWRaBitQ(t, hnswRaBitQCandidates(32, 64), hnswRaBitQTestOptions(MetricL2)))
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{0, 1, hnswRaBitQHeaderSize - 1, hnswRaBitQHeaderSize, len(valid) - 1} {
		if _, err := decodeHNSWRaBitQIndex(context.Background(), valid[:cut]); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
			t.Fatalf("cut %d error = %v", cut, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	if _, err := decodeHNSWRaBitQIndex(context.Background(), trailing); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("trailing error = %v", err)
	}
	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	if _, err := decodeHNSWRaBitQIndex(context.Background(), badMagic); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("magic error = %v", err)
	}
	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], hnswRaBitQFileVersion+1)
	if _, err := decodeHNSWRaBitQIndex(context.Background(), badVersion); !errors.Is(err, ErrUnsupportedHNSWRaBitQVersion) {
		t.Fatalf("version error = %v", err)
	}
	badHeader := slices.Clone(valid)
	badHeader[80] ^= 1
	if _, err := decodeHNSWRaBitQIndex(context.Background(), badHeader); !errors.Is(err, ErrHNSWRaBitQChecksumMismatch) {
		t.Fatalf("header checksum error = %v", err)
	}
	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	if _, err := decodeHNSWRaBitQIndex(context.Background(), badPayload); !errors.Is(err, ErrHNSWRaBitQChecksumMismatch) {
		t.Fatalf("payload checksum error = %v", err)
	}

	badFingerprint := slices.Clone(valid)
	badFingerprint[72] ^= 1
	refreshHNSWRaBitQChecksums(badFingerprint)
	if _, err := decodeHNSWRaBitQIndex(context.Background(), badFingerprint); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("fingerprint error = %v", err)
	}
	badCluster := slices.Clone(valid)
	baseLength := int(binary.LittleEndian.Uint64(badCluster[32:40]))
	dimension := int(binary.LittleEndian.Uint32(badCluster[48:52]))
	centroids := int(binary.LittleEndian.Uint32(badCluster[56:60]))
	rotationBytes := int(binary.LittleEndian.Uint32(badCluster[60:64]))
	codeOffset := hnswRaBitQHeaderSize + baseLength + centroids*dimension*4 + rotationBytes
	binary.LittleEndian.PutUint32(badCluster[codeOffset:codeOffset+4], mathMaxUint32)
	refreshHNSWRaBitQChecksums(badCluster)
	if _, err := decodeHNSWRaBitQIndex(context.Background(), badCluster); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("cluster error = %v", err)
	}
	badGraph := slices.Clone(valid)
	badGraph[hnswRaBitQHeaderSize+hnswHeaderSize] ^= 1
	refreshHNSWRaBitQChecksums(badGraph)
	if _, err := decodeHNSWRaBitQIndex(context.Background(), badGraph); !errors.Is(err, ErrInvalidHNSWRaBitQFile) {
		t.Fatalf("nested graph error = %v", err)
	}
}

func FuzzHNSWRaBitQDecode(f *testing.F) {
	valid, err := encodeHNSWRaBitQIndex(context.Background(), buildHNSWRaBitQ(f, hnswRaBitQCandidates(8, 64), hnswRaBitQTestOptions(MetricL2)))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte("ZVECHR BQ"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeHNSWRaBitQIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		if err := validateHNSWRaBitQIndex(context.Background(), index); err != nil {
			t.Fatal(err)
		}
	})
}

const mathMaxUint32 = ^uint32(0)

func refreshHNSWRaBitQChecksums(encoded []byte) {
	header := encoded[:hnswRaBitQHeaderSize]
	payload := encoded[hnswRaBitQHeaderSize:]
	binary.LittleEndian.PutUint32(header[112:116], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
}

func assertSameHNSWRaBitQIndex(t testing.TB, got, want *HNSWRaBitQIndex) {
	t.Helper()
	if got.options != want.options || !reflect.DeepEqual(got.ModelState(), want.ModelState()) ||
		!slices.Equal(got.base.keys, want.base.keys) || !slices.Equal(got.base.vectors, want.base.vectors) ||
		!slices.Equal(got.base.levels, want.base.levels) || len(got.base.neighbors) != len(want.base.neighbors) ||
		got.base.entryPoint != want.base.entryPoint || got.base.maxLevel != want.base.maxLevel ||
		got.base.levelRNGState != want.base.levelRNGState || len(got.codes) != len(want.codes) {
		t.Fatal("HNSW-RaBitQ indexes differ")
	}
	for position := range got.base.neighbors {
		if len(got.base.neighbors[position]) != len(want.base.neighbors[position]) {
			t.Fatal("HNSW-RaBitQ level counts differ")
		}
		for level := range got.base.neighbors[position] {
			if !slices.Equal(got.base.neighbors[position][level], want.base.neighbors[position][level]) {
				t.Fatal("HNSW-RaBitQ neighbors differ")
			}
		}
	}
	for position := range got.codes {
		if !reflect.DeepEqual(got.codes[position], want.codes[position]) {
			t.Fatal("HNSW-RaBitQ codes differ")
		}
	}
}
