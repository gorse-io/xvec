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

func TestVamanaPersistenceRoundTripReplaceAndIncrement(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricCosine)
	options.MaxDegree, options.SearchListSize, options.MaxOcclusionSize = 8, 40, 80
	index := buildVamana(t, hnswRaBitQCandidates(180, 70), options)
	path := filepath.Join(t.TempDir(), "vectors.vamana")
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
	opened, err := OpenVamanaIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameVamanaIndex(t, opened, index)
	query := hnswRaBitQCandidates(1, 70)[0].Vector
	search := VamanaSearchOptions{SearchOptions: SearchOptions{TopK: 20}, EFSearch: 80}
	want, err := index.SearchVamana(context.Background(), query, search)
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchVamana(context.Background(), query, search)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened search = %#v, want %#v", got, want)
	}

	next := hnswRaBitQCandidates(1, 70)[0]
	next.Key = 999999
	if err := opened.Add(context.Background(), next.Key, next.Vector); err != nil {
		t.Fatal(err)
	}
	if err := opened.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenVamanaIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameVamanaIndex(t, reopened, opened)

	replacementOptions := DefaultVamanaBuildOptions(MetricIP)
	replacementOptions.MaxDegree, replacementOptions.SearchListSize = 4, 16
	replacement := buildVamana(t, hnswBuildInputs(40), replacementOptions)
	if err := replacement.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	reopened, err = OpenVamanaIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameVamanaIndex(t, reopened, replacement)
}

func TestVamanaPersistenceEmptyCancellationAndErrors(t *testing.T) {
	index := buildVamana(t, nil, DefaultVamanaBuildOptions(MetricL2))
	largeOptions := DefaultVamanaBuildOptions(MetricL2)
	largeOptions.MaxDegree, largeOptions.SearchListSize = 8, 40
	largeIndex := buildVamana(t, hnswRaBitQCandidates(300, 70), largeOptions)
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.vamana")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := OpenVamanaIndex(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	assertSameVamanaIndex(t, opened, index)
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
		t.Fatal("canceled Save changed artifact")
	}
	if _, err := OpenVamanaIndex(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open error = %v", err)
	}
	midSave := newCancelAfterChecks(7)
	if err := largeIndex.Save(midSave, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-save cancellation error = %v", err)
	}
	after, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after, original) {
		t.Fatal("mid-save cancellation changed artifact")
	}
	midEncode := newCancelAfterChecks(5)
	if _, err := encodeVamanaIndex(midEncode, largeIndex); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-encode cancellation error = %v", err)
	}
	largeEncoded, err := encodeVamanaIndex(context.Background(), largeIndex)
	if err != nil {
		t.Fatal(err)
	}
	midDecode := newCancelAfterChecks(3)
	if _, err := decodeVamanaIndex(midDecode, largeEncoded); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-decode cancellation error = %v", err)
	}
	if err := index.Save(nil, path); err == nil {
		t.Fatal("nil Save context succeeded")
	}
	if _, err := OpenVamanaIndex(nil, path); err == nil {
		t.Fatal("nil Open context succeeded")
	}
	if _, err := encodeVamanaIndex(nil, index); err == nil {
		t.Fatal("nil encode context succeeded")
	}
	if _, err := decodeVamanaIndex(nil, original); err == nil {
		t.Fatal("nil decode context succeeded")
	}
	if err := index.Save(context.Background(), ""); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("empty Save path error = %v", err)
	}
	if _, err := OpenVamanaIndex(context.Background(), ""); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("empty Open path error = %v", err)
	}
	if _, err := OpenVamanaIndex(context.Background(), filepath.Join(dir, "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing Open error = %v", err)
	}
	var nilIndex *VamanaIndex
	if err := nilIndex.Save(context.Background(), filepath.Join(dir, "nil")); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("nil Save error = %v", err)
	}
}

func TestVamanaPersistenceDetectsCorruption(t *testing.T) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 4, 16
	valid, err := encodeVamanaIndex(context.Background(), buildVamana(t, hnswBuildInputs(32), options))
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{0, 1, vamanaHeaderSize - 1, vamanaHeaderSize, len(valid) - 1} {
		if _, err := decodeVamanaIndex(context.Background(), valid[:cut]); !errors.Is(err, ErrInvalidVamanaFile) {
			t.Fatalf("cut %d error = %v", cut, err)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	if _, err := decodeVamanaIndex(context.Background(), trailing); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("trailing error = %v", err)
	}
	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	if _, err := decodeVamanaIndex(context.Background(), badMagic); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("magic error = %v", err)
	}
	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], vamanaFileVersion+1)
	if _, err := decodeVamanaIndex(context.Background(), badVersion); !errors.Is(err, ErrUnsupportedVamanaVersion) {
		t.Fatalf("version error = %v", err)
	}
	badHeader := slices.Clone(valid)
	badHeader[56] ^= 1
	if _, err := decodeVamanaIndex(context.Background(), badHeader); !errors.Is(err, ErrVamanaChecksumMismatch) {
		t.Fatalf("header checksum error = %v", err)
	}
	badPayload := slices.Clone(valid)
	badPayload[len(badPayload)-1] ^= 1
	if _, err := decodeVamanaIndex(context.Background(), badPayload); !errors.Is(err, ErrVamanaChecksumMismatch) {
		t.Fatalf("payload checksum error = %v", err)
	}

	badSelfLoop := slices.Clone(valid)
	count := int(binary.LittleEndian.Uint64(badSelfLoop[32:40]))
	dimension := int(binary.LittleEndian.Uint32(badSelfLoop[48:52]))
	adjacencyOffset := vamanaHeaderSize + count*8 + count*dimension*4
	degree := int(binary.LittleEndian.Uint32(badSelfLoop[adjacencyOffset : adjacencyOffset+4]))
	if degree == 0 {
		t.Fatal("fixture entry has no neighbors")
	}
	binary.LittleEndian.PutUint32(badSelfLoop[adjacencyOffset+4:adjacencyOffset+8], 0)
	refreshVamanaChecksums(badSelfLoop)
	if _, err := decodeVamanaIndex(context.Background(), badSelfLoop); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("self-loop error = %v", err)
	}

	badDegree := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badDegree[adjacencyOffset:adjacencyOffset+4], uint32(options.MaxDegree+1))
	refreshVamanaChecksums(badDegree)
	if _, err := decodeVamanaIndex(context.Background(), badDegree); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("degree error = %v", err)
	}

	badDuplicateKey := slices.Clone(valid)
	copy(badDuplicateKey[vamanaHeaderSize+8:vamanaHeaderSize+16], badDuplicateKey[vamanaHeaderSize:vamanaHeaderSize+8])
	refreshVamanaChecksums(badDuplicateKey)
	if _, err := decodeVamanaIndex(context.Background(), badDuplicateKey); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("duplicate key error = %v", err)
	}

	badVector := slices.Clone(valid)
	vectorOffset := vamanaHeaderSize + count*8
	binary.LittleEndian.PutUint32(badVector[vectorOffset:vectorOffset+4], math.Float32bits(float32(math.NaN())))
	refreshVamanaChecksums(badVector)
	if _, err := decodeVamanaIndex(context.Background(), badVector); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("non-finite vector error = %v", err)
	}

	badEntry := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badEntry[72:80], uint64(count))
	refreshVamanaChecksums(badEntry)
	if _, err := decodeVamanaIndex(context.Background(), badEntry); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("entry point error = %v", err)
	}

	badNeighbor := slices.Clone(valid)
	binary.LittleEndian.PutUint32(badNeighbor[adjacencyOffset+4:adjacencyOffset+8], uint32(count))
	refreshVamanaChecksums(badNeighbor)
	if _, err := decodeVamanaIndex(context.Background(), badNeighbor); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("neighbor range error = %v", err)
	}

	badDuplicateNeighbor := slices.Clone(valid)
	duplicateOffset, found := vamanaAdjacencyWithDegree(badDuplicateNeighbor, 2)
	if !found {
		t.Fatal("fixture has no node with two neighbors")
	}
	copy(badDuplicateNeighbor[duplicateOffset+8:duplicateOffset+12], badDuplicateNeighbor[duplicateOffset+4:duplicateOffset+8])
	refreshVamanaChecksums(badDuplicateNeighbor)
	if _, err := decodeVamanaIndex(context.Background(), badDuplicateNeighbor); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("duplicate neighbor error = %v", err)
	}

	badCount := slices.Clone(valid)
	binary.LittleEndian.PutUint64(badCount[32:40], uint64(math.MaxUint32)+1)
	refreshVamanaChecksums(badCount)
	if _, err := decodeVamanaIndex(context.Background(), badCount); !errors.Is(err, ErrInvalidVamanaFile) {
		t.Fatalf("node capacity error = %v", err)
	}
}

func FuzzVamanaDecode(f *testing.F) {
	options := DefaultVamanaBuildOptions(MetricL2)
	options.MaxDegree, options.SearchListSize = 4, 12
	valid, err := encodeVamanaIndex(context.Background(), buildVamana(f, hnswBuildInputs(8), options))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte("not-vamana"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		index, err := decodeVamanaIndex(context.Background(), encoded)
		if err != nil {
			return
		}
		if err := validateVamanaIndex(context.Background(), index); err != nil {
			t.Fatalf("decoded invalid index: %v", err)
		}
	})
}

func refreshVamanaChecksums(encoded []byte) {
	header := encoded[:vamanaHeaderSize]
	payload := encoded[vamanaHeaderSize:]
	binary.LittleEndian.PutUint32(header[80:84], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[124:128], ailego.CRC32C(header[:124]))
}

func vamanaAdjacencyWithDegree(encoded []byte, minimum int) (int, bool) {
	count := int(binary.LittleEndian.Uint64(encoded[32:40]))
	dimension := int(binary.LittleEndian.Uint32(encoded[48:52]))
	offset := vamanaHeaderSize + count*8 + count*dimension*4
	for range count {
		degree := int(binary.LittleEndian.Uint32(encoded[offset : offset+4]))
		if degree >= minimum {
			return offset, true
		}
		offset += 4 + degree*4
	}
	return 0, false
}

func assertSameVamanaIndex(t testing.TB, got, want *VamanaIndex) {
	t.Helper()
	if got.dimension != want.dimension || got.options != want.options || got.entryPoint != want.entryPoint ||
		!slices.Equal(got.keys, want.keys) || !slices.Equal(got.vectors, want.vectors) ||
		!reflect.DeepEqual(got.positions, want.positions) || !reflect.DeepEqual(got.neighbors, want.neighbors) ||
		!reflect.DeepEqual(got.neighborDistances, want.neighborDistances) {
		t.Fatal("Vamana indexes differ")
	}
}
