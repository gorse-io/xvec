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
	"bytes"
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

func TestDiskANNPersistenceRoundTripSearchCacheAndReplace(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricCosine)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 20
	candidates := diskANNIndexCandidates(96, 8)
	index := buildDiskANNIndex(t, candidates, options)
	dir := t.TempDir()
	path := filepath.Join(dir, "vectors.diskann")
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
	opened, err := OpenDiskANNIndex(context.Background(), path, 24, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if opened.Dimension() != index.Dimension() || opened.Metric() != index.Metric() || opened.Len() != index.Len() ||
		opened.PQChunks() != index.PQChunks() || !reflect.DeepEqual(opened.BuildOptions(), DiskANNBuildOptions{
		Metric: MetricCosine, MaxDegree: 8, ListSize: 24, PQChunks: 4, Workers: 4, CacheCapacity: 24,
	}) {
		t.Fatalf("opened metadata = %#v", opened.BuildOptions())
	}
	query := candidates[31].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 15}, ListSize: 72}
	want, err := index.SearchDiskANN(context.Background(), query, search)
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchDiskANN(context.Background(), query, search)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("opened search = %#v, want %#v", got, want)
	}
	if warmed, err := opened.WarmCache(context.Background(), 10); err != nil || warmed != 10 {
		t.Fatalf("warm cache = %d, %v", warmed, err)
	}
	vector, found := opened.Vector(candidates[40].Key)
	if !found || !slices.Equal(vector, candidates[40].Vector) {
		t.Fatal("opened original vector differs")
	}

	copyPath := filepath.Join(dir, "copy.diskann")
	if err := opened.Save(context.Background(), copyPath); err != nil {
		t.Fatal(err)
	}
	copyIndex, err := OpenDiskANNIndex(context.Background(), copyPath, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	copyResults, err := copyIndex.SearchDiskANN(context.Background(), query, search)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(copyResults, want) {
		t.Fatal("saved-opened index changed results")
	}
	if err := copyIndex.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := copyIndex.SearchDiskANN(context.Background(), query, search); !errors.Is(err, ErrDiskANNClosed) {
		t.Fatalf("post-close search error = %v", err)
	}

	replacementOptions := DefaultDiskANNBuildOptions(MetricIP)
	replacementOptions.MaxDegree, replacementOptions.ListSize, replacementOptions.PQChunks = 5, 12, 2
	replacement := buildDiskANNIndex(t, diskANNIndexCandidates(40, 6), replacementOptions)
	if err := replacement.Save(context.Background(), copyPath); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenDiskANNIndex(context.Background(), copyPath, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if reopened.Metric() != MetricIP || reopened.Dimension() != 6 || reopened.Len() != 40 {
		t.Fatalf("replacement metadata = metric %d dimension %d len %d", reopened.Metric(), reopened.Dimension(), reopened.Len())
	}
}

func TestDiskANNPersistenceMIPSL2TraversalState(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricMIPSL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 6, 16, 3
	candidates := diskANNIndexCandidates(48, 6)
	index := buildDiskANNIndex(t, candidates, options)
	path := filepath.Join(t.TempDir(), "mips.diskann")
	if err := index.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	opened, err := OpenDiskANNIndex(context.Background(), path, 8, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if len(opened.codeNorms) != len(candidates) || opened.traversalMetric != MetricL2 {
		t.Fatalf("MIPSL2 traversal state = norms %d metric %d", len(opened.codeNorms), opened.traversalMetric)
	}
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: len(candidates)}
	want, err := index.SearchDiskANN(context.Background(), candidates[19].Vector, search)
	if err != nil {
		t.Fatal(err)
	}
	got, err := opened.SearchDiskANN(context.Background(), candidates[19].Vector, search)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened MIPSL2 = %#v, want %#v", got, want)
	}
}

func TestDiskANNPersistenceEmptyCancellationAndErrors(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize = 4, 8
	empty := buildDiskANNIndex(t, nil, options)
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.diskann")
	if err := empty.Save(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := OpenDiskANNIndex(context.Background(), path, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if opened.Len() != 0 || opened.PQChunks() != 0 {
		t.Fatal("empty metadata differs")
	}
	if err := opened.Close(); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := empty.Save(canceled, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Save error = %v", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after, original) {
		t.Fatal("canceled Save changed artifact")
	}
	nonEmptyOptions := DefaultDiskANNBuildOptions(MetricL2)
	nonEmptyOptions.MaxDegree, nonEmptyOptions.ListSize, nonEmptyOptions.PQChunks = 6, 16, 4
	nonEmpty := buildDiskANNIndex(t, diskANNIndexCandidates(96, 8), nonEmptyOptions)
	midSave := newCancelAfterChecks(5)
	if err := nonEmpty.Save(midSave, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-save cancellation error = %v", err)
	}
	after, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after, original) {
		t.Fatal("mid-save cancellation changed artifact")
	}
	midOpen := newCancelAfterChecks(5)
	if _, err := OpenDiskANNIndex(midOpen, path, 0, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-open cancellation error = %v", err)
	}
	if _, err := OpenDiskANNIndex(canceled, path, 0, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Open error = %v", err)
	}
	if err := empty.Save(nil, path); err == nil {
		t.Fatal("nil Save context succeeded")
	}
	if _, err := OpenDiskANNIndex(nil, path, 0, 0); err == nil {
		t.Fatal("nil Open context succeeded")
	}
	if err := empty.Save(context.Background(), ""); !errors.Is(err, ErrInvalidDiskANNFile) {
		t.Fatalf("empty Save path error = %v", err)
	}
	if _, err := OpenDiskANNIndex(context.Background(), "", 0, 0); !errors.Is(err, ErrInvalidDiskANNFile) {
		t.Fatalf("empty Open path error = %v", err)
	}
	if _, err := OpenDiskANNIndex(context.Background(), filepath.Join(dir, "missing"), 0, 0); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing Open error = %v", err)
	}
	if _, err := OpenDiskANNIndex(context.Background(), path, -1, 0); !errors.Is(err, ErrInvalidDiskANNOptions) {
		t.Fatalf("negative cache error = %v", err)
	}
	var nilIndex *DiskANNIndex
	if err := nilIndex.Save(context.Background(), path); !errors.Is(err, ErrInvalidDiskANNFile) {
		t.Fatalf("nil Save error = %v", err)
	}
}

func TestDiskANNPersistenceDetectsHeaderSectionAndRecordCorruption(t *testing.T) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 5, 12, 3
	valid, err := encodeDiskANNIndex(context.Background(), buildDiskANNIndex(t, diskANNIndexCandidates(24, 6), options))
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{0, 1, diskANNIndexHeaderSize - 1, diskANNIndexHeaderSize, len(valid) - 1} {
		if _, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(valid[:cut]), int64(cut), 0, 1, nil); err == nil {
			t.Fatalf("cut %d succeeded", cut)
		}
	}
	trailing := append(slices.Clone(valid), 0)
	if _, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(trailing), int64(len(trailing)), 0, 1, nil); !errors.Is(err, ErrInvalidDiskANNFile) {
		t.Fatalf("trailing error = %v", err)
	}
	badMagic := slices.Clone(valid)
	badMagic[0] ^= 1
	if _, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badMagic), int64(len(badMagic)), 0, 1, nil); !errors.Is(err, ErrInvalidDiskANNFile) {
		t.Fatalf("magic error = %v", err)
	}
	badVersion := slices.Clone(valid)
	binary.LittleEndian.PutUint16(badVersion[8:10], diskANNIndexFileVersion+1)
	if _, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badVersion), int64(len(badVersion)), 0, 1, nil); !errors.Is(err, ErrUnsupportedDiskANNIndexVersion) {
		t.Fatalf("version error = %v", err)
	}
	badHeader := slices.Clone(valid)
	badHeader[40] ^= 1
	if _, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(badHeader), int64(len(badHeader)), 0, 1, nil); !errors.Is(err, ErrDiskANNIndexChecksumMismatch) {
		t.Fatalf("header checksum error = %v", err)
	}

	for _, offset := range []int{
		int(binary.LittleEndian.Uint64(valid[64:72])),
		int(binary.LittleEndian.Uint64(valid[80:88])),
		int(binary.LittleEndian.Uint64(valid[96:104])),
		int(binary.LittleEndian.Uint64(valid[112:120])),
		int(binary.LittleEndian.Uint64(valid[128:136])),
	} {
		corrupt := slices.Clone(valid)
		corrupt[offset] ^= 1
		if _, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(corrupt), int64(len(corrupt)), 0, 1, nil); !errors.Is(err, ErrDiskANNIndexChecksumMismatch) {
			t.Fatalf("section offset %d error = %v", offset, err)
		}
	}

	// Refresh both whole-section checksums but leave the node record checksum
	// stale. Open accepts the complete sections; the random read detects the
	// damaged record at its own integrity boundary.
	recordCorrupt := slices.Clone(valid)
	nodesOffset := int(binary.LittleEndian.Uint64(recordCorrupt[128:136]))
	nodesLength := int(binary.LittleEndian.Uint64(recordCorrupt[136:144]))
	recordCorrupt[nodesOffset+diskANNNodeHeaderSize+4] ^= 1
	nodeHeader := recordCorrupt[nodesOffset : nodesOffset+diskANNNodeHeaderSize]
	nodeData := recordCorrupt[nodesOffset+diskANNNodeHeaderSize : nodesOffset+nodesLength]
	binary.LittleEndian.PutUint32(nodeHeader[72:76], ailego.CRC32C(nodeData))
	binary.LittleEndian.PutUint32(nodeHeader[diskANNNodeHeaderCRCPos:], ailego.CRC32C(nodeHeader[:diskANNNodeHeaderCRCPos]))
	binary.LittleEndian.PutUint32(recordCorrupt[160:164], ailego.CRC32C(recordCorrupt[nodesOffset:nodesOffset+nodesLength]))
	binary.LittleEndian.PutUint32(recordCorrupt[diskANNIndexHeaderCRCPos:], ailego.CRC32C(recordCorrupt[:diskANNIndexHeaderCRCPos]))
	opened, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(recordCorrupt), int64(len(recordCorrupt)), 0, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := opened.nodes.ReadNode(context.Background(), 0); !errors.Is(err, ErrDiskANNChecksumMismatch) {
		t.Fatalf("record checksum error = %v", err)
	}
}

func FuzzDiskANNIndexFile(f *testing.F) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks = 3, 6, 2
	valid, err := encodeDiskANNIndex(context.Background(), buildDiskANNIndex(f, diskANNIndexCandidates(8, 4), options))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte("not-a-diskann-index"))
	f.Fuzz(func(t *testing.T, encoded []byte) {
		if len(encoded) > 4<<20 {
			return
		}
		index, err := openDiskANNIndexReader(context.Background(), bytes.NewReader(encoded), int64(len(encoded)), 0, 1, nil)
		if err == nil {
			_ = index.Close()
		}
	})
}

func BenchmarkDiskANNSearchWarmCache(b *testing.B) {
	options := DefaultDiskANNBuildOptions(MetricL2)
	options.MaxDegree, options.ListSize, options.PQChunks, options.CacheCapacity = 8, 24, 4, 128
	candidates := diskANNIndexCandidates(128, 8)
	index := buildDiskANNIndex(b, candidates, options)
	_, _ = index.WarmCache(context.Background(), 128)
	query := candidates[27].Vector
	search := DiskANNSearchOptions{SearchOptions: SearchOptions{TopK: 10}, ListSize: 48}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := index.SearchDiskANN(context.Background(), query, search); err != nil {
			b.Fatal(err)
		}
	}
}
