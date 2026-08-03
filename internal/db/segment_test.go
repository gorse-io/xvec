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

package db

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestWriteSegmentSealAndOpen(t *testing.T) {
	segment, err := NewWriteSegment(7, 100, 3)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte(`{"title":"one"}`)
	first, err := segment.Append(context.Background(), "one", payload)
	if err != nil || first.DocID != 100 {
		t.Fatalf("first append = %#v, %v", first, err)
	}
	payload[0] = '['
	second, err := segment.Append(context.Background(), "two", nil)
	if err != nil || second.DocID != 101 {
		t.Fatalf("second append = %#v, %v", second, err)
	}
	third, err := segment.Append(context.Background(), "one", []byte("replacement"))
	if err != nil || third.DocID != 102 {
		t.Fatalf("duplicate-key append = %#v, %v", third, err)
	}
	if _, err := segment.Append(context.Background(), "four", nil); !errors.Is(err, ErrSegmentFull) {
		t.Fatalf("full append = %v", err)
	}
	if doc, found := segment.Document(100); !found || string(doc.Payload) != `{"title":"one"}` {
		t.Fatalf("document = %#v, %v", doc, found)
	}
	if metadata := segment.Metadata(); metadata.ID != 7 || metadata.MinDocID != 100 || metadata.MaxDocID != 102 || metadata.DocCount != 3 {
		t.Fatalf("metadata = %#v", metadata)
	}

	dir := t.TempDir()
	immutable, err := segment.Seal(context.Background(), dir, "segments/7/data.seg")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := segment.Append(context.Background(), "sealed", nil); !errors.Is(err, ErrSegmentSealed) {
		t.Fatalf("sealed append = %v", err)
	}
	metadata := immutable.Metadata()
	if !reflect.DeepEqual(metadata.Files, []string{"segments/7/data.seg"}) {
		t.Fatalf("files = %#v", metadata.Files)
	}
	metadata.Files[0] = "changed"
	if immutable.Metadata().Files[0] == "changed" {
		t.Fatal("metadata shares files")
	}

	reopened, err := OpenImmutableSegment(context.Background(), dir, immutable.Metadata())
	if err != nil {
		t.Fatal(err)
	}
	docs := reopened.Documents()
	if len(docs) != 3 || docs[0].DocID != 100 || string(docs[0].Payload) != `{"title":"one"}` {
		t.Fatalf("reopened docs = %#v", docs)
	}
	docs[0].Payload[0] = 'X'
	if doc, _ := reopened.Document(100); doc.Payload[0] == 'X' {
		t.Fatal("document result shares payload")
	}
}

func TestWriteSegmentFailedSealRemainsWritable(t *testing.T) {
	segment, _ := NewWriteSegment(1, 0, 2)
	_, _ = segment.Append(context.Background(), "one", nil)
	dir := t.TempDir()
	existing := filepath.Join(dir, "segment.seg")
	if err := os.WriteFile(existing, []byte("existing"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := segment.Seal(context.Background(), dir, "segment.seg"); !errors.Is(err, os.ErrExist) {
		t.Fatalf("existing seal error = %v", err)
	}
	if _, err := segment.Append(context.Background(), "two", nil); err != nil {
		t.Fatalf("append after failed seal: %v", err)
	}
	if _, err := segment.Seal(context.Background(), dir, "segment-2.seg"); err != nil {
		t.Fatalf("second seal: %v", err)
	}
}

func TestSegmentValidationAndCorruption(t *testing.T) {
	if _, err := NewWriteSegment(1, 0, 0); err == nil {
		t.Fatal("zero capacity succeeded")
	}
	if _, err := NewWriteSegment(1, ^uint64(0), 2); err == nil {
		t.Fatal("overflow range succeeded")
	}
	empty, _ := NewWriteSegment(1, 0, 1)
	if _, err := empty.Seal(context.Background(), t.TempDir(), "empty.seg"); err == nil {
		t.Fatal("empty seal succeeded")
	}
	if _, err := empty.Append(context.Background(), "", nil); err == nil {
		t.Fatal("empty key succeeded")
	}
	if _, err := empty.Append(nil, "key", nil); err == nil {
		t.Fatal("nil context succeeded")
	}

	metadata, docs := sampleSegmentData()
	encoded, err := encodeSegment(metadata, docs)
	if err != nil {
		t.Fatal(err)
	}
	for cut := 0; cut < len(encoded); cut++ {
		if _, _, err := decodeSegment(context.Background(), encoded[:cut]); err == nil {
			t.Fatalf("segment truncation at %d succeeded", cut)
		}
	}
	tests := []func([]byte){
		func(data []byte) { data[0] ^= 1 },
		func(data []byte) { binary.LittleEndian.PutUint16(data[8:10], 2) },
		func(data []byte) { data[60] ^= 1 },
		func(data []byte) { data[56] ^= 1 },
		func(data []byte) { data[segmentHeaderSize] ^= 1 },
		func(data []byte) { data[len(data)-1] ^= 1 },
	}
	for index, mutate := range tests {
		corrupted := append([]byte(nil), encoded...)
		mutate(corrupted)
		if _, _, err := decodeSegment(context.Background(), corrupted); err == nil {
			t.Fatalf("corruption %d succeeded", index)
		}
	}

	badDocID := append([]byte(nil), encoded...)
	binary.LittleEndian.PutUint64(badDocID[segmentHeaderSize:segmentHeaderSize+8], 999)
	record := badDocID[segmentHeaderSize : segmentHeaderSize+segmentRecordHeaderSize]
	binary.LittleEndian.PutUint32(record[16:20], ailego.CRC32C(badDocID[segmentHeaderSize+segmentRecordHeaderSize:segmentHeaderSize+segmentRecordHeaderSize+len(docs[0].PrimaryKey)+len(docs[0].Payload)]))
	binary.LittleEndian.PutUint32(badDocID[56:60], ailego.CRC32C(badDocID[segmentHeaderSize:]))
	binary.LittleEndian.PutUint32(badDocID[60:64], ailego.CRC32C(badDocID[:60]))
	if _, _, err := decodeSegment(context.Background(), badDocID); !errors.Is(err, ErrSegmentCorrupt) {
		t.Fatalf("bad document ID = %v", err)
	}
}

func TestWriteSegmentConcurrentAppend(t *testing.T) {
	segment, _ := NewWriteSegment(1, 500, 400)
	var wait sync.WaitGroup
	ids := make(chan uint64, 400)
	errs := make(chan error, 8)
	for worker := range 8 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for record := range 50 {
				doc, err := segment.Append(context.Background(), string(rune('a'+worker))+string(rune(0x100+record)), []byte{byte(record)})
				if err != nil {
					errs <- err
					return
				}
				ids <- doc.DocID
			}
		}()
	}
	wait.Wait()
	close(ids)
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	all := make([]uint64, 0, 400)
	for id := range ids {
		all = append(all, id)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	for index, id := range all {
		if id != 500+uint64(index) {
			t.Fatalf("ID[%d] = %d", index, id)
		}
	}
}

func FuzzDecodeSegment(f *testing.F) {
	metadata, docs := sampleSegmentData()
	encoded, err := encodeSegment(metadata, docs)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(encoded)
	f.Add(encoded[:segmentHeaderSize])
	f.Add([]byte("not a segment"))
	f.Fuzz(func(t *testing.T, data []byte) {
		metadata, docs, err := decodeSegment(context.Background(), data)
		if err == nil {
			if metadata.DocCount != uint64(len(docs)) {
				t.Fatalf("metadata count %d, docs %d", metadata.DocCount, len(docs))
			}
		}
	})
}

func sampleSegmentData() (SegmentMetadata, []StoredDocument) {
	docs := []StoredDocument{
		{DocID: 10, PrimaryKey: "alpha", Payload: []byte("A")},
		{DocID: 11, PrimaryKey: "beta", Payload: []byte("BB")},
	}
	return segmentMetadata(3, docs, []string{"segments/3/data.seg"}), docs
}

func TestSegmentPayloadCRC(t *testing.T) {
	metadata, docs := sampleSegmentData()
	encoded, err := encodeSegment(metadata, docs)
	if err != nil {
		t.Fatal(err)
	}
	if actual, expected := binary.LittleEndian.Uint32(encoded[56:60]), ailego.CRC32C(encoded[segmentHeaderSize:]); actual != expected {
		t.Fatalf("payload CRC = %08x, want %08x", actual, expected)
	}
	if bytes.Equal(encoded[:8], make([]byte, 8)) {
		t.Fatal("segment magic is empty")
	}
}
