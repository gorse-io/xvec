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
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestWALCreateAppendReplayAndReopen(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{SyncEvery: 2})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if wal.HasRecords() {
		t.Fatal("new WAL has records")
	}
	if recovery := wal.Recovery(); recovery != (WALRecovery{ValidBytes: walFileHeaderSize}) {
		t.Fatalf("create recovery = %#v", recovery)
	}

	firstPayload := []byte("insert:book-1")
	firstLSN, err := wal.Append(context.Background(), firstPayload)
	if err != nil || firstLSN != 1 {
		t.Fatalf("first append = %d, %v", firstLSN, err)
	}
	firstPayload[0] = 'X'
	secondLSN, err := wal.Append(context.Background(), []byte("update:book-1"))
	if err != nil || secondLSN != 2 {
		t.Fatalf("second append = %d, %v", secondLSN, err)
	}
	reader, err := wal.NewReader()
	if err != nil {
		t.Fatal(err)
	}
	thirdLSN, err := wal.Append(context.Background(), []byte("delete:book-1"))
	if err != nil || thirdLSN != 3 {
		t.Fatalf("third append = %d, %v", thirdLSN, err)
	}
	if err := wal.Sync(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !wal.HasRecords() {
		t.Fatal("appended WAL reports no records")
	}

	snapshot, err := readAllWAL(t, reader)
	if err != nil {
		t.Fatal(err)
	}
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	if len(snapshot) != 2 {
		t.Fatalf("snapshot record count = %d, want 2", len(snapshot))
	}
	if string(snapshot[0].Payload) != "insert:book-1" {
		t.Fatalf("first payload = %q", snapshot[0].Payload)
	}
	snapshot[0].Payload[0] = 'X'

	var replayed []WALRecord
	if err := wal.Replay(context.Background(), func(record WALRecord) error {
		replayed = append(replayed, record)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if got := recordPayloads(replayed); !reflect.DeepEqual(got, []string{"insert:book-1", "update:book-1", "delete:book-1"}) {
		t.Fatalf("replayed = %#v", got)
	}
	if err := wal.Close(); err != nil {
		t.Fatal(err)
	}
	if err := wal.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}

	reopened, err := OpenWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer reopened.Close()
	recovery := reopened.Recovery()
	if recovery.Records != 3 || recovery.LastLSN != 3 || recovery.TruncatedBytes != 0 {
		t.Fatalf("recovery = %#v", recovery)
	}
	if info, err := os.Stat(name); err != nil || recovery.ValidBytes != info.Size() {
		t.Fatalf("valid bytes = %d, stat = %#v, %v", recovery.ValidBytes, info, err)
	}
}

func TestWALReplayUsesStableSnapshot(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	if _, err := wal.Append(context.Background(), []byte("one")); err != nil {
		t.Fatal(err)
	}
	var replayed []string
	if err := wal.Replay(context.Background(), func(record WALRecord) error {
		replayed = append(replayed, string(record.Payload))
		if record.LSN == 1 {
			_, err := wal.Append(context.Background(), []byte("two"))
			return err
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(replayed, []string{"one"}) {
		t.Fatalf("snapshot replay = %#v", replayed)
	}
	if err := wal.Replay(context.Background(), func(record WALRecord) error {
		if record.LSN == 2 && string(record.Payload) != "two" {
			t.Fatalf("second payload = %q", record.Payload)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestWALTruncatesEveryPartialTailBoundary(t *testing.T) {
	base := makeWALBytes(t, []byte("one"), []byte("two"))
	partial := encodeWALRecord(3, []byte("three"))
	for cut := 1; cut < len(partial); cut++ {
		t.Run(fmt.Sprintf("bytes_%d", cut), func(t *testing.T) {
			name := filepath.Join(t.TempDir(), "data.wal")
			contents := append(append([]byte(nil), base...), partial[:cut]...)
			if err := os.WriteFile(name, contents, 0o600); err != nil {
				t.Fatal(err)
			}
			wal, err := OpenWAL(context.Background(), name, WALOptions{})
			if err != nil {
				t.Fatalf("open partial tail: %v", err)
			}
			recovery := wal.Recovery()
			if recovery.Records != 2 || recovery.LastLSN != 2 || recovery.ValidBytes != int64(len(base)) || recovery.TruncatedBytes != int64(cut) {
				t.Fatalf("recovery = %#v", recovery)
			}
			if info, err := os.Stat(name); err != nil || info.Size() != int64(len(base)) {
				t.Fatalf("repaired size = %#v, %v", info, err)
			}
			lsn, err := wal.Append(context.Background(), []byte("replacement"))
			if err != nil || lsn != 3 {
				t.Fatalf("append after repair = %d, %v", lsn, err)
			}
			if err := wal.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestWALRejectsCorruptionWithoutTruncating(t *testing.T) {
	base := makeWALBytes(t, []byte("one"), []byte("two"))
	firstHeader := walFileHeaderSize
	firstPayload := firstHeader + walRecordHeaderSize
	secondHeader := firstPayload + len("one")
	tests := []struct {
		name     string
		mutate   func([]byte) []byte
		expected error
	}{
		{name: "file magic", mutate: func(data []byte) []byte { data[0] ^= 1; return data }, expected: ErrWALCorrupt},
		{name: "file version", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint16(data[8:10], 2); return data }, expected: ErrUnsupportedFormatVersion},
		{name: "file header size", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint16(data[10:12], 31); return data }, expected: ErrWALCorrupt},
		{name: "file maximum", mutate: func(data []byte) []byte { binary.LittleEndian.PutUint32(data[12:16], 1); return data }, expected: ErrWALCorrupt},
		{name: "file reserved", mutate: func(data []byte) []byte { data[16] = 1; return data }, expected: ErrWALCorrupt},
		{name: "file checksum", mutate: func(data []byte) []byte { data[24] ^= 1; return data }, expected: ErrWALCorrupt},
		{name: "record magic", mutate: func(data []byte) []byte { data[firstHeader] ^= 1; return data }, expected: ErrWALCorrupt},
		{name: "record version", mutate: func(data []byte) []byte {
			binary.LittleEndian.PutUint16(data[firstHeader+4:firstHeader+6], 2)
			return data
		}, expected: ErrWALCorrupt},
		{name: "record header size", mutate: func(data []byte) []byte {
			binary.LittleEndian.PutUint16(data[firstHeader+6:firstHeader+8], 31)
			return data
		}, expected: ErrWALCorrupt},
		{name: "record LSN", mutate: func(data []byte) []byte {
			binary.LittleEndian.PutUint64(data[firstHeader+8:firstHeader+16], 2)
			return data
		}, expected: ErrWALCorrupt},
		{name: "zero record length", mutate: func(data []byte) []byte {
			binary.LittleEndian.PutUint32(data[firstHeader+16:firstHeader+20], 0)
			return data
		}, expected: ErrWALCorrupt},
		{name: "large record length", mutate: func(data []byte) []byte {
			binary.LittleEndian.PutUint32(data[firstHeader+16:firstHeader+20], MaxWALRecordSize+1)
			return data
		}, expected: ErrWALCorrupt},
		{name: "record header checksum", mutate: func(data []byte) []byte { data[firstHeader+24] ^= 1; return data }, expected: ErrWALCorrupt},
		{name: "record reserved", mutate: func(data []byte) []byte { data[firstHeader+28] = 1; return data }, expected: ErrWALCorrupt},
		{name: "first payload", mutate: func(data []byte) []byte { data[firstPayload] ^= 1; return data }, expected: ErrWALCorrupt},
		{name: "second payload", mutate: func(data []byte) []byte { data[len(data)-1] ^= 1; return data }, expected: ErrWALCorrupt},
		{name: "second LSN", mutate: func(data []byte) []byte {
			binary.LittleEndian.PutUint64(data[secondHeader+8:secondHeader+16], 9)
			return data
		}, expected: ErrWALCorrupt},
		{name: "complete junk record", mutate: func(data []byte) []byte { return append(data, make([]byte, walRecordHeaderSize)...) }, expected: ErrWALCorrupt},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			name := filepath.Join(t.TempDir(), "data.wal")
			contents := testCase.mutate(append([]byte(nil), base...))
			if err := os.WriteFile(name, contents, 0o600); err != nil {
				t.Fatal(err)
			}
			before := fileSize(t, name)
			if _, err := OpenWAL(context.Background(), name, WALOptions{}); !errors.Is(err, testCase.expected) {
				t.Fatalf("open error = %v, want %v", err, testCase.expected)
			}
			if after := fileSize(t, name); after != before {
				t.Fatalf("corrupt WAL size changed from %d to %d", before, after)
			}
		})
	}
}

func TestWALRejectsTruncatedFileHeader(t *testing.T) {
	header := encodeWALFileHeader()
	for cut := 0; cut < len(header); cut++ {
		name := filepath.Join(t.TempDir(), fmt.Sprintf("data-%d.wal", cut))
		if err := os.WriteFile(name, header[:cut], 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenWAL(context.Background(), name, WALOptions{}); !errors.Is(err, ErrWALCorrupt) {
			t.Fatalf("header cut %d error = %v", cut, err)
		}
		if size := fileSize(t, name); size != int64(cut) {
			t.Fatalf("header cut %d changed size to %d", cut, size)
		}
	}
}

func TestWALConcurrentAppend(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()

	const goroutines = 8
	const recordsPerGoroutine = 40
	lsns := make(chan uint64, goroutines*recordsPerGoroutine)
	errorsByAppend := make(chan error, goroutines)
	var wait sync.WaitGroup
	for worker := range goroutines {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for record := range recordsPerGoroutine {
				payload := []byte(fmt.Sprintf("worker-%d-record-%d", worker, record))
				lsn, err := wal.Append(context.Background(), payload)
				if err != nil {
					errorsByAppend <- err
					return
				}
				lsns <- lsn
			}
		}()
	}
	wait.Wait()
	close(lsns)
	close(errorsByAppend)
	for err := range errorsByAppend {
		t.Fatal(err)
	}
	allLSNs := make([]uint64, 0, goroutines*recordsPerGoroutine)
	for lsn := range lsns {
		allLSNs = append(allLSNs, lsn)
	}
	sort.Slice(allLSNs, func(left, right int) bool { return allLSNs[left] < allLSNs[right] })
	for index, lsn := range allLSNs {
		if lsn != uint64(index+1) {
			t.Fatalf("LSN[%d] = %d", index, lsn)
		}
	}
	if err := wal.Sync(context.Background()); err != nil {
		t.Fatal(err)
	}
	reader, err := wal.NewReader()
	if err != nil {
		t.Fatal(err)
	}
	records, err := readAllWAL(t, reader)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != goroutines*recordsPerGoroutine {
		t.Fatalf("record count = %d", len(records))
	}
	for index, record := range records {
		if record.LSN != uint64(index+1) {
			t.Fatalf("record LSN[%d] = %d", index, record.LSN)
		}
	}
}

func TestWALWriterLockHonorsContext(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	first, err := CreateWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if _, err := OpenWAL(ctx, name, WALOptions{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second writer error = %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := OpenWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatalf("open after unlock: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestWALArgumentAndClosedValidation(t *testing.T) {
	if _, err := CreateWAL(nil, "x", WALOptions{}); err == nil {
		t.Fatal("nil create context succeeded")
	}
	if _, err := OpenWAL(nil, "x", WALOptions{}); err == nil {
		t.Fatal("nil open context succeeded")
	}
	if _, err := CreateWAL(context.Background(), "", WALOptions{}); err == nil {
		t.Fatal("empty create path succeeded")
	}
	if _, err := OpenWAL(context.Background(), filepath.Join(t.TempDir(), "missing.wal"), WALOptions{}); !errors.Is(err, ErrWALNotFound) {
		t.Fatalf("missing WAL error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	name := filepath.Join(t.TempDir(), "canceled.wal")
	if _, err := CreateWAL(canceled, name, WALOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled create error = %v", err)
	}
	if _, err := os.Stat(name); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("canceled create wrote WAL: %v", err)
	}

	name = filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CreateWAL(context.Background(), name, WALOptions{}); !errors.Is(err, ErrWALExists) {
		t.Fatalf("duplicate create error = %v", err)
	}
	if _, err := wal.Append(context.Background(), nil); err == nil {
		t.Fatal("empty record succeeded")
	}
	if _, err := wal.Append(context.Background(), make([]byte, MaxWALRecordSize+1)); !errors.Is(err, ErrWALRecordTooLarge) {
		t.Fatalf("large record error = %v", err)
	}
	if _, err := wal.Append(nil, []byte("x")); err == nil {
		t.Fatal("nil append context succeeded")
	}
	if err := wal.Sync(nil); err == nil {
		t.Fatal("nil sync context succeeded")
	}
	if err := wal.Replay(context.Background(), nil); err == nil {
		t.Fatal("nil replay function succeeded")
	}
	callbackError := errors.New("stop")
	if _, err := wal.Append(context.Background(), []byte("one")); err != nil {
		t.Fatal(err)
	}
	if err := wal.Replay(context.Background(), func(WALRecord) error { return callbackError }); !errors.Is(err, callbackError) {
		t.Fatalf("callback error = %v", err)
	}
	reader, err := wal.NewReader()
	if err != nil {
		t.Fatal(err)
	}
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := reader.Next(context.Background()); !errors.Is(err, ErrWALClosed) {
		t.Fatalf("closed reader error = %v", err)
	}
	if err := wal.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := wal.Append(context.Background(), []byte("x")); !errors.Is(err, ErrWALClosed) {
		t.Fatalf("closed append error = %v", err)
	}
	if err := wal.Sync(context.Background()); !errors.Is(err, ErrWALClosed) {
		t.Fatalf("closed sync error = %v", err)
	}
	if _, err := wal.NewReader(); !errors.Is(err, ErrWALClosed) {
		t.Fatalf("closed NewReader error = %v", err)
	}

	var nilWAL *WAL
	if _, err := nilWAL.Append(context.Background(), []byte("x")); err == nil {
		t.Fatal("nil WAL append succeeded")
	}
	if err := nilWAL.Sync(context.Background()); err == nil {
		t.Fatal("nil WAL sync succeeded")
	}
	var nilReader *WALReader
	if _, err := nilReader.Next(context.Background()); err == nil {
		t.Fatal("nil reader succeeded")
	}
}

func TestWALReaderDetectsPostOpenCorruption(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	if _, err := wal.Append(context.Background(), []byte("one")); err != nil {
		t.Fatal(err)
	}
	reader, err := wal.NewReader()
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	file, err := os.OpenFile(name, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	byteAtPayload := []byte{0}
	if _, err := file.ReadAt(byteAtPayload, walFileHeaderSize+walRecordHeaderSize); err != nil {
		t.Fatal(err)
	}
	byteAtPayload[0] ^= 1
	if _, err := file.WriteAt(byteAtPayload, walFileHeaderSize+walRecordHeaderSize); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := reader.Next(context.Background()); !errors.Is(err, ErrWALCorrupt) {
		t.Fatalf("reader corruption error = %v", err)
	}
}

func TestOpenWALReadOnlyPreservesPartialTail(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	complete := makeWALBytes(t, []byte("one"))
	partial := encodeWALRecord(2, []byte("two"))[:walRecordHeaderSize+1]
	contents := append(append([]byte(nil), complete...), partial...)
	if err := os.WriteFile(name, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	readOnly, err := OpenWALReadOnly(context.Background(), name)
	if err != nil {
		t.Fatal(err)
	}
	if recovery := readOnly.Recovery(); recovery.Records != 1 || recovery.TruncatedBytes != int64(len(partial)) {
		t.Fatalf("recovery = %#v", recovery)
	}
	if _, err := readOnly.Append(context.Background(), []byte("x")); !errors.Is(err, ErrWALReadOnly) {
		t.Fatalf("append error = %v", err)
	}
	if err := readOnly.Sync(context.Background()); !errors.Is(err, ErrWALReadOnly) {
		t.Fatalf("sync error = %v", err)
	}
	if err := readOnly.Close(); err != nil {
		t.Fatal(err)
	}
	if got := fileSize(t, name); got != int64(len(contents)) {
		t.Fatalf("read-only open changed size to %d, want %d", got, len(contents))
	}
	writable, err := OpenWAL(context.Background(), name, WALOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := writable.Close(); err != nil {
		t.Fatal(err)
	}
	if got := fileSize(t, name); got != int64(len(complete)) {
		t.Fatalf("writable repair size = %d, want %d", got, len(complete))
	}
}

func FuzzScanWAL(f *testing.F) {
	valid := append(encodeWALFileHeader(), encodeWALRecord(1, []byte("one"))...)
	f.Add(valid)
	f.Add(encodeWALFileHeader())
	f.Add(valid[:len(valid)-1])
	f.Add([]byte("not a WAL"))
	f.Fuzz(func(t *testing.T, data []byte) {
		recovery, err := scanWAL(context.Background(), bytes.NewReader(data), int64(len(data)))
		if err == nil {
			if recovery.ValidBytes < walFileHeaderSize || recovery.ValidBytes > int64(len(data)) {
				t.Fatalf("invalid recovery = %#v for %d bytes", recovery, len(data))
			}
			if recovery.ValidBytes+recovery.TruncatedBytes != int64(len(data)) {
				t.Fatalf("recovery does not partition input: %#v, size %d", recovery, len(data))
			}
		}
	})
}

func makeWALBytes(t *testing.T, payloads ...[]byte) []byte {
	t.Helper()
	contents := encodeWALFileHeader()
	for index, payload := range payloads {
		contents = append(contents, encodeWALRecord(uint64(index+1), payload)...)
	}
	return contents
}

func readAllWAL(t *testing.T, reader *WALReader) ([]WALRecord, error) {
	t.Helper()
	var records []WALRecord
	for {
		record, err := reader.Next(context.Background())
		if errors.Is(err, io.EOF) {
			return records, nil
		}
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}
}

func recordPayloads(records []WALRecord) []string {
	payloads := make([]string, len(records))
	for index := range records {
		payloads[index] = string(records[index].Payload)
	}
	return payloads
}

func fileSize(t *testing.T, name string) int64 {
	t.Helper()
	info, err := os.Stat(name)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

func TestWALRecordHeaderCRCUsesCastagnoli(t *testing.T) {
	header := encodeWALRecord(1, []byte("payload"))[:walRecordHeaderSize]
	if actual, expected := binary.LittleEndian.Uint32(header[24:28]), ailego.CRC32C(header[:24]); actual != expected {
		t.Fatalf("record header CRC = %08x, want %08x", actual, expected)
	}
}
