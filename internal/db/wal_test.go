// Copyright 2026-present the xvec project
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
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestWALCreateAppendReplayAndReopen(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{SyncEvery: 2})
	require.NoError(t, err)
	require.False(t, wal.HasRecords(),
		"new WAL has records")
	{
		recovery := wal.Recovery()
		require.Equal(t, WALRecovery{ValidBytes: walFileHeaderSize}, recovery)
	}

	firstPayload := []byte("insert:book-1")
	firstLSN, err := wal.Append(context.Background(), firstPayload)
	require.NoError(t, err)
	require.True(t, firstLSN == 1)

	firstPayload[0] = 'X'
	secondLSN, err := wal.Append(context.Background(), []byte("update:book-1"))
	require.NoError(t, err)
	require.True(t, secondLSN == 2)

	reader, err := wal.NewReader()
	require.NoError(t, err)

	thirdLSN, err := wal.Append(context.Background(), []byte("delete:book-1"))
	require.NoError(t, err)
	require.True(t, thirdLSN == 3)
	{
		err := wal.Sync(context.Background())
		require.NoError(t, err)
	}
	require.True(t, wal.HasRecords(),
		"appended WAL reports no records")

	snapshot, err := readAllWAL(t, reader)
	require.NoError(t, err)
	{
		err := reader.Close()
		require.NoError(t, err)
	}
	require.Len(t, snapshot, 2)
	require.True(t, string(snapshot[0].Payload) == "insert:book-1")

	snapshot[0].Payload[0] = 'X'

	var replayed []WALRecord
	{
		err := wal.Replay(context.Background(), func(record WALRecord) error {
			replayed = append(replayed, record)
			return nil
		})
		require.NoError(t, err)
	}
	{
		got := recordPayloads(replayed)
		require.Equal(t, []string{"insert:book-1", "update:book-1", "delete:book-1"}, got)
	}
	{
		err := wal.Close()
		require.NoError(t, err)
	}
	{
		err := wal.Close()
		require.NoError(t, err)
	}

	reopened, err := OpenWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)

	defer reopened.Close()
	recovery := reopened.Recovery()
	require.True(t, recovery.Records == 3)
	require.True(t, recovery.LastLSN == 3)
	require.True(t, recovery.TruncatedBytes == 0)
	{
		info, err := os.Stat(name)
		require.NoError(t, err)
		require.Equal(t, info.Size(), recovery.ValidBytes)
	}
}

func TestWALReplayUsesStableSnapshot(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)

	defer wal.Close()
	{
		_, err := wal.Append(context.Background(), []byte("one"))
		require.NoError(t, err)
	}

	var replayed []string
	{
		err := wal.Replay(context.Background(), func(record WALRecord) error {
			replayed = append(replayed, string(record.Payload))
			if record.LSN == 1 {
				_, err := wal.Append(context.Background(), []byte("two"))
				return err
			}
			return nil
		})
		require.NoError(t, err)
	}
	require.Equal(t, []string{"one"}, replayed)
	{
		err := wal.Replay(context.Background(), func(record WALRecord) error {
			require.False(t, record.LSN == 2 && string(record.Payload) != "two")

			return nil
		})
		require.NoError(t, err)
	}
}

func TestWALTruncatesEveryPartialTailBoundary(t *testing.T) {
	base := makeWALBytes(t, []byte("one"), []byte("two"))
	partial := encodeWALRecord(3, []byte("three"))
	for cut := 1; cut < len(partial); cut++ {
		t.Run(fmt.Sprintf("bytes_%d", cut), func(t *testing.T) {
			name := filepath.Join(t.TempDir(), "data.wal")
			contents := append(append([]byte(nil), base...), partial[:cut]...)
			{
				err := os.WriteFile(name, contents, 0o600)
				require.NoError(t, err)
			}

			wal, err := OpenWAL(context.Background(), name, WALOptions{})
			require.NoError(t, err)

			recovery := wal.Recovery()
			require.True(t, recovery.Records == 2)
			require.True(t, recovery.LastLSN == 2)
			require.Equal(t, int64(len(base)), recovery.ValidBytes)
			require.Equal(t, int64(cut), recovery.TruncatedBytes)
			{
				info, err := os.Stat(name)
				require.NoError(t, err)
				require.Equal(t, int64(len(base)), info.Size())
			}

			lsn, err := wal.Append(context.Background(), []byte("replacement"))
			require.NoError(t, err)
			require.True(t, lsn == 3)
			{
				err := wal.Close()
				require.NoError(t, err)
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
			{
				err := os.WriteFile(name, contents, 0o600)
				require.NoError(t, err)
			}

			before := fileSize(t, name)
			{
				_, err := OpenWAL(context.Background(), name, WALOptions{})
				require.ErrorIs(t, err, testCase.expected)
			}
			{
				after := fileSize(t, name)
				require.Equal(t, before, after)
			}
		})
	}
}

func TestWALRejectsTruncatedFileHeader(t *testing.T) {
	header := encodeWALFileHeader()
	for cut := 0; cut < len(header); cut++ {
		name := filepath.Join(t.TempDir(), fmt.Sprintf("data-%d.wal", cut))
		{
			err := os.WriteFile(name, header[:cut], 0o600)
			require.NoError(t, err)
		}
		{
			_, err := OpenWAL(context.Background(), name, WALOptions{})
			require.ErrorIs(t, err, ErrWALCorrupt)
		}
		{
			size := fileSize(t, name)
			require.Equal(t, int64(cut), size)
		}
	}
}

func TestWALConcurrentAppend(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)

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
		require.NoError(t, err)
	}
	allLSNs := make([]uint64, 0, goroutines*recordsPerGoroutine)
	for lsn := range lsns {
		allLSNs = append(allLSNs, lsn)
	}
	sort.Slice(allLSNs, func(left, right int) bool { return allLSNs[left] < allLSNs[right] })
	for index, lsn := range allLSNs {
		require.Equal(t, uint64(index+1), lsn)
	}
	{
		err := wal.Sync(context.Background())
		require.NoError(t, err)
	}

	reader, err := wal.NewReader()
	require.NoError(t, err)

	records, err := readAllWAL(t, reader)
	require.NoError(t, err)
	require.NoError(t, reader.Close())
	require.Len(t, records, goroutines*recordsPerGoroutine)
	for index, record := range records {
		require.Equal(t, uint64(index+1), record.LSN)
	}
}

func TestWALWriterLockHonorsContext(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	first, err := CreateWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)

	defer first.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	{
		_, err := OpenWAL(ctx, name, WALOptions{})
		require.ErrorIs(t, err, context.DeadlineExceeded)
	}
	{
		err := first.Close()
		require.NoError(t, err)
	}

	second, err := OpenWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)
	{
		err := second.Close()
		require.NoError(t, err)
	}
}

func TestWALArgumentAndClosedValidation(t *testing.T) {
	{
		_, err := CreateWAL(nil, "x", WALOptions{})
		require.Error(t, err,
			"nil create context succeeded")
	}
	{
		_, err := OpenWAL(nil, "x", WALOptions{})
		require.Error(t, err,
			"nil open context succeeded")
	}
	{
		_, err := CreateWAL(context.Background(), "", WALOptions{})
		require.Error(t, err,
			"empty create path succeeded")
	}
	{
		_, err := OpenWAL(context.Background(), filepath.Join(t.TempDir(), "missing.wal"), WALOptions{})
		require.ErrorIs(t, err, ErrWALNotFound)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	name := filepath.Join(t.TempDir(), "canceled.wal")
	{
		_, err := CreateWAL(canceled, name, WALOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := os.Stat(name)
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	name = filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)
	{
		_, err := CreateWAL(context.Background(), name, WALOptions{})
		require.ErrorIs(t, err, ErrWALExists)
	}
	{
		_, err := wal.Append(context.Background(), nil)
		require.Error(t, err,
			"empty record succeeded")
	}
	{
		_, err := wal.Append(context.Background(), make([]byte, MaxWALRecordSize+1))
		require.ErrorIs(t, err, ErrWALRecordTooLarge)
	}
	{
		_, err := wal.Append(nil, []byte("x"))
		require.Error(t, err,
			"nil append context succeeded")
	}
	{
		err := wal.Sync(nil)
		require.Error(t, err,
			"nil sync context succeeded")
	}
	{
		err := wal.Replay(context.Background(), nil)
		require.Error(t, err,
			"nil replay function succeeded")
	}

	callbackError := errors.New("stop")
	{
		_, err := wal.Append(context.Background(), []byte("one"))
		require.NoError(t, err)
	}
	{
		err := wal.Replay(context.Background(), func(WALRecord) error { return callbackError })
		require.ErrorIs(t, err, callbackError)
	}

	reader, err := wal.NewReader()
	require.NoError(t, err)
	{
		err := reader.Close()
		require.NoError(t, err)
	}
	{
		_, err := reader.Next(context.Background())
		require.ErrorIs(t, err, ErrWALClosed)
	}
	{
		err := wal.Close()
		require.NoError(t, err)
	}
	{
		_, err := wal.Append(context.Background(), []byte("x"))
		require.ErrorIs(t, err, ErrWALClosed)
	}
	{
		err := wal.Sync(context.Background())
		require.ErrorIs(t, err, ErrWALClosed)
	}
	{
		_, err := wal.NewReader()
		require.ErrorIs(t, err, ErrWALClosed)
	}

	var nilWAL *WAL
	{
		_, err := nilWAL.Append(context.Background(), []byte("x"))
		require.Error(t, err,
			"nil WAL append succeeded")
	}
	{
		err := nilWAL.Sync(context.Background())
		require.Error(t, err,
			"nil WAL sync succeeded")
	}

	var nilReader *WALReader
	{
		_, err := nilReader.Next(context.Background())
		require.Error(t, err,
			"nil reader succeeded")
	}
}

func TestWALSyncFailurePoisonsHandle(t *testing.T) {
	wal, err := CreateWAL(context.Background(), filepath.Join(t.TempDir(), "data.wal"), WALOptions{})
	require.NoError(t, err)
	defer wal.Close()
	_, err = wal.Append(context.Background(), []byte("record"))
	require.NoError(t, err)
	syncError := errors.New("injected sync failure")
	wal.syncFile = func() error { return syncError }

	err = wal.Sync(context.Background())
	require.ErrorIs(t, err, syncError)
	require.ErrorIs(t, err, ErrWALPoisoned)
	_, err = wal.Append(context.Background(), []byte("next"))
	require.ErrorIs(t, err, ErrWALPoisoned)
}

func TestWALReaderDetectsPostOpenCorruption(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	wal, err := CreateWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)

	defer wal.Close()
	{
		_, err := wal.Append(context.Background(), []byte("one"))
		require.NoError(t, err)
	}

	reader, err := wal.NewReader()
	require.NoError(t, err)

	defer reader.Close()
	file, err := os.OpenFile(name, os.O_RDWR, 0)
	require.NoError(t, err)

	byteAtPayload := []byte{0}
	{
		_, err := file.ReadAt(byteAtPayload, walFileHeaderSize+walRecordHeaderSize)
		require.NoError(t, err)
	}

	byteAtPayload[0] ^= 1
	{
		_, err := file.WriteAt(byteAtPayload, walFileHeaderSize+walRecordHeaderSize)
		require.NoError(t, err)
	}
	{
		err := file.Close()
		require.NoError(t, err)
	}
	{
		_, err := reader.Next(context.Background())
		require.ErrorIs(t, err, ErrWALCorrupt)
	}
}

func TestOpenWALReadOnlyPreservesPartialTail(t *testing.T) {
	name := filepath.Join(t.TempDir(), "data.wal")
	complete := makeWALBytes(t, []byte("one"))
	partial := encodeWALRecord(2, []byte("two"))[:walRecordHeaderSize+1]
	contents := append(append([]byte(nil), complete...), partial...)
	{
		err := os.WriteFile(name, contents, 0o600)
		require.NoError(t, err)
	}

	readOnly, err := OpenWALReadOnly(context.Background(), name)
	require.NoError(t, err)
	{
		recovery := readOnly.Recovery()
		require.True(t, recovery.Records == 1)
		require.Equal(t, int64(len(partial)), recovery.TruncatedBytes)
	}
	{
		_, err := readOnly.Append(context.Background(), []byte("x"))
		require.ErrorIs(t, err, ErrWALReadOnly)
	}
	{
		err := readOnly.Sync(context.Background())
		require.ErrorIs(t, err, ErrWALReadOnly)
	}
	{
		err := readOnly.Close()
		require.NoError(t, err)
	}
	{
		got := fileSize(t, name)
		require.Equal(t, int64(len(contents)), got)
	}

	writable, err := OpenWAL(context.Background(), name, WALOptions{})
	require.NoError(t, err)
	{
		err := writable.Close()
		require.NoError(t, err)
	}
	{
		got := fileSize(t, name)
		require.Equal(t, int64(len(complete)), got)
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
			require.True(t, recovery.ValidBytes >= walFileHeaderSize)
			require.True(t, recovery.ValidBytes <= int64(len(data)))
			require.Equal(t, int64(len(data)), recovery.ValidBytes+recovery.TruncatedBytes)
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
	require.NoError(t, err)

	return info.Size()
}

func TestWALRecordHeaderCRCUsesCastagnoli(t *testing.T) {
	header := encodeWALRecord(1, []byte("payload"))[:walRecordHeaderSize]
	{
		actual, expected := binary.LittleEndian.Uint32(header[24:28]), ailego.CRC32C(header[:24])
		require.Equal(t, expected, actual)
	}
}
