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
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestWriteSegmentSealAndOpen(t *testing.T) {
	segment, err := NewWriteSegment(7, 100, 3)
	require.NoError(t, err)

	payload := []byte(`{"title":"one"}`)
	first, err := segment.Append(context.Background(), "one", payload)
	require.NoError(t, err)
	require.True(t, first.DocID == 100)

	payload[0] = '['
	second, err := segment.Append(context.Background(), "two", nil)
	require.NoError(t, err)
	require.True(t, second.DocID == 101)

	third, err := segment.Append(context.Background(), "one", []byte("replacement"))
	require.NoError(t, err)
	require.True(t, third.DocID == 102)
	{
		_, err := segment.Append(context.Background(), "four", nil)
		require.ErrorIs(t, err, ErrSegmentFull)
	}
	{
		doc, found := segment.Document(100)
		require.True(t, found)
		require.True(t, string(doc.Payload) == `{"title":"one"}`)
	}
	{
		metadata := segment.Metadata()
		require.True(t, metadata.ID == 7)
		require.True(t, metadata.MinDocID == 100)
		require.True(t, metadata.MaxDocID == 102)
		require.True(t, metadata.DocCount == 3)
	}

	dir := t.TempDir()
	immutable, err := segment.Seal(context.Background(), dir, "segments/7/data.seg")
	require.NoError(t, err)
	{
		_, err := segment.Append(context.Background(), "sealed", nil)
		require.ErrorIs(t, err, ErrSegmentSealed)
	}

	metadata := immutable.Metadata()
	require.Equal(t, []string{"segments/7/data.seg"}, metadata.Files)

	metadata.Files[0] = "changed"
	require.False(t, immutable.Metadata().Files[0] == "changed",
		"metadata shares files")

	reopened, err := OpenImmutableSegment(context.Background(), dir, immutable.Metadata())
	require.NoError(t, err)

	docs := reopened.Documents()
	require.Len(t, docs, 3)
	require.True(t, docs[0].DocID == 100)
	require.True(t, string(docs[0].Payload) == `{"title":"one"}`)

	docs[0].Payload[0] = 'X'
	{
		doc, _ := reopened.Document(100)
		require.False(t, doc.Payload[0] == 'X',
			"document result shares payload")
	}
}

func TestWriteSegmentFailedSealRemainsWritable(t *testing.T) {
	segment, _ := NewWriteSegment(1, 0, 2)
	_, _ = segment.Append(context.Background(), "one", nil)
	dir := t.TempDir()
	existing := filepath.Join(dir, "segment.seg")
	{
		err := os.WriteFile(existing, []byte("existing"), 0o600)
		require.NoError(t, err)
	}
	{
		_, err := segment.Seal(context.Background(), dir, "segment.seg")
		require.ErrorIs(t, err, os.ErrExist)
	}
	{
		_, err := segment.Append(context.Background(), "two", nil)
		require.NoError(t, err)
	}
	{
		_, err := segment.Seal(context.Background(), dir, "segment-2.seg")
		require.NoError(t, err)
	}
}

func TestSegmentValidationAndCorruption(t *testing.T) {
	{
		_, err := NewWriteSegment(1, 0, 0)
		require.Error(t, err,
			"zero capacity succeeded")
	}
	{
		_, err := NewWriteSegment(1, ^uint64(0), 2)
		require.Error(t, err,
			"overflow range succeeded")
	}

	empty, _ := NewWriteSegment(1, 0, 1)
	{
		_, err := empty.Seal(context.Background(), t.TempDir(), "empty.seg")
		require.Error(t, err,
			"empty seal succeeded")
	}
	{
		_, err := empty.Append(context.Background(), "", nil)
		require.Error(t, err,
			"empty key succeeded")
	}
	{
		_, err := empty.Append(nil, "key", nil)
		require.Error(t, err,
			"nil context succeeded")
	}

	metadata, docs := sampleSegmentData()
	encoded, err := encodeSegment(metadata, docs)
	require.NoError(t, err)

	for cut := 0; cut < len(encoded); cut++ {
		{
			_, _, err := decodeSegment(context.Background(), encoded[:cut])
			require.Error(t, err)
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
	for _, mutate := range tests {
		corrupted := append([]byte(nil), encoded...)
		mutate(corrupted)
		{
			_, _, err := decodeSegment(context.Background(), corrupted)
			require.Error(t, err)
		}
	}

	badDocID := append([]byte(nil), encoded...)
	binary.LittleEndian.PutUint64(badDocID[segmentHeaderSize:segmentHeaderSize+8], 999)
	record := badDocID[segmentHeaderSize : segmentHeaderSize+segmentRecordHeaderSize]
	binary.LittleEndian.PutUint32(record[16:20], ailego.CRC32C(badDocID[segmentHeaderSize+segmentRecordHeaderSize:segmentHeaderSize+segmentRecordHeaderSize+len(docs[0].PrimaryKey)+len(docs[0].Payload)]))
	binary.LittleEndian.PutUint32(badDocID[56:60], ailego.CRC32C(badDocID[segmentHeaderSize:]))
	binary.LittleEndian.PutUint32(badDocID[60:64], ailego.CRC32C(badDocID[:60]))
	{
		_, _, err := decodeSegment(context.Background(), badDocID)
		require.ErrorIs(t, err, ErrSegmentCorrupt)
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
		require.NoError(t, err)
	}
	all := make([]uint64, 0, 400)
	for id := range ids {
		all = append(all, id)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	for index, id := range all {
		require.Equal(t, 500+uint64(index), id)
	}
}

func FuzzDecodeSegment(f *testing.F) {
	metadata, docs := sampleSegmentData()
	encoded, err := encodeSegment(metadata, docs)
	require.NoError(f, err)

	f.Add(encoded)
	f.Add(encoded[:segmentHeaderSize])
	f.Add([]byte("not a segment"))
	f.Fuzz(func(t *testing.T, data []byte) {
		metadata, docs, err := decodeSegment(context.Background(), data)
		if err == nil {
			require.Equal(t, uint64(len(docs)), metadata.DocCount)
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
	require.NoError(t, err)
	{
		actual, expected := binary.LittleEndian.Uint32(encoded[56:60]), ailego.CRC32C(encoded[segmentHeaderSize:])
		require.Equal(t, expected, actual)
	}
	require.False(t, bytes.Equal(encoded[:8], make([]byte, 8)),
		"segment magic is empty")
}
