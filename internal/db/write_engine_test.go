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
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWriteEngineInsert(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 10, 10)
	defer wal.Close()
	payload := []byte(`{"title":"one"}`)
	results, err := engine.Insert(context.Background(), []WriteInput{
		{PrimaryKey: "one", Payload: payload},
		{PrimaryKey: "two", Payload: []byte(`{"title":"two"}`)},
	})
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.True(t, results[0].DocID == 10)
	require.True(t, results[1].DocID == 11)
	require.NoError(t, results[0].Err)
	require.NoError(t, results[1].Err)

	payload[0] = '['
	{
		doc, found := manager.DocumentByPrimaryKey("one")
		require.True(t, found)
		require.True(t, string(doc.Payload) == `{"title":"one"}`)
	}

	var operations []writeOperation
	{
		err := wal.Replay(context.Background(), func(record WALRecord) error {
			operation, err := decodeWriteOperation(record.Payload)
			if err != nil {
				return err
			}
			operations = append(operations, operation)
			return nil
		})
		require.NoError(t, err)
	}
	require.Len(t, operations, 2)
	require.Equal(t, writeOperationInsert, operations[0].Type)
	require.True(t, operations[0].SegmentID == 1)
	require.True(t, operations[0].DocID == 10)
	require.True(t, operations[0].PrimaryKey == "one")
}

func TestWriteEngineBatchesWALSync(t *testing.T) {
	engine, _, wal := newTestWriteEngine(t, 0, 10)
	defer wal.Close()
	wal.options.SyncEvery = 4

	_, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "key"}})
	require.NoError(t, err)
	require.Equal(t, uint64(1), wal.dirtyRecords)

	_, err = engine.Upsert(context.Background(), []WriteInput{{PrimaryKey: "key"}})
	require.NoError(t, err)
	require.Equal(t, uint64(2), wal.dirtyRecords)

	_, err = engine.Update(context.Background(), []WriteInput{{PrimaryKey: "key"}})
	require.NoError(t, err)
	require.Equal(t, uint64(3), wal.dirtyRecords)

	_, err = engine.Delete(context.Background(), []string{"key"})
	require.NoError(t, err)
	require.Equal(t, uint64(0), wal.dirtyRecords)
}

func TestWriteEngineAppliesRecordWhenAutomaticSyncFails(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 10)
	defer wal.Close()
	wal.options.SyncEvery = 1
	syncError := errors.New("injected WAL sync failure")
	wal.syncFile = func() error { return syncError }

	results, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "key"}})
	require.ErrorIs(t, err, syncError)
	require.ErrorIs(t, err, ErrWALPoisoned)
	require.Len(t, results, 1)
	require.ErrorIs(t, results[0].Err, syncError)
	document, found := manager.DocumentByPrimaryKey("key")
	require.True(t, found)
	require.Equal(t, uint64(0), document.DocID)

	_, err = engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "next"}})
	require.ErrorIs(t, err, ErrWALPoisoned)
}

func TestWriteEngineInsertReportsPerDocumentFailures(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 10)
	defer wal.Close()
	{
		_, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "exists"}})
		require.NoError(t, err)
	}

	results, err := engine.Insert(context.Background(), []WriteInput{
		{PrimaryKey: "exists"},
		{PrimaryKey: ""},
		{PrimaryKey: "new", Payload: []byte("payload")},
	})
	var batchError *BatchWriteError
	require.ErrorAs(t, err, &batchError)
	require.True(t, batchError.Failed == 2)
	require.ErrorIs(t, err, ErrPrimaryKeyExists)
	require.Len(t, results, 3)
	require.ErrorIs(t, results[0].Err, ErrPrimaryKeyExists)
	require.Error(t, results[1].Err)
	require.NoError(t, results[2].Err)
	require.True(t, results[2].DocID == 1)
	{
		_, found := manager.DocumentByPrimaryKey("new")
		require.True(t, found,
			"valid document after failures was not inserted")
	}
}

func TestWriteEngineInsertFullAndCanceled(t *testing.T) {
	engine, _, wal := newTestWriteEngine(t, 0, 1)
	defer wal.Close()
	results, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}, {PrimaryKey: "two"}})
	require.ErrorIs(t, err, ErrSegmentFull)
	require.NoError(t, results[0].Err)
	require.ErrorIs(t, results[1].Err, ErrSegmentFull)

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	results, err = engine.Insert(canceled, []WriteInput{{PrimaryKey: "three"}, {PrimaryKey: "four"}})
	require.ErrorIs(t, err, context.Canceled)
	require.Len(t, results, 2)
	require.ErrorIs(t, results[0].Err, context.Canceled)
	require.ErrorIs(t, results[1].Err, context.Canceled)
	{
		_, err := engine.Insert(context.Background(), nil)
		require.Error(t, err,
			"empty batch succeeded")
	}
	{
		_, err := engine.Insert(nil, []WriteInput{{PrimaryKey: "x"}})
		require.Error(t, err,
			"nil context succeeded")
	}
}

func TestWriteEngineUpsertCreatesAndReplaces(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 50, 10)
	defer wal.Close()
	results, err := engine.Upsert(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v1")}})
	require.NoError(t, err)
	require.True(t, results[0].DocID == 50)

	results, err = engine.Upsert(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v2")}})
	require.NoError(t, err)
	require.True(t, results[0].DocID == 51)
	require.True(t, manager.Deletes().IsDeleted(50),
		"upsert deletion set is wrong")
	require.False(t, manager.Deletes().IsDeleted(51),
		"upsert deletion set is wrong")
	{
		doc, found := manager.DocumentByPrimaryKey("key")
		require.True(t, found)
		require.True(t, doc.DocID == 51)
		require.True(t, string(doc.Payload) == "v2")
	}
	{
		_, found := manager.Document(50)
		require.False(t, found,
			"old upsert version is visible")
	}

	var types []writeOperationType
	{
		err := wal.Replay(context.Background(), func(record WALRecord) error {
			operation, err := decodeWriteOperation(record.Payload)
			if err == nil {
				types = append(types, operation.Type)
			}
			return err
		})
		require.NoError(t, err)
	}
	require.Equal(t, []writeOperationType{writeOperationUpsert, writeOperationUpsert}, types)
}

func TestWriteEngineUpsertPerDocumentErrors(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 2)
	defer wal.Close()
	results, err := engine.Upsert(context.Background(), []WriteInput{
		{PrimaryKey: ""}, {PrimaryKey: "one", Payload: []byte("1")}, {PrimaryKey: "two", Payload: []byte("2")}, {PrimaryKey: "three"},
	})
	var batchError *BatchWriteError
	require.ErrorAs(t, err, &batchError)
	require.True(t, batchError.Failed == 2)
	require.Error(t, results[0].Err)
	require.NoError(t, results[1].Err)
	require.NoError(t, results[2].Err)
	require.ErrorIs(t, results[3].Err, ErrSegmentFull)
	require.True(t, manager.PrimaryKeys().Count() == 2)
	{
		_, err := engine.Upsert(context.Background(), nil)
		require.Error(t, err,
			"empty upsert succeeded")
	}
}

func TestWriteEngineConcurrentUpsertSameKey(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 64)
	defer wal.Close()
	const count = 32
	var wait sync.WaitGroup
	errs := make(chan error, count)
	for index := range count {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := engine.Upsert(context.Background(), []WriteInput{{PrimaryKey: "shared", Payload: []byte(fmt.Sprintf("v%d", index))}})
			errs <- err
		}()
	}
	wait.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	require.True(t, manager.PrimaryKeys().Count() == 1)
	require.Equal(t, uint64(count-1), manager.Deletes().Count())
	{
		doc, found := manager.DocumentByPrimaryKey("shared")
		require.True(t, found)
		require.False(t, manager.Deletes().IsDeleted(doc.DocID))
	}
}

func TestWriteEngineUpdateReplacesExistingOnly(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 10, 10)
	defer wal.Close()
	{
		_, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v1")}})
		require.NoError(t, err)
	}

	results, err := engine.Update(context.Background(), []WriteInput{
		{PrimaryKey: "missing", Payload: []byte("none")},
		{PrimaryKey: "key", Payload: []byte("v2")},
	})
	require.ErrorIs(t, err, ErrPrimaryKeyNotFound)
	require.ErrorIs(t, results[0].Err, ErrPrimaryKeyNotFound)
	require.NoError(t, results[1].Err)
	require.True(t, results[1].DocID == 11)
	require.True(t, manager.Deletes().IsDeleted(10),
		"prior update version is not deleted")
	{
		doc, found := manager.DocumentByPrimaryKey("key")
		require.True(t, found)
		require.True(t, doc.DocID == 11)
		require.True(t, string(doc.Payload) == "v2")
	}

	var types []writeOperationType
	{
		err := wal.Replay(context.Background(), func(record WALRecord) error {
			operation, err := decodeWriteOperation(record.Payload)
			if err == nil {
				types = append(types, operation.Type)
			}
			return err
		})
		require.NoError(t, err)
	}
	require.Equal(t, []writeOperationType{writeOperationInsert, writeOperationUpdate}, types)
}

func TestWriteEngineUpdateValidationAndCapacity(t *testing.T) {
	engine, _, wal := newTestWriteEngine(t, 0, 2)
	defer wal.Close()
	{
		_, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "key"}})
		require.NoError(t, err)
	}

	results, err := engine.Update(context.Background(), []WriteInput{{PrimaryKey: ""}, {PrimaryKey: "key", Payload: []byte("v2")}})
	var batchError *BatchWriteError
	require.ErrorAs(t, err, &batchError)
	require.True(t, batchError.Failed == 1)
	require.Error(t, results[0].Err)
	require.NoError(t, results[1].Err)

	results, err = engine.Update(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v3")}})
	require.ErrorIs(t, err, ErrSegmentFull)
	require.ErrorIs(t, results[0].Err, ErrSegmentFull)
	{
		_, err := engine.Update(context.Background(), nil)
		require.Error(t, err,
			"empty update succeeded")
	}
}

func TestWriteEngineDeleteExistingAndMissing(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 20, 10)
	defer wal.Close()
	{
		_, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}, {PrimaryKey: "two"}})
		require.NoError(t, err)
	}

	results, err := engine.Delete(context.Background(), []string{"missing", "one"})
	require.ErrorIs(t, err, ErrPrimaryKeyNotFound)
	require.ErrorIs(t, results[0].Err, ErrPrimaryKeyNotFound)
	require.NoError(t, results[1].Err)
	require.True(t, results[1].DocID == 20)
	require.True(t, manager.Deletes().IsDeleted(20),
		"deleted document ID is not marked")
	{
		_, found := manager.PrimaryKeys().Get("one")
		require.False(t, found,
			"deleted primary key remains mapped")
	}
	{
		_, found := manager.Document(20)
		require.False(t, found,
			"deleted document remains visible")
	}
	{
		doc, found := manager.DocumentByPrimaryKey("two")
		require.True(t, found)
		require.True(t, doc.DocID == 21)
	}

	results, err = engine.Delete(context.Background(), []string{"one"})
	require.ErrorIs(t, err, ErrPrimaryKeyNotFound)
	require.ErrorIs(t, results[0].Err, ErrPrimaryKeyNotFound)

	var final writeOperation
	{
		err := wal.Replay(context.Background(), func(record WALRecord) error {
			operation, err := decodeWriteOperation(record.Payload)
			if err == nil {
				final = operation
			}
			return err
		})
		require.NoError(t, err)
	}
	require.Equal(t, writeOperationDelete, final.Type)
	require.True(t, final.SegmentID == 1)
	require.True(t, final.DocID == 20)
	require.True(t, final.PrimaryKey == "one")
	require.Len(t, final.Payload, 0)
}

func TestWriteEngineDeleteValidationAndConcurrency(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 10)
	defer wal.Close()
	{
		_, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "shared"}})
		require.NoError(t, err)
	}

	const attempts = 8
	var wait sync.WaitGroup
	errs := make(chan error, attempts)
	for range attempts {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := engine.Delete(context.Background(), []string{"shared"})
			errs <- err
		}()
	}
	wait.Wait()
	close(errs)
	succeeded, missing := 0, 0
	for err := range errs {
		switch {
		case err == nil:
			succeeded++
		case errors.Is(err, ErrPrimaryKeyNotFound):
			missing++
		default:
			require.NoError(t, err)
		}
	}
	require.True(t, succeeded == 1)
	require.Equal(t, attempts-1, missing)
	require.True(t, manager.Deletes().Count() == 1)
	{
		results, err := engine.Delete(context.Background(), []string{""})
		require.Error(t, err)
		require.Error(t, results[0].Err)
	}
	{
		_, err := engine.Delete(context.Background(), nil)
		require.Error(t, err,
			"empty delete batch succeeded")
	}
}

func TestWriteEngineConcurrentInsert(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 100, 200)
	defer wal.Close()
	const count = 100
	ids := make(chan uint64, count)
	errs := make(chan error, count)
	var wait sync.WaitGroup
	for index := range count {
		wait.Add(1)
		go func() {
			defer wait.Done()
			results, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: fmt.Sprintf("key-%03d", index)}})
			if err != nil {
				errs <- err
				return
			}
			ids <- results[0].DocID
		}()
	}
	wait.Wait()
	close(ids)
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	all := make([]uint64, 0, count)
	for id := range ids {
		all = append(all, id)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	for index, id := range all {
		require.Equal(t, 100+uint64(index), id)
	}
	require.Equal(t, count, manager.PrimaryKeys().Count())
}

func TestWriteOperationCodec(t *testing.T) {
	original := writeOperation{
		Type: writeOperationInsert, SegmentID: 9, DocID: 42,
		PrimaryKey: "文档", Payload: []byte{0, 1, 2, 3},
	}
	encoded, err := encodeWriteOperation(original)
	require.NoError(t, err)

	decoded, err := decodeWriteOperation(encoded)
	require.NoError(t, err)
	require.Equal(t, original, decoded)

	for cut := 0; cut < len(encoded); cut++ {
		{
			_, err := decodeWriteOperation(encoded[:cut])
			require.Error(t, err)
		}
	}
	for index := range encoded {
		corrupted := append([]byte(nil), encoded...)
		corrupted[index] ^= 1
		{
			_, err := decodeWriteOperation(corrupted)
			require.Error(t, err)
		}
	}
}

func FuzzDecodeWriteOperation(f *testing.F) {
	encoded, err := encodeWriteOperation(writeOperation{Type: writeOperationInsert, SegmentID: 1, DocID: 2, PrimaryKey: "key", Payload: []byte("value")})
	require.NoError(f, err)

	f.Add(encoded)
	f.Add(encoded[:writeOperationHeaderSize])
	f.Add([]byte("not an operation"))
	f.Fuzz(func(t *testing.T, data []byte) {
		operation, err := decodeWriteOperation(data)
		if err == nil {
			{
				_, err := encodeWriteOperation(operation)
				require.NoError(t, err)
			}
		}
	})
}

func TestNewWriteEngineValidation(t *testing.T) {
	{
		_, err := NewWriteEngine(nil, nil)
		require.Error(t, err,
			"nil manager succeeded")
	}

	manager := NewSegmentManager(nil, nil)
	{
		_, err := NewWriteEngine(manager, nil)
		require.Error(t, err,
			"manager without writing segment succeeded")
	}
}

func newTestWriteEngine(t *testing.T, minDocID, maxDocs uint64) (*WriteEngine, *SegmentManager, *WAL) {
	t.Helper()
	segment, err := NewWriteSegment(1, minDocID, maxDocs)
	require.NoError(t, err)

	manager := NewSegmentManager(nil, nil)
	{
		err := manager.SetWriting(segment)
		require.NoError(t, err)
	}

	wal, err := CreateWAL(context.Background(), filepath.Join(t.TempDir(), "data.wal"), WALOptions{})
	require.NoError(t, err)

	engine, err := NewWriteEngine(manager, wal)
	if err != nil {
		wal.Close()
	}
	require.NoError(t, err)
	return engine, manager, wal
}
