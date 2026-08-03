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
	"reflect"
	"sort"
	"sync"
	"testing"
)

func TestWriteEngineInsert(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 10, 10)
	defer wal.Close()
	payload := []byte(`{"title":"one"}`)
	results, err := engine.Insert(context.Background(), []WriteInput{
		{PrimaryKey: "one", Payload: payload},
		{PrimaryKey: "two", Payload: []byte(`{"title":"two"}`)},
	})
	if err != nil {
		t.Fatalf("insert: %v", err)
	}
	if len(results) != 2 || results[0].DocID != 10 || results[1].DocID != 11 || results[0].Err != nil || results[1].Err != nil {
		t.Fatalf("results = %#v", results)
	}
	payload[0] = '['
	if doc, found := manager.DocumentByPrimaryKey("one"); !found || string(doc.Payload) != `{"title":"one"}` {
		t.Fatalf("stored one = %#v, %v", doc, found)
	}

	var operations []writeOperation
	if err := wal.Replay(context.Background(), func(record WALRecord) error {
		operation, err := decodeWriteOperation(record.Payload)
		if err != nil {
			return err
		}
		operations = append(operations, operation)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(operations) != 2 || operations[0].Type != writeOperationInsert || operations[0].SegmentID != 1 || operations[0].DocID != 10 || operations[0].PrimaryKey != "one" {
		t.Fatalf("operations = %#v", operations)
	}
}

func TestWriteEngineInsertReportsPerDocumentFailures(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 10)
	defer wal.Close()
	if _, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "exists"}}); err != nil {
		t.Fatal(err)
	}
	results, err := engine.Insert(context.Background(), []WriteInput{
		{PrimaryKey: "exists"},
		{PrimaryKey: ""},
		{PrimaryKey: "new", Payload: []byte("payload")},
	})
	var batchError *BatchWriteError
	if !errors.As(err, &batchError) || batchError.Failed != 2 || !errors.Is(err, ErrPrimaryKeyExists) {
		t.Fatalf("batch error = %#v, %v", batchError, err)
	}
	if len(results) != 3 || !errors.Is(results[0].Err, ErrPrimaryKeyExists) || results[1].Err == nil || results[2].Err != nil || results[2].DocID != 1 {
		t.Fatalf("results = %#v", results)
	}
	if _, found := manager.DocumentByPrimaryKey("new"); !found {
		t.Fatal("valid document after failures was not inserted")
	}
}

func TestWriteEngineInsertFullAndCanceled(t *testing.T) {
	engine, _, wal := newTestWriteEngine(t, 0, 1)
	defer wal.Close()
	results, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}, {PrimaryKey: "two"}})
	if !errors.Is(err, ErrSegmentFull) || results[0].Err != nil || !errors.Is(results[1].Err, ErrSegmentFull) {
		t.Fatalf("full results = %#v, %v", results, err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	results, err = engine.Insert(canceled, []WriteInput{{PrimaryKey: "three"}, {PrimaryKey: "four"}})
	if !errors.Is(err, context.Canceled) || len(results) != 2 || !errors.Is(results[0].Err, context.Canceled) || !errors.Is(results[1].Err, context.Canceled) {
		t.Fatalf("canceled results = %#v, %v", results, err)
	}
	if _, err := engine.Insert(context.Background(), nil); err == nil {
		t.Fatal("empty batch succeeded")
	}
	if _, err := engine.Insert(nil, []WriteInput{{PrimaryKey: "x"}}); err == nil {
		t.Fatal("nil context succeeded")
	}
}

func TestWriteEngineUpsertCreatesAndReplaces(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 50, 10)
	defer wal.Close()
	results, err := engine.Upsert(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v1")}})
	if err != nil || results[0].DocID != 50 {
		t.Fatalf("create upsert = %#v, %v", results, err)
	}
	results, err = engine.Upsert(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v2")}})
	if err != nil || results[0].DocID != 51 {
		t.Fatalf("replace upsert = %#v, %v", results, err)
	}
	if !manager.Deletes().IsDeleted(50) || manager.Deletes().IsDeleted(51) {
		t.Fatal("upsert deletion set is wrong")
	}
	if doc, found := manager.DocumentByPrimaryKey("key"); !found || doc.DocID != 51 || string(doc.Payload) != "v2" {
		t.Fatalf("visible version = %#v, %v", doc, found)
	}
	if _, found := manager.Document(50); found {
		t.Fatal("old upsert version is visible")
	}

	var types []writeOperationType
	if err := wal.Replay(context.Background(), func(record WALRecord) error {
		operation, err := decodeWriteOperation(record.Payload)
		if err == nil {
			types = append(types, operation.Type)
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(types, []writeOperationType{writeOperationUpsert, writeOperationUpsert}) {
		t.Fatalf("operation types = %#v", types)
	}
}

func TestWriteEngineUpsertPerDocumentErrors(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 2)
	defer wal.Close()
	results, err := engine.Upsert(context.Background(), []WriteInput{
		{PrimaryKey: ""}, {PrimaryKey: "one", Payload: []byte("1")}, {PrimaryKey: "two", Payload: []byte("2")}, {PrimaryKey: "three"},
	})
	var batchError *BatchWriteError
	if !errors.As(err, &batchError) || batchError.Failed != 2 {
		t.Fatalf("batch error = %#v, %v", batchError, err)
	}
	if results[0].Err == nil || results[1].Err != nil || results[2].Err != nil || !errors.Is(results[3].Err, ErrSegmentFull) {
		t.Fatalf("results = %#v", results)
	}
	if manager.PrimaryKeys().Count() != 2 {
		t.Fatalf("primary count = %d", manager.PrimaryKeys().Count())
	}
	if _, err := engine.Upsert(context.Background(), nil); err == nil {
		t.Fatal("empty upsert succeeded")
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
		if err != nil {
			t.Fatal(err)
		}
	}
	if manager.PrimaryKeys().Count() != 1 || manager.Deletes().Count() != count-1 {
		t.Fatalf("primary=%d deleted=%d", manager.PrimaryKeys().Count(), manager.Deletes().Count())
	}
	if doc, found := manager.DocumentByPrimaryKey("shared"); !found || manager.Deletes().IsDeleted(doc.DocID) {
		t.Fatalf("latest shared document = %#v, %v", doc, found)
	}
}

func TestWriteEngineUpdateReplacesExistingOnly(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 10, 10)
	defer wal.Close()
	if _, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v1")}}); err != nil {
		t.Fatal(err)
	}
	results, err := engine.Update(context.Background(), []WriteInput{
		{PrimaryKey: "missing", Payload: []byte("none")},
		{PrimaryKey: "key", Payload: []byte("v2")},
	})
	if !errors.Is(err, ErrPrimaryKeyNotFound) || !errors.Is(results[0].Err, ErrPrimaryKeyNotFound) || results[1].Err != nil || results[1].DocID != 11 {
		t.Fatalf("update results = %#v, %v", results, err)
	}
	if !manager.Deletes().IsDeleted(10) {
		t.Fatal("prior update version is not deleted")
	}
	if doc, found := manager.DocumentByPrimaryKey("key"); !found || doc.DocID != 11 || string(doc.Payload) != "v2" {
		t.Fatalf("updated document = %#v, %v", doc, found)
	}

	var types []writeOperationType
	if err := wal.Replay(context.Background(), func(record WALRecord) error {
		operation, err := decodeWriteOperation(record.Payload)
		if err == nil {
			types = append(types, operation.Type)
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(types, []writeOperationType{writeOperationInsert, writeOperationUpdate}) {
		t.Fatalf("operation types = %#v", types)
	}
}

func TestWriteEngineUpdateValidationAndCapacity(t *testing.T) {
	engine, _, wal := newTestWriteEngine(t, 0, 2)
	defer wal.Close()
	if _, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "key"}}); err != nil {
		t.Fatal(err)
	}
	results, err := engine.Update(context.Background(), []WriteInput{{PrimaryKey: ""}, {PrimaryKey: "key", Payload: []byte("v2")}})
	var batchError *BatchWriteError
	if !errors.As(err, &batchError) || batchError.Failed != 1 || results[0].Err == nil || results[1].Err != nil {
		t.Fatalf("update validation = %#v, %#v, %v", results, batchError, err)
	}
	results, err = engine.Update(context.Background(), []WriteInput{{PrimaryKey: "key", Payload: []byte("v3")}})
	if !errors.Is(err, ErrSegmentFull) || !errors.Is(results[0].Err, ErrSegmentFull) {
		t.Fatalf("update full = %#v, %v", results, err)
	}
	if _, err := engine.Update(context.Background(), nil); err == nil {
		t.Fatal("empty update succeeded")
	}
}

func TestWriteEngineDeleteExistingAndMissing(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 20, 10)
	defer wal.Close()
	if _, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}, {PrimaryKey: "two"}}); err != nil {
		t.Fatal(err)
	}
	results, err := engine.Delete(context.Background(), []string{"missing", "one"})
	if !errors.Is(err, ErrPrimaryKeyNotFound) || !errors.Is(results[0].Err, ErrPrimaryKeyNotFound) || results[1].Err != nil || results[1].DocID != 20 {
		t.Fatalf("delete results = %#v, %v", results, err)
	}
	if !manager.Deletes().IsDeleted(20) {
		t.Fatal("deleted document ID is not marked")
	}
	if _, found := manager.PrimaryKeys().Get("one"); found {
		t.Fatal("deleted primary key remains mapped")
	}
	if _, found := manager.Document(20); found {
		t.Fatal("deleted document remains visible")
	}
	if doc, found := manager.DocumentByPrimaryKey("two"); !found || doc.DocID != 21 {
		t.Fatalf("unrelated document = %#v, %v", doc, found)
	}
	results, err = engine.Delete(context.Background(), []string{"one"})
	if !errors.Is(err, ErrPrimaryKeyNotFound) || !errors.Is(results[0].Err, ErrPrimaryKeyNotFound) {
		t.Fatalf("repeated delete = %#v, %v", results, err)
	}

	var final writeOperation
	if err := wal.Replay(context.Background(), func(record WALRecord) error {
		operation, err := decodeWriteOperation(record.Payload)
		if err == nil {
			final = operation
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if final.Type != writeOperationDelete || final.SegmentID != 1 || final.DocID != 20 || final.PrimaryKey != "one" || len(final.Payload) != 0 {
		t.Fatalf("delete operation = %#v", final)
	}
}

func TestWriteEngineDeleteValidationAndConcurrency(t *testing.T) {
	engine, manager, wal := newTestWriteEngine(t, 0, 10)
	defer wal.Close()
	if _, err := engine.Insert(context.Background(), []WriteInput{{PrimaryKey: "shared"}}); err != nil {
		t.Fatal(err)
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
			t.Fatal(err)
		}
	}
	if succeeded != 1 || missing != attempts-1 || manager.Deletes().Count() != 1 {
		t.Fatalf("succeeded=%d missing=%d deleted=%d", succeeded, missing, manager.Deletes().Count())
	}
	if results, err := engine.Delete(context.Background(), []string{""}); err == nil || results[0].Err == nil {
		t.Fatalf("empty key delete = %#v, %v", results, err)
	}
	if _, err := engine.Delete(context.Background(), nil); err == nil {
		t.Fatal("empty delete batch succeeded")
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
		t.Fatal(err)
	}
	all := make([]uint64, 0, count)
	for id := range ids {
		all = append(all, id)
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	for index, id := range all {
		if id != 100+uint64(index) {
			t.Fatalf("ID[%d] = %d", index, id)
		}
	}
	if manager.PrimaryKeys().Count() != count {
		t.Fatalf("primary-key count = %d", manager.PrimaryKeys().Count())
	}
}

func TestWriteOperationCodec(t *testing.T) {
	original := writeOperation{
		Type: writeOperationInsert, SegmentID: 9, DocID: 42,
		PrimaryKey: "文档", Payload: []byte{0, 1, 2, 3},
	}
	encoded, err := encodeWriteOperation(original)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeWriteOperation(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, original) {
		t.Fatalf("decoded = %#v, want %#v", decoded, original)
	}
	for cut := 0; cut < len(encoded); cut++ {
		if _, err := decodeWriteOperation(encoded[:cut]); err == nil {
			t.Fatalf("truncation at %d succeeded", cut)
		}
	}
	for index := range encoded {
		corrupted := append([]byte(nil), encoded...)
		corrupted[index] ^= 1
		if _, err := decodeWriteOperation(corrupted); err == nil {
			t.Fatalf("corruption at %d succeeded", index)
		}
	}
}

func FuzzDecodeWriteOperation(f *testing.F) {
	encoded, err := encodeWriteOperation(writeOperation{Type: writeOperationInsert, SegmentID: 1, DocID: 2, PrimaryKey: "key", Payload: []byte("value")})
	if err != nil {
		f.Fatal(err)
	}
	f.Add(encoded)
	f.Add(encoded[:writeOperationHeaderSize])
	f.Add([]byte("not an operation"))
	f.Fuzz(func(t *testing.T, data []byte) {
		operation, err := decodeWriteOperation(data)
		if err == nil {
			if _, err := encodeWriteOperation(operation); err != nil {
				t.Fatalf("decoded operation does not encode: %v", err)
			}
		}
	})
}

func TestNewWriteEngineValidation(t *testing.T) {
	if _, err := NewWriteEngine(nil, nil); err == nil {
		t.Fatal("nil manager succeeded")
	}
	manager := NewSegmentManager(nil, nil)
	if _, err := NewWriteEngine(manager, nil); err == nil {
		t.Fatal("manager without writing segment succeeded")
	}
}

func newTestWriteEngine(t *testing.T, minDocID, maxDocs uint64) (*WriteEngine, *SegmentManager, *WAL) {
	t.Helper()
	segment, err := NewWriteSegment(1, minDocID, maxDocs)
	if err != nil {
		t.Fatal(err)
	}
	manager := NewSegmentManager(nil, nil)
	if err := manager.SetWriting(segment); err != nil {
		t.Fatal(err)
	}
	wal, err := CreateWAL(context.Background(), filepath.Join(t.TempDir(), "data.wal"), WALOptions{})
	if err != nil {
		t.Fatal(err)
	}
	engine, err := NewWriteEngine(manager, wal)
	if err != nil {
		wal.Close()
		t.Fatal(err)
	}
	return engine, manager, wal
}
