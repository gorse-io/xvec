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
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/ailego/hash"
	"github.com/gorse-io/xvec/internal/db/index/common"
	"github.com/gorse-io/xvec/internal/db/index/segment"
	"github.com/gorse-io/xvec/internal/db/index/storage/wal"
)

const (
	writeOperationVersion    = 3
	writeOperationHeaderSize = 24
)

var (
	writeOperationMagic = [4]byte{'Z', 'O', 'P', '1'}

	ErrPrimaryKeyExists    = errors.New("db: primary key already exists")
	ErrPrimaryKeyNotFound  = errors.New("db: primary key not found")
	ErrWriteEnginePoisoned = errors.New("db: write engine requires reopen")
)

type writeOperationType uint8

const (
	writeOperationInsert writeOperationType = iota + 1
	writeOperationUpsert
	writeOperationUpdate
	writeOperationDelete
)

type writeOperation struct {
	Type       writeOperationType
	DocID      uint64
	PrimaryKey string
	Payload    []byte
}

// WriteInput is one schema-encoded document requested by a write API.
type WriteInput struct {
	PrimaryKey string
	Payload    []byte
}

// WriteResult reports the outcome for one input in the same position.
type WriteResult struct {
	PrimaryKey string
	DocID      uint64
	Err        error
}

// BatchWriteError summarizes per-document failures while preserving errors.Is.
type BatchWriteError struct {
	Failed int
	causes []error
}

func (e *BatchWriteError) Error() string {
	return fmt.Sprintf("db: %d document writes failed", e.Failed)
}

func (e *BatchWriteError) Unwrap() []error { return slices.Clone(e.causes) }

type writeAheadLog interface {
	Append(context.Context, []byte) (uint64, error)
}

// WriteEngine serializes WAL-backed mutations over a SegmentManager. When an
// automatic synchronization fails after a complete append, the engine still
// applies that record, returns the synchronization error, and relies on the
// poisoned WAL to reject later mutations until reopen.
type WriteEngine struct {
	mu       sync.Mutex
	manager  *segment.SegmentManager
	wal      writeAheadLog
	poisoned error
}

// Err reports whether a record reached the outer WAL but failed to apply
// completely. Such a handle must be reopened and replayed before more writes
// or publication operations are safe.
func (e *WriteEngine) Err() error {
	if e == nil {
		return errors.New("db: nil write engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.poisoned
}

// NewWriteEngine validates the dependencies required for WAL-backed mutations.
func NewWriteEngine(manager *segment.SegmentManager, wal *wal.WAL) (*WriteEngine, error) {
	if manager == nil {
		return nil, errors.New("db: nil segment manager")
	}
	if manager.Writing() == nil {
		return nil, errors.New("db: segment manager has no writing segment")
	}
	if wal == nil {
		return nil, errors.New("db: nil write-ahead log")
	}
	return &WriteEngine{manager: manager, wal: wal}, nil
}

// Insert appends documents whose primary keys do not exist to the WAL before
// applying them. WALOptions controls when appended records are synchronized.
// Validation and duplicate errors are reported per input; every failure is
// also included in the returned BatchWriteError.
func (e *WriteEngine) Insert(ctx context.Context, inputs []WriteInput) ([]WriteResult, error) {
	if e == nil {
		return nil, errors.New("db: nil write engine")
	}
	if ctx == nil {
		return nil, errors.New("db: nil insert context")
	}
	if len(inputs) == 0 {
		return nil, errors.New("db: insert batch is empty")
	}
	results := make([]WriteResult, len(inputs))
	e.mu.Lock()
	defer e.mu.Unlock()
	batchError := &BatchWriteError{}
	for index, input := range inputs {
		results[index].PrimaryKey = input.PrimaryKey
		if err := ctx.Err(); err != nil {
			results[index].Err = err
			batchError.add(err)
			for remaining := index + 1; remaining < len(inputs); remaining++ {
				results[remaining] = WriteResult{PrimaryKey: inputs[remaining].PrimaryKey, Err: err}
				batchError.add(err)
			}
			break
		}
		docID, err := e.insertOneLocked(ctx, input)
		results[index].DocID = docID
		results[index].Err = err
		if err != nil {
			batchError.add(err)
		}
	}
	if batchError.Failed > 0 {
		return results, batchError
	}
	return results, nil
}

// Upsert appends a new document version to the WAL. If the key already exists,
// its prior document ID is logically deleted after the WAL append succeeds.
func (e *WriteEngine) Upsert(ctx context.Context, inputs []WriteInput) ([]WriteResult, error) {
	if e == nil {
		return nil, errors.New("db: nil write engine")
	}
	if ctx == nil {
		return nil, errors.New("db: nil upsert context")
	}
	if len(inputs) == 0 {
		return nil, errors.New("db: upsert batch is empty")
	}
	results := make([]WriteResult, len(inputs))
	e.mu.Lock()
	defer e.mu.Unlock()
	batchError := &BatchWriteError{}
	for index, input := range inputs {
		results[index].PrimaryKey = input.PrimaryKey
		if err := ctx.Err(); err != nil {
			results[index].Err = err
			batchError.add(err)
			for remaining := index + 1; remaining < len(inputs); remaining++ {
				results[remaining] = WriteResult{PrimaryKey: inputs[remaining].PrimaryKey, Err: err}
				batchError.add(err)
			}
			break
		}
		docID, err := e.upsertOneLocked(ctx, input)
		results[index].DocID = docID
		results[index].Err = err
		if err != nil {
			batchError.add(err)
		}
	}
	if batchError.Failed > 0 {
		return results, batchError
	}
	return results, nil
}

// Update appends replacement document versions for existing keys to the WAL.
func (e *WriteEngine) Update(ctx context.Context, inputs []WriteInput) ([]WriteResult, error) {
	if e == nil {
		return nil, errors.New("db: nil write engine")
	}
	if ctx == nil {
		return nil, errors.New("db: nil update context")
	}
	if len(inputs) == 0 {
		return nil, errors.New("db: update batch is empty")
	}
	results := make([]WriteResult, len(inputs))
	e.mu.Lock()
	defer e.mu.Unlock()
	batchError := &BatchWriteError{}
	for index, input := range inputs {
		results[index].PrimaryKey = input.PrimaryKey
		if err := ctx.Err(); err != nil {
			results[index].Err = err
			batchError.add(err)
			for remaining := index + 1; remaining < len(inputs); remaining++ {
				results[remaining] = WriteResult{PrimaryKey: inputs[remaining].PrimaryKey, Err: err}
				batchError.add(err)
			}
			break
		}
		docID, err := e.updateOneLocked(ctx, input)
		results[index].DocID = docID
		results[index].Err = err
		if err != nil {
			batchError.add(err)
		}
	}
	if batchError.Failed > 0 {
		return results, batchError
	}
	return results, nil
}

// Delete appends removals to the WAL before updating primary-key mappings and
// logically deleting current document IDs. Immutable segment bytes are never
// rewritten.
func (e *WriteEngine) Delete(ctx context.Context, primaryKeys []string) ([]WriteResult, error) {
	if e == nil {
		return nil, errors.New("db: nil write engine")
	}
	if ctx == nil {
		return nil, errors.New("db: nil delete context")
	}
	if len(primaryKeys) == 0 {
		return nil, errors.New("db: delete batch is empty")
	}
	results := make([]WriteResult, len(primaryKeys))
	e.mu.Lock()
	defer e.mu.Unlock()
	batchError := &BatchWriteError{}
	for index, primaryKey := range primaryKeys {
		results[index].PrimaryKey = primaryKey
		if err := ctx.Err(); err != nil {
			results[index].Err = err
			batchError.add(err)
			for remaining := index + 1; remaining < len(primaryKeys); remaining++ {
				results[remaining] = WriteResult{PrimaryKey: primaryKeys[remaining], Err: err}
				batchError.add(err)
			}
			break
		}
		docID, err := e.deleteOneLocked(ctx, primaryKey)
		results[index].DocID = docID
		results[index].Err = err
		if err != nil {
			batchError.add(err)
		}
	}
	if batchError.Failed > 0 {
		return results, batchError
	}
	return results, nil
}

func (e *WriteEngine) insertOneLocked(ctx context.Context, input WriteInput) (uint64, error) {
	if e.poisoned != nil {
		return 0, e.poisoned
	}
	if err := common.ValidatePrimaryKey(input.PrimaryKey); err != nil {
		return 0, err
	}
	if len(input.Payload) > segment.MaxDocumentPayloadSize {
		return 0, fmt.Errorf("db: document payload is %d bytes, maximum %d", len(input.Payload), segment.MaxDocumentPayloadSize)
	}
	if _, exists, err := e.manager.PrimaryKeys().Get(input.PrimaryKey); err != nil {
		return 0, err
	} else if exists {
		return 0, fmt.Errorf("%w: %q", ErrPrimaryKeyExists, input.PrimaryKey)
	}
	writing := e.manager.Writing()
	if writing == nil {
		return 0, errors.New("db: no writing segment")
	}
	docID, err := writing.NextDocumentID()
	if err != nil {
		return 0, err
	}
	operation := writeOperation{
		Type: writeOperationInsert, DocID: docID,
		PrimaryKey: input.PrimaryKey, Payload: input.Payload,
	}
	encoded, err := encodeWriteOperation(operation)
	if err != nil {
		return 0, err
	}
	lsn, syncErr := e.wal.Append(ctx, encoded)
	if syncErr != nil && lsn == 0 {
		return 0, syncErr
	}
	applyContext := context.WithoutCancel(ctx)
	if err := writing.ApplyExpected(applyContext, docID, input.PrimaryKey, input.Payload); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: apply WAL insert: %w", err)))
	}
	if err := e.manager.PrimaryKeys().PutNew(applyContext, input.PrimaryKey, docID); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: index WAL insert: %w", err)))
	}
	if syncErr != nil {
		return docID, e.poisonLocked(syncErr)
	}
	return docID, syncErr
}

func (e *WriteEngine) upsertOneLocked(ctx context.Context, input WriteInput) (uint64, error) {
	if e.poisoned != nil {
		return 0, e.poisoned
	}
	if err := common.ValidatePrimaryKey(input.PrimaryKey); err != nil {
		return 0, err
	}
	if len(input.Payload) > segment.MaxDocumentPayloadSize {
		return 0, fmt.Errorf("db: document payload is %d bytes, maximum %d", len(input.Payload), segment.MaxDocumentPayloadSize)
	}
	previous, existed, err := e.manager.PrimaryKeys().Get(input.PrimaryKey)
	if err != nil {
		return 0, err
	}
	writing := e.manager.Writing()
	if writing == nil {
		return 0, errors.New("db: no writing segment")
	}
	docID, err := writing.NextDocumentID()
	if err != nil {
		return 0, err
	}
	operation := writeOperation{
		Type: writeOperationUpsert, DocID: docID,
		PrimaryKey: input.PrimaryKey, Payload: input.Payload,
	}
	encoded, err := encodeWriteOperation(operation)
	if err != nil {
		return 0, err
	}
	lsn, syncErr := e.wal.Append(ctx, encoded)
	if syncErr != nil && lsn == 0 {
		return 0, syncErr
	}
	applyContext := context.WithoutCancel(ctx)
	if err := writing.ApplyExpected(applyContext, docID, input.PrimaryKey, input.Payload); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: apply WAL upsert: %w", err)))
	}
	if existed {
		if _, err := e.manager.Deletes().MarkDeleted(applyContext, previous); err != nil {
			return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: delete prior upsert version: %w", err)))
		}
	}
	if _, _, err := e.manager.PrimaryKeys().Put(applyContext, input.PrimaryKey, docID); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: index WAL upsert: %w", err)))
	}
	if syncErr != nil {
		return docID, e.poisonLocked(syncErr)
	}
	return docID, syncErr
}

func (e *WriteEngine) updateOneLocked(ctx context.Context, input WriteInput) (uint64, error) {
	if e.poisoned != nil {
		return 0, e.poisoned
	}
	if err := common.ValidatePrimaryKey(input.PrimaryKey); err != nil {
		return 0, err
	}
	if len(input.Payload) > segment.MaxDocumentPayloadSize {
		return 0, fmt.Errorf("db: document payload is %d bytes, maximum %d", len(input.Payload), segment.MaxDocumentPayloadSize)
	}
	previous, existed, err := e.manager.PrimaryKeys().Get(input.PrimaryKey)
	if err != nil {
		return 0, err
	}
	if !existed {
		return 0, fmt.Errorf("%w: %q", ErrPrimaryKeyNotFound, input.PrimaryKey)
	}
	writing := e.manager.Writing()
	if writing == nil {
		return 0, errors.New("db: no writing segment")
	}
	docID, err := writing.NextDocumentID()
	if err != nil {
		return 0, err
	}
	encoded, err := encodeWriteOperation(writeOperation{
		Type: writeOperationUpdate, DocID: docID,
		PrimaryKey: input.PrimaryKey, Payload: input.Payload,
	})
	if err != nil {
		return 0, err
	}
	lsn, syncErr := e.wal.Append(ctx, encoded)
	if syncErr != nil && lsn == 0 {
		return 0, syncErr
	}
	applyContext := context.WithoutCancel(ctx)
	if err := writing.ApplyExpected(applyContext, docID, input.PrimaryKey, input.Payload); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: apply WAL update: %w", err)))
	}
	if _, err := e.manager.Deletes().MarkDeleted(applyContext, previous); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: delete prior update version: %w", err)))
	}
	if _, _, err := e.manager.PrimaryKeys().Put(applyContext, input.PrimaryKey, docID); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: index WAL update: %w", err)))
	}
	if syncErr != nil {
		return docID, e.poisonLocked(syncErr)
	}
	return docID, syncErr
}

func (e *WriteEngine) deleteOneLocked(ctx context.Context, primaryKey string) (uint64, error) {
	if e.poisoned != nil {
		return 0, e.poisoned
	}
	if err := common.ValidatePrimaryKey(primaryKey); err != nil {
		return 0, err
	}
	docID, existed, err := e.manager.PrimaryKeys().Get(primaryKey)
	if err != nil {
		return 0, err
	}
	if !existed {
		return 0, fmt.Errorf("%w: %q", ErrPrimaryKeyNotFound, primaryKey)
	}
	encoded, err := encodeWriteOperation(writeOperation{
		Type: writeOperationDelete, DocID: docID, PrimaryKey: primaryKey,
	})
	if err != nil {
		return 0, err
	}
	lsn, syncErr := e.wal.Append(ctx, encoded)
	if syncErr != nil && lsn == 0 {
		return 0, syncErr
	}
	applyContext := context.WithoutCancel(ctx)
	if _, err := e.manager.Deletes().MarkDeleted(applyContext, docID); err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: apply WAL delete: %w", err)))
	}
	removed, found, err := e.manager.PrimaryKeys().Delete(applyContext, primaryKey)
	if err != nil {
		return 0, e.poisonLocked(errors.Join(syncErr, fmt.Errorf("db: remove deleted primary key: %w", err)))
	}
	if !found || removed != docID {
		return 0, e.poisonLocked(errors.Join(syncErr, errors.New("db: IDMap changed while applying delete")))
	}
	if syncErr != nil {
		return docID, e.poisonLocked(syncErr)
	}
	return docID, syncErr
}

func (e *WriteEngine) poisonLocked(cause error) error {
	if cause == nil {
		return nil
	}
	if e.poisoned == nil {
		e.poisoned = errors.Join(ErrWriteEnginePoisoned, cause)
	}
	return e.poisoned
}

func (e *BatchWriteError) add(err error) {
	e.Failed++
	e.causes = append(e.causes, err)
}

func encodeWriteOperation(operation writeOperation) ([]byte, error) {
	if operation.Type < writeOperationInsert || operation.Type > writeOperationDelete {
		return nil, errors.New("db: invalid write operation type")
	}
	if err := common.ValidatePrimaryKey(operation.PrimaryKey); err != nil {
		return nil, err
	}
	if len(operation.Payload) > segment.MaxDocumentPayloadSize {
		return nil, errors.New("db: write operation payload is too large")
	}
	encoded := make([]byte, writeOperationHeaderSize+4+len(operation.PrimaryKey)+len(operation.Payload))
	copy(encoded[:4], writeOperationMagic[:])
	binary.LittleEndian.PutUint16(encoded[4:6], writeOperationVersion)
	encoded[6] = byte(operation.Type)
	binary.LittleEndian.PutUint64(encoded[8:16], operation.DocID)
	binary.LittleEndian.PutUint32(encoded[16:20], uint32(len(operation.PrimaryKey)))
	binary.LittleEndian.PutUint32(encoded[20:24], uint32(len(operation.Payload)))
	keyStart := writeOperationHeaderSize + 4
	copy(encoded[keyStart:keyStart+len(operation.PrimaryKey)], operation.PrimaryKey)
	copy(encoded[keyStart+len(operation.PrimaryKey):], operation.Payload)
	crc := hashutil.CRC32C(encoded[:writeOperationHeaderSize])
	crc = hashutil.UpdateCRC32C(crc, encoded[writeOperationHeaderSize+4:])
	binary.LittleEndian.PutUint32(encoded[writeOperationHeaderSize:writeOperationHeaderSize+4], crc)
	return encoded, nil
}

func decodeWriteOperation(encoded []byte) (writeOperation, error) {
	if len(encoded) < writeOperationHeaderSize+4 {
		return writeOperation{}, errors.New("db: truncated write operation")
	}
	if !bytes.Equal(encoded[:4], writeOperationMagic[:]) {
		return writeOperation{}, errors.New("db: invalid write operation magic")
	}
	if version := binary.LittleEndian.Uint16(encoded[4:6]); version != writeOperationVersion {
		return writeOperation{}, fmt.Errorf("%w: write operation version %d", common.ErrUnsupportedFormatVersion, version)
	}
	operationType := writeOperationType(encoded[6])
	if operationType < writeOperationInsert || operationType > writeOperationDelete || encoded[7] != 0 {
		return writeOperation{}, errors.New("db: invalid write operation type or flags")
	}
	keyLength := uint64(binary.LittleEndian.Uint32(encoded[16:20]))
	payloadLength := uint64(binary.LittleEndian.Uint32(encoded[20:24]))
	if keyLength == 0 || keyLength > common.MaxPrimaryKeyBytes || payloadLength > segment.MaxDocumentPayloadSize || keyLength+payloadLength != uint64(len(encoded)-writeOperationHeaderSize-4) {
		return writeOperation{}, errors.New("db: invalid write operation lengths")
	}
	expectedCRC := binary.LittleEndian.Uint32(encoded[writeOperationHeaderSize : writeOperationHeaderSize+4])
	crc := hashutil.CRC32C(encoded[:writeOperationHeaderSize])
	crc = hashutil.UpdateCRC32C(crc, encoded[writeOperationHeaderSize+4:])
	if crc != expectedCRC {
		return writeOperation{}, errors.New("db: write operation checksum mismatch")
	}
	keyStart := writeOperationHeaderSize + 4
	key := string(encoded[keyStart : keyStart+int(keyLength)])
	if err := common.ValidatePrimaryKey(key); err != nil {
		return writeOperation{}, err
	}
	return writeOperation{
		Type: operationType, DocID: binary.LittleEndian.Uint64(encoded[8:16]), PrimaryKey: key,
		Payload: slices.Clone(encoded[keyStart+int(keyLength):]),
	}, nil
}
