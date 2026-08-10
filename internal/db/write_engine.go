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
	"slices"
	"sync"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	writeOperationVersion    uint16 = 1
	writeOperationHeaderSize        = 32
)

var (
	writeOperationMagic = [4]byte{'Z', 'O', 'P', '1'}

	ErrPrimaryKeyExists   = errors.New("db: primary key already exists")
	ErrPrimaryKeyNotFound = errors.New("db: primary key not found")
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
	SegmentID  uint64
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

// WriteEngine serializes WAL-backed mutations over a SegmentManager. When an
// automatic synchronization fails after a complete append, the engine still
// applies that record, returns the synchronization error, and relies on the
// poisoned WAL to reject later mutations until reopen.
type WriteEngine struct {
	mu      sync.Mutex
	manager *SegmentManager
	wal     *WAL
}

// NewWriteEngine validates the dependencies required for WAL-backed mutations.
func NewWriteEngine(manager *SegmentManager, wal *WAL) (*WriteEngine, error) {
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
	if err := validatePrimaryKey(input.PrimaryKey); err != nil {
		return 0, err
	}
	if len(input.Payload) > MaxDocumentPayloadSize {
		return 0, fmt.Errorf("db: document payload is %d bytes, maximum %d", len(input.Payload), MaxDocumentPayloadSize)
	}
	if _, exists := e.manager.PrimaryKeys().Get(input.PrimaryKey); exists {
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
		Type: writeOperationInsert, SegmentID: writing.ID(), DocID: docID,
		PrimaryKey: input.PrimaryKey, Payload: slices.Clone(input.Payload),
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
	doc, err := writing.AppendExpected(applyContext, docID, input.PrimaryKey, input.Payload)
	if err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: apply WAL insert: %w", err))
	}
	if _, _, err := e.manager.PrimaryKeys().Put(applyContext, input.PrimaryKey, DocumentLocation{SegmentID: writing.ID(), DocID: doc.DocID}); err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: index WAL insert: %w", err))
	}
	return doc.DocID, syncErr
}

func (e *WriteEngine) upsertOneLocked(ctx context.Context, input WriteInput) (uint64, error) {
	if err := validatePrimaryKey(input.PrimaryKey); err != nil {
		return 0, err
	}
	if len(input.Payload) > MaxDocumentPayloadSize {
		return 0, fmt.Errorf("db: document payload is %d bytes, maximum %d", len(input.Payload), MaxDocumentPayloadSize)
	}
	previous, existed := e.manager.PrimaryKeys().Get(input.PrimaryKey)
	writing := e.manager.Writing()
	if writing == nil {
		return 0, errors.New("db: no writing segment")
	}
	docID, err := writing.NextDocumentID()
	if err != nil {
		return 0, err
	}
	operation := writeOperation{
		Type: writeOperationUpsert, SegmentID: writing.ID(), DocID: docID,
		PrimaryKey: input.PrimaryKey, Payload: slices.Clone(input.Payload),
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
	doc, err := writing.AppendExpected(applyContext, docID, input.PrimaryKey, input.Payload)
	if err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: apply WAL upsert: %w", err))
	}
	if existed {
		if _, err := e.manager.Deletes().MarkDeleted(applyContext, previous.DocID); err != nil {
			return 0, errors.Join(syncErr, fmt.Errorf("db: delete prior upsert version: %w", err))
		}
	}
	if _, _, err := e.manager.PrimaryKeys().Put(applyContext, input.PrimaryKey, DocumentLocation{SegmentID: writing.ID(), DocID: doc.DocID}); err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: index WAL upsert: %w", err))
	}
	return doc.DocID, syncErr
}

func (e *WriteEngine) updateOneLocked(ctx context.Context, input WriteInput) (uint64, error) {
	if err := validatePrimaryKey(input.PrimaryKey); err != nil {
		return 0, err
	}
	if len(input.Payload) > MaxDocumentPayloadSize {
		return 0, fmt.Errorf("db: document payload is %d bytes, maximum %d", len(input.Payload), MaxDocumentPayloadSize)
	}
	previous, existed := e.manager.PrimaryKeys().Get(input.PrimaryKey)
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
		Type: writeOperationUpdate, SegmentID: writing.ID(), DocID: docID,
		PrimaryKey: input.PrimaryKey, Payload: slices.Clone(input.Payload),
	})
	if err != nil {
		return 0, err
	}
	lsn, syncErr := e.wal.Append(ctx, encoded)
	if syncErr != nil && lsn == 0 {
		return 0, syncErr
	}
	applyContext := context.WithoutCancel(ctx)
	doc, err := writing.AppendExpected(applyContext, docID, input.PrimaryKey, input.Payload)
	if err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: apply WAL update: %w", err))
	}
	if _, err := e.manager.Deletes().MarkDeleted(applyContext, previous.DocID); err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: delete prior update version: %w", err))
	}
	if _, _, err := e.manager.PrimaryKeys().Put(applyContext, input.PrimaryKey, DocumentLocation{SegmentID: writing.ID(), DocID: doc.DocID}); err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: index WAL update: %w", err))
	}
	return doc.DocID, syncErr
}

func (e *WriteEngine) deleteOneLocked(ctx context.Context, primaryKey string) (uint64, error) {
	if err := validatePrimaryKey(primaryKey); err != nil {
		return 0, err
	}
	location, existed := e.manager.PrimaryKeys().Get(primaryKey)
	if !existed {
		return 0, fmt.Errorf("%w: %q", ErrPrimaryKeyNotFound, primaryKey)
	}
	encoded, err := encodeWriteOperation(writeOperation{
		Type: writeOperationDelete, SegmentID: location.SegmentID,
		DocID: location.DocID, PrimaryKey: primaryKey,
	})
	if err != nil {
		return 0, err
	}
	lsn, syncErr := e.wal.Append(ctx, encoded)
	if syncErr != nil && lsn == 0 {
		return 0, syncErr
	}
	applyContext := context.WithoutCancel(ctx)
	if _, err := e.manager.Deletes().MarkDeleted(applyContext, location.DocID); err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: apply WAL delete: %w", err))
	}
	removed, found, err := e.manager.PrimaryKeys().Delete(applyContext, primaryKey)
	if err != nil {
		return 0, errors.Join(syncErr, fmt.Errorf("db: remove deleted primary key: %w", err))
	}
	if !found || removed != location {
		return 0, errors.Join(syncErr, errors.New("db: primary-key map changed while applying delete"))
	}
	return location.DocID, syncErr
}

func (e *BatchWriteError) add(err error) {
	e.Failed++
	e.causes = append(e.causes, err)
}

func encodeWriteOperation(operation writeOperation) ([]byte, error) {
	if operation.Type < writeOperationInsert || operation.Type > writeOperationDelete {
		return nil, errors.New("db: invalid write operation type")
	}
	if err := validatePrimaryKey(operation.PrimaryKey); err != nil {
		return nil, err
	}
	if len(operation.Payload) > MaxDocumentPayloadSize {
		return nil, errors.New("db: write operation payload is too large")
	}
	encoded := make([]byte, writeOperationHeaderSize+4+len(operation.PrimaryKey)+len(operation.Payload))
	copy(encoded[:4], writeOperationMagic[:])
	binary.LittleEndian.PutUint16(encoded[4:6], writeOperationVersion)
	encoded[6] = byte(operation.Type)
	binary.LittleEndian.PutUint64(encoded[8:16], operation.SegmentID)
	binary.LittleEndian.PutUint64(encoded[16:24], operation.DocID)
	binary.LittleEndian.PutUint32(encoded[24:28], uint32(len(operation.PrimaryKey)))
	binary.LittleEndian.PutUint32(encoded[28:32], uint32(len(operation.Payload)))
	keyStart := writeOperationHeaderSize + 4
	copy(encoded[keyStart:keyStart+len(operation.PrimaryKey)], operation.PrimaryKey)
	copy(encoded[keyStart+len(operation.PrimaryKey):], operation.Payload)
	crc := ailego.CRC32C(encoded[:writeOperationHeaderSize])
	crc = ailego.UpdateCRC32C(crc, encoded[writeOperationHeaderSize+4:])
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
		return writeOperation{}, fmt.Errorf("%w: write operation version %d", ErrUnsupportedFormatVersion, version)
	}
	operationType := writeOperationType(encoded[6])
	if operationType < writeOperationInsert || operationType > writeOperationDelete || encoded[7] != 0 {
		return writeOperation{}, errors.New("db: invalid write operation type or flags")
	}
	keyLength := uint64(binary.LittleEndian.Uint32(encoded[24:28]))
	payloadLength := uint64(binary.LittleEndian.Uint32(encoded[28:32]))
	if keyLength == 0 || keyLength > maxPrimaryKeyBytes || payloadLength > MaxDocumentPayloadSize || keyLength+payloadLength != uint64(len(encoded)-writeOperationHeaderSize-4) {
		return writeOperation{}, errors.New("db: invalid write operation lengths")
	}
	expectedCRC := binary.LittleEndian.Uint32(encoded[writeOperationHeaderSize : writeOperationHeaderSize+4])
	crc := ailego.CRC32C(encoded[:writeOperationHeaderSize])
	crc = ailego.UpdateCRC32C(crc, encoded[writeOperationHeaderSize+4:])
	if crc != expectedCRC {
		return writeOperation{}, errors.New("db: write operation checksum mismatch")
	}
	keyStart := writeOperationHeaderSize + 4
	key := string(encoded[keyStart : keyStart+int(keyLength)])
	if err := validatePrimaryKey(key); err != nil {
		return writeOperation{}, err
	}
	return writeOperation{
		Type: operationType, SegmentID: binary.LittleEndian.Uint64(encoded[8:16]),
		DocID: binary.LittleEndian.Uint64(encoded[16:24]), PrimaryKey: key,
		Payload: slices.Clone(encoded[keyStart+int(keyLength):]),
	}, nil
}
