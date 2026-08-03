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

	ErrPrimaryKeyExists = errors.New("db: primary key already exists")
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

// WriteEngine serializes WAL-backed mutations over a SegmentManager.
type WriteEngine struct {
	mu      sync.Mutex
	manager *SegmentManager
	wal     *WAL
}

// NewWriteEngine validates the dependencies required for durable mutations.
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

// Insert durably appends documents whose primary keys do not exist. Validation
// and duplicate errors are reported per input; every failure is also included
// in the returned BatchWriteError.
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

// Upsert durably appends a new document version. If the key already exists,
// its prior document ID is logically deleted after the WAL is synchronized.
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
	if _, err := e.wal.Append(ctx, encoded); err != nil {
		return 0, err
	}
	if err := e.wal.Sync(ctx); err != nil {
		return 0, err
	}
	applyContext := context.WithoutCancel(ctx)
	doc, err := writing.AppendExpected(applyContext, docID, input.PrimaryKey, input.Payload)
	if err != nil {
		return 0, fmt.Errorf("db: apply WAL insert: %w", err)
	}
	if _, _, err := e.manager.PrimaryKeys().Put(applyContext, input.PrimaryKey, DocumentLocation{SegmentID: writing.ID(), DocID: doc.DocID}); err != nil {
		return 0, fmt.Errorf("db: index WAL insert: %w", err)
	}
	return doc.DocID, nil
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
	if _, err := e.wal.Append(ctx, encoded); err != nil {
		return 0, err
	}
	if err := e.wal.Sync(ctx); err != nil {
		return 0, err
	}
	applyContext := context.WithoutCancel(ctx)
	doc, err := writing.AppendExpected(applyContext, docID, input.PrimaryKey, input.Payload)
	if err != nil {
		return 0, fmt.Errorf("db: apply WAL upsert: %w", err)
	}
	if existed {
		if _, err := e.manager.Deletes().MarkDeleted(applyContext, previous.DocID); err != nil {
			return 0, fmt.Errorf("db: delete prior upsert version: %w", err)
		}
	}
	if _, _, err := e.manager.PrimaryKeys().Put(applyContext, input.PrimaryKey, DocumentLocation{SegmentID: writing.ID(), DocID: doc.DocID}); err != nil {
		return 0, fmt.Errorf("db: index WAL upsert: %w", err)
	}
	return doc.DocID, nil
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
