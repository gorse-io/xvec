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
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/ailego"
)

const (
	segmentCodecVersion     uint16 = 1
	segmentHeaderSize              = 64
	segmentRecordHeaderSize        = 24
	MaxDocumentPayloadSize         = 64 << 20
	maxSegmentPayloadSize   uint64 = 4 << 30
)

var (
	segmentMagic = [8]byte{'Z', 'V', 'E', 'C', 'S', 'E', 'G', 0}

	ErrSegmentCorrupt   = errors.New("db: corrupt segment")
	ErrSegmentSealed    = errors.New("db: write segment is sealed")
	ErrSegmentFull      = errors.New("db: write segment is full")
	ErrSegmentNotFound  = errors.New("db: segment not found")
	ErrDocumentNotFound = errors.New("db: document not found")
)

// StoredDocument is the schema-independent representation held by segments.
// Payload is produced and interpreted by the collection's schema codec.
type StoredDocument struct {
	DocID      uint64
	PrimaryKey string
	Payload    []byte
}

// Clone returns a deep copy of d.
func (d StoredDocument) Clone() StoredDocument {
	d.Payload = slices.Clone(d.Payload)
	return d
}

// WriteSegment accepts sequential documents until it is sealed.
type WriteSegment struct {
	mu       sync.RWMutex
	id       uint64
	minDocID uint64
	maxDocs  uint64
	docs     []StoredDocument
	sealed   bool
}

// NewWriteSegment constructs an empty segment with a fixed global ID range
// start and capacity.
func NewWriteSegment(id, minDocID, maxDocs uint64) (*WriteSegment, error) {
	if maxDocs == 0 {
		return nil, errors.New("db: write segment capacity must be positive")
	}
	if maxDocs-1 > math.MaxUint64-minDocID {
		return nil, errors.New("db: write segment document ID range overflows")
	}
	return &WriteSegment{id: id, minDocID: minDocID, maxDocs: maxDocs}, nil
}

// ID returns the segment ID.
func (s *WriteSegment) ID() uint64 {
	if s == nil {
		return 0
	}
	return s.id
}

// ReservedRange returns the inclusive document-ID range owned by the segment.
func (s *WriteSegment) ReservedRange() (uint64, uint64) {
	if s == nil {
		return 0, 0
	}
	return s.minDocID, s.minDocID + s.maxDocs - 1
}

// Append stores a cloned payload and assigns the next contiguous document ID.
func (s *WriteSegment) Append(ctx context.Context, primaryKey string, payload []byte) (StoredDocument, error) {
	return s.append(ctx, nil, primaryKey, payload)
}

// NextDocumentID returns the ID that the next append will receive.
func (s *WriteSegment) NextDocumentID() (uint64, error) {
	if s == nil {
		return 0, errors.New("db: nil write segment")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.sealed {
		return 0, ErrSegmentSealed
	}
	if uint64(len(s.docs)) >= s.maxDocs {
		return 0, ErrSegmentFull
	}
	return s.minDocID + uint64(len(s.docs)), nil
}

// AppendExpected appends only if expectedDocID is still next. It lets the WAL
// record and in-memory application agree on the assigned global ID.
func (s *WriteSegment) AppendExpected(ctx context.Context, expectedDocID uint64, primaryKey string, payload []byte) (StoredDocument, error) {
	return s.append(ctx, &expectedDocID, primaryKey, payload)
}

func (s *WriteSegment) append(ctx context.Context, expectedDocID *uint64, primaryKey string, payload []byte) (StoredDocument, error) {
	if s == nil {
		return StoredDocument{}, errors.New("db: nil write segment")
	}
	if ctx == nil {
		return StoredDocument{}, errors.New("db: nil segment append context")
	}
	if err := ctx.Err(); err != nil {
		return StoredDocument{}, err
	}
	if err := validatePrimaryKey(primaryKey); err != nil {
		return StoredDocument{}, err
	}
	if len(payload) > MaxDocumentPayloadSize {
		return StoredDocument{}, fmt.Errorf("db: document payload is %d bytes, maximum %d", len(payload), MaxDocumentPayloadSize)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sealed {
		return StoredDocument{}, ErrSegmentSealed
	}
	if uint64(len(s.docs)) >= s.maxDocs {
		return StoredDocument{}, ErrSegmentFull
	}
	nextDocID := s.minDocID + uint64(len(s.docs))
	if expectedDocID != nil && *expectedDocID != nextDocID {
		return StoredDocument{}, fmt.Errorf("db: next document ID is %d, expected %d", nextDocID, *expectedDocID)
	}
	doc := StoredDocument{
		DocID: nextDocID, PrimaryKey: primaryKey,
		Payload: slices.Clone(payload),
	}
	s.docs = append(s.docs, doc)
	return doc.Clone(), nil
}

// Document returns an independent document copy.
func (s *WriteSegment) Document(docID uint64) (StoredDocument, bool) {
	if s == nil {
		return StoredDocument{}, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if docID < s.minDocID || docID-s.minDocID >= uint64(len(s.docs)) {
		return StoredDocument{}, false
	}
	return s.docs[docID-s.minDocID].Clone(), true
}

// Documents returns all documents in ascending ID order.
func (s *WriteSegment) Documents() []StoredDocument {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneDocuments(s.docs)
}

// MemoryUsageBytes returns the encoded record bytes retained by the segment.
func (s *WriteSegment) MemoryUsageBytes() uint64 {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return storedDocumentsMemoryBytes(s.docs)
}

// Metadata returns the current in-memory range without file references.
func (s *WriteSegment) Metadata() SegmentMetadata {
	if s == nil {
		return SegmentMetadata{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return segmentMetadata(s.id, s.docs, nil)
}

// Seal writes one immutable segment file relative to collectionDir and makes
// this write segment reject further appends.
func (s *WriteSegment) Seal(ctx context.Context, collectionDir, relativeName string) (*ImmutableSegment, error) {
	return s.writeImmutable(ctx, collectionDir, relativeName, true)
}

// Snapshot writes the current non-empty contents as an immutable segment
// without sealing the write segment. Collection flush uses this before the
// manifest commit point so a failed publication can safely keep accepting WAL
// backed writes and retry with fresh immutable artifacts.
func (s *WriteSegment) Snapshot(ctx context.Context, collectionDir, relativeName string) (*ImmutableSegment, error) {
	return s.writeImmutable(ctx, collectionDir, relativeName, false)
}

func (s *WriteSegment) writeImmutable(ctx context.Context, collectionDir, relativeName string, seal bool) (*ImmutableSegment, error) {
	if s == nil {
		return nil, errors.New("db: nil write segment")
	}
	if ctx == nil {
		return nil, errors.New("db: nil segment seal context")
	}
	if collectionDir == "" {
		return nil, errors.New("db: empty segment collection directory")
	}
	if err := validatePortableRelativePath(relativeName); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sealed {
		return nil, ErrSegmentSealed
	}
	if len(s.docs) == 0 {
		return nil, errors.New("db: cannot seal an empty segment")
	}
	metadata := segmentMetadata(s.id, s.docs, []string{relativeName})
	encoded, err := encodeSegment(metadata, s.docs)
	if err != nil {
		return nil, err
	}
	if err := writeImmutableSnapshot(ctx, filepath.Join(collectionDir, filepath.FromSlash(relativeName)), encoded); err != nil {
		return nil, err
	}
	if seal {
		s.sealed = true
	}
	return newImmutableSegment(metadata, s.docs), nil
}

// ImmutableSegment is a verified read-only segment snapshot.
type ImmutableSegment struct {
	metadata SegmentMetadata
	docs     []StoredDocument
}

func newImmutableSegment(metadata SegmentMetadata, docs []StoredDocument) *ImmutableSegment {
	return &ImmutableSegment{metadata: cloneSegment(metadata), docs: cloneDocuments(docs)}
}

// OpenImmutableSegment loads and verifies the first data file in metadata.
func OpenImmutableSegment(ctx context.Context, collectionDir string, metadata SegmentMetadata) (*ImmutableSegment, error) {
	if ctx == nil {
		return nil, errors.New("db: nil segment open context")
	}
	if collectionDir == "" {
		return nil, errors.New("db: empty segment collection directory")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateSegment(metadata); err != nil {
		return nil, fmt.Errorf("%w: metadata: %v", ErrSegmentCorrupt, err)
	}
	if len(metadata.Files) == 0 {
		return nil, fmt.Errorf("%w: segment has no data file", ErrSegmentCorrupt)
	}
	name := filepath.Join(collectionDir, filepath.FromSlash(metadata.Files[0]))
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < segmentHeaderSize || uint64(info.Size()-segmentHeaderSize) > maxSegmentPayloadSize {
		return nil, fmt.Errorf("%w: invalid file size %d", ErrSegmentCorrupt, info.Size())
	}
	encoded := make([]byte, int(info.Size()))
	if _, err := io.ReadFull(file, encoded); err != nil {
		return nil, fmt.Errorf("%w: read file: %v", ErrSegmentCorrupt, err)
	}
	decodedMetadata, docs, err := decodeSegment(ctx, encoded)
	if err != nil {
		return nil, err
	}
	decodedMetadata.Files = slices.Clone(metadata.Files)
	if !segmentMetadataEqual(decodedMetadata, metadata) {
		return nil, fmt.Errorf("%w: manifest metadata differs from data file", ErrSegmentCorrupt)
	}
	return newImmutableSegment(metadata, docs), nil
}

// ID returns the immutable segment ID.
func (s *ImmutableSegment) ID() uint64 {
	if s == nil {
		return 0
	}
	return s.metadata.ID
}

// Metadata returns a deep copy.
func (s *ImmutableSegment) Metadata() SegmentMetadata {
	if s == nil {
		return SegmentMetadata{}
	}
	return cloneSegment(s.metadata)
}

// Document returns an independent document copy.
func (s *ImmutableSegment) Document(docID uint64) (StoredDocument, bool) {
	if s == nil || docID < s.metadata.MinDocID || docID > s.metadata.MaxDocID {
		return StoredDocument{}, false
	}
	index := docID - s.metadata.MinDocID
	if index >= uint64(len(s.docs)) || s.docs[index].DocID != docID {
		return StoredDocument{}, false
	}
	return s.docs[index].Clone(), true
}

// Documents returns all immutable documents in ascending ID order.
func (s *ImmutableSegment) Documents() []StoredDocument {
	if s == nil {
		return nil
	}
	return cloneDocuments(s.docs)
}

// MemoryUsageBytes returns the encoded record bytes retained by the segment.
func (s *ImmutableSegment) MemoryUsageBytes() uint64 {
	if s == nil {
		return 0
	}
	return storedDocumentsMemoryBytes(s.docs)
}

func storedDocumentsMemoryBytes(documents []StoredDocument) uint64 {
	var total uint64
	for _, document := range documents {
		size := uint64(segmentRecordHeaderSize + len(document.PrimaryKey) + len(document.Payload))
		if size > math.MaxUint64-total {
			return math.MaxUint64
		}
		total += size
	}
	return total
}

func encodeSegment(metadata SegmentMetadata, docs []StoredDocument) ([]byte, error) {
	if len(docs) == 0 || metadata.DocCount != uint64(len(docs)) {
		return nil, errors.New("db: invalid segment document count")
	}
	var payloadLength uint64
	for index, doc := range docs {
		if err := validateStoredDocument(doc, metadata.MinDocID+uint64(index)); err != nil {
			return nil, err
		}
		payloadLength += segmentRecordHeaderSize + uint64(len(doc.PrimaryKey)) + uint64(len(doc.Payload))
		if payloadLength > maxSegmentPayloadSize || payloadLength > uint64(math.MaxInt) {
			return nil, errors.New("db: segment payload is too large")
		}
	}
	payload := make([]byte, 0, int(payloadLength))
	var recordHeader [segmentRecordHeaderSize]byte
	for _, doc := range docs {
		clear(recordHeader[:])
		binary.LittleEndian.PutUint64(recordHeader[:8], doc.DocID)
		binary.LittleEndian.PutUint32(recordHeader[8:12], uint32(len(doc.PrimaryKey)))
		binary.LittleEndian.PutUint32(recordHeader[12:16], uint32(len(doc.Payload)))
		crc := ailego.CRC32C([]byte(doc.PrimaryKey))
		crc = ailego.UpdateCRC32C(crc, doc.Payload)
		binary.LittleEndian.PutUint32(recordHeader[16:20], crc)
		payload = append(payload, recordHeader[:]...)
		payload = append(payload, doc.PrimaryKey...)
		payload = append(payload, doc.Payload...)
	}
	header := make([]byte, segmentHeaderSize)
	copy(header[:8], segmentMagic[:])
	binary.LittleEndian.PutUint16(header[8:10], segmentCodecVersion)
	binary.LittleEndian.PutUint16(header[10:12], segmentHeaderSize)
	binary.LittleEndian.PutUint64(header[16:24], metadata.ID)
	binary.LittleEndian.PutUint64(header[24:32], metadata.MinDocID)
	binary.LittleEndian.PutUint64(header[32:40], metadata.MaxDocID)
	binary.LittleEndian.PutUint64(header[40:48], metadata.DocCount)
	binary.LittleEndian.PutUint64(header[48:56], uint64(len(payload)))
	binary.LittleEndian.PutUint32(header[56:60], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(header[60:64], ailego.CRC32C(header[:60]))
	return append(header, payload...), nil
}

func decodeSegment(ctx context.Context, encoded []byte) (SegmentMetadata, []StoredDocument, error) {
	if len(encoded) < segmentHeaderSize {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: file is shorter than header", ErrSegmentCorrupt)
	}
	header := encoded[:segmentHeaderSize]
	if !bytes.Equal(header[:8], segmentMagic[:]) {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: invalid magic", ErrSegmentCorrupt)
	}
	if version := binary.LittleEndian.Uint16(header[8:10]); version != segmentCodecVersion {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: segment version %d", ErrUnsupportedFormatVersion, version)
	}
	if size := binary.LittleEndian.Uint16(header[10:12]); size != segmentHeaderSize {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: invalid header size %d", ErrSegmentCorrupt, size)
	}
	if binary.LittleEndian.Uint32(header[12:16]) != 0 {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: nonzero reserved bytes", ErrSegmentCorrupt)
	}
	if actual, expected := ailego.CRC32C(header[:60]), binary.LittleEndian.Uint32(header[60:64]); actual != expected {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: header checksum got %08x, want %08x", ErrSegmentCorrupt, actual, expected)
	}
	payloadLength := binary.LittleEndian.Uint64(header[48:56])
	if payloadLength > maxSegmentPayloadSize || payloadLength != uint64(len(encoded)-segmentHeaderSize) {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: invalid payload length %d", ErrSegmentCorrupt, payloadLength)
	}
	payload := encoded[segmentHeaderSize:]
	if actual, expected := ailego.CRC32C(payload), binary.LittleEndian.Uint32(header[56:60]); actual != expected {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: payload checksum got %08x, want %08x", ErrSegmentCorrupt, actual, expected)
	}
	metadata := SegmentMetadata{
		ID: binary.LittleEndian.Uint64(header[16:24]), MinDocID: binary.LittleEndian.Uint64(header[24:32]),
		MaxDocID: binary.LittleEndian.Uint64(header[32:40]), DocCount: binary.LittleEndian.Uint64(header[40:48]),
	}
	if metadata.DocCount == 0 || metadata.DocCount > uint64(math.MaxInt) {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: invalid document count %d", ErrSegmentCorrupt, metadata.DocCount)
	}
	if metadata.MaxDocID < metadata.MinDocID || metadata.DocCount-1 != metadata.MaxDocID-metadata.MinDocID {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: non-contiguous document range", ErrSegmentCorrupt)
	}
	docs := make([]StoredDocument, 0, int(metadata.DocCount))
	offset := 0
	for index := uint64(0); index < metadata.DocCount; index++ {
		if err := ctx.Err(); err != nil {
			return SegmentMetadata{}, nil, err
		}
		if len(payload)-offset < segmentRecordHeaderSize {
			return SegmentMetadata{}, nil, fmt.Errorf("%w: truncated record %d", ErrSegmentCorrupt, index)
		}
		recordHeader := payload[offset : offset+segmentRecordHeaderSize]
		docID := binary.LittleEndian.Uint64(recordHeader[:8])
		keyLength := uint64(binary.LittleEndian.Uint32(recordHeader[8:12]))
		documentLength := uint64(binary.LittleEndian.Uint32(recordHeader[12:16]))
		expectedCRC := binary.LittleEndian.Uint32(recordHeader[16:20])
		if binary.LittleEndian.Uint32(recordHeader[20:24]) != 0 {
			return SegmentMetadata{}, nil, fmt.Errorf("%w: record %d reserved bytes", ErrSegmentCorrupt, index)
		}
		offset += segmentRecordHeaderSize
		remaining := uint64(len(payload) - offset)
		if keyLength == 0 || keyLength > maxPrimaryKeyBytes || documentLength > MaxDocumentPayloadSize || keyLength+documentLength > remaining {
			return SegmentMetadata{}, nil, fmt.Errorf("%w: invalid record %d lengths", ErrSegmentCorrupt, index)
		}
		key := string(payload[offset : offset+int(keyLength)])
		offset += int(keyLength)
		documentPayload := slices.Clone(payload[offset : offset+int(documentLength)])
		offset += int(documentLength)
		crc := ailego.CRC32C([]byte(key))
		crc = ailego.UpdateCRC32C(crc, documentPayload)
		if crc != expectedCRC {
			return SegmentMetadata{}, nil, fmt.Errorf("%w: record %d checksum", ErrSegmentCorrupt, index)
		}
		doc := StoredDocument{DocID: docID, PrimaryKey: key, Payload: documentPayload}
		if err := validateStoredDocument(doc, metadata.MinDocID+index); err != nil {
			return SegmentMetadata{}, nil, fmt.Errorf("%w: record %d: %v", ErrSegmentCorrupt, index, err)
		}
		docs = append(docs, doc)
	}
	if offset != len(payload) {
		return SegmentMetadata{}, nil, fmt.Errorf("%w: trailing segment payload", ErrSegmentCorrupt)
	}
	return metadata, docs, nil
}

func validateStoredDocument(doc StoredDocument, expectedDocID uint64) error {
	if doc.DocID != expectedDocID {
		return fmt.Errorf("db: document ID %d, want %d", doc.DocID, expectedDocID)
	}
	if err := validatePrimaryKey(doc.PrimaryKey); err != nil {
		return err
	}
	if len(doc.Payload) > MaxDocumentPayloadSize {
		return errors.New("db: document payload is too large")
	}
	return nil
}

func segmentMetadata(id uint64, docs []StoredDocument, files []string) SegmentMetadata {
	metadata := SegmentMetadata{ID: id, DocCount: uint64(len(docs)), Files: slices.Clone(files)}
	if len(docs) > 0 {
		metadata.MinDocID = docs[0].DocID
		metadata.MaxDocID = docs[len(docs)-1].DocID
	}
	return metadata
}

func segmentMetadataEqual(left, right SegmentMetadata) bool {
	return left.ID == right.ID && left.MinDocID == right.MinDocID && left.MaxDocID == right.MaxDocID && left.DocCount == right.DocCount && slices.Equal(left.Files, right.Files)
}

func cloneDocuments(docs []StoredDocument) []StoredDocument {
	if docs == nil {
		return nil
	}
	cloned := make([]StoredDocument, len(docs))
	for index := range docs {
		cloned[index] = docs[index].Clone()
	}
	return cloned
}
