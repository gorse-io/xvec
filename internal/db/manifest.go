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
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	// DiskFormatVersion is the first native Go collection format. It is not
	// compatible with the C++ collection format.
	DiskFormatVersion uint32 = 1

	manifestHeaderSize = 32
	maxManifestSize    = 64 << 20
)

var (
	manifestMagic = [8]byte{'Z', 'V', 'E', 'C', 'M', 'A', 'N', 0}

	ErrManifestNotFound         = errors.New("db: manifest not found")
	ErrManifestCorrupt          = errors.New("db: corrupt manifest")
	ErrManifestConflict         = errors.New("db: manifest version conflict")
	ErrManifestExists           = errors.New("db: collection manifest already exists")
	ErrUnsupportedFormatVersion = errors.New("db: unsupported disk format version")
)

// SegmentMetadata identifies immutable files owned by one segment. File names
// use slash-separated paths relative to the collection directory so manifests
// remain portable across operating systems.
type SegmentMetadata struct {
	ID       uint64   `json:"id"`
	MinDocID uint64   `json:"min_doc_id"`
	MaxDocID uint64   `json:"max_doc_id"`
	DocCount uint64   `json:"doc_count"`
	Files    []string `json:"files,omitempty"`
}

// Manifest is one immutable collection metadata snapshot. Schema contains the
// versioned JSON representation owned by the schema codec; the manifest layer
// preserves it without importing the public package.
type Manifest struct {
	FormatVersion            uint32            `json:"format_version"`
	Generation               uint64            `json:"generation"`
	Schema                   json.RawMessage   `json:"schema"`
	EnableMmap               bool              `json:"enable_mmap"`
	SegmentMaxDocuments      uint64            `json:"segment_max_documents"`
	PersistedSegments        []SegmentMetadata `json:"persisted_segments,omitempty"`
	WritingSegment           *SegmentMetadata  `json:"writing_segment,omitempty"`
	WritingSegmentStartDocID uint64            `json:"writing_segment_start_doc_id,omitempty"`
	IDMapGeneration          uint64            `json:"id_map_generation"`
	DeleteSnapshotGeneration uint64            `json:"delete_snapshot_generation"`
	NextSegmentID            uint64            `json:"next_segment_id"`
}

// Clone returns a deep copy of m.
func (m Manifest) Clone() Manifest {
	clone := m
	clone.Schema = slices.Clone(m.Schema)
	clone.PersistedSegments = cloneSegments(m.PersistedSegments)
	if m.WritingSegment != nil {
		writing := cloneSegment(*m.WritingSegment)
		clone.WritingSegment = &writing
	}
	return clone
}

// Validate checks all format-level invariants. Schema semantics are checked by
// the schema codec; this layer requires a non-null JSON object.
func (m Manifest) Validate() error {
	if m.FormatVersion != DiskFormatVersion {
		return fmt.Errorf("%w: got %d, support %d", ErrUnsupportedFormatVersion, m.FormatVersion, DiskFormatVersion)
	}
	if m.Generation == 0 {
		return fmt.Errorf("%w: generation must be positive", ErrManifestCorrupt)
	}
	if !json.Valid(m.Schema) {
		return fmt.Errorf("%w: schema is not valid JSON", ErrManifestCorrupt)
	}
	if m.SegmentMaxDocuments == 0 {
		return fmt.Errorf("%w: segment capacity must be positive", ErrManifestCorrupt)
	}
	var schemaObject map[string]json.RawMessage
	if err := json.Unmarshal(m.Schema, &schemaObject); err != nil || schemaObject == nil {
		return fmt.Errorf("%w: schema must be a JSON object", ErrManifestCorrupt)
	}

	ids := make(map[uint64]struct{}, len(m.PersistedSegments)+1)
	maxSegmentID := uint64(0)
	for index := range m.PersistedSegments {
		segment := &m.PersistedSegments[index]
		if err := validateSegment(*segment); err != nil {
			return fmt.Errorf("%w: persisted segment %d: %v", ErrManifestCorrupt, index, err)
		}
		if _, exists := ids[segment.ID]; exists {
			return fmt.Errorf("%w: duplicate segment ID %d", ErrManifestCorrupt, segment.ID)
		}
		ids[segment.ID] = struct{}{}
		maxSegmentID = max(maxSegmentID, segment.ID)
	}
	if m.WritingSegment != nil {
		if err := validateSegment(*m.WritingSegment); err != nil {
			return fmt.Errorf("%w: writing segment: %v", ErrManifestCorrupt, err)
		}
		if _, exists := ids[m.WritingSegment.ID]; exists {
			return fmt.Errorf("%w: writing segment ID %d is already persisted", ErrManifestCorrupt, m.WritingSegment.ID)
		}
		maxSegmentID = max(maxSegmentID, m.WritingSegment.ID)
	}
	if len(m.PersistedSegments) > 0 || m.WritingSegment != nil {
		if m.NextSegmentID <= maxSegmentID {
			return fmt.Errorf("%w: next segment ID %d must exceed %d", ErrManifestCorrupt, m.NextSegmentID, maxSegmentID)
		}
	}
	return nil
}

// MarshalManifest encodes m with a magic value, format version, payload
// length, generation, and CRC32C checksum.
func MarshalManifest(m Manifest) ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("db: marshal manifest payload: %w", err)
	}
	if len(payload) > maxManifestSize {
		return nil, fmt.Errorf("%w: payload is %d bytes", ErrManifestCorrupt, len(payload))
	}

	encoded := make([]byte, manifestHeaderSize+len(payload))
	copy(encoded[:8], manifestMagic[:])
	binary.LittleEndian.PutUint16(encoded[8:10], uint16(DiskFormatVersion))
	binary.LittleEndian.PutUint16(encoded[10:12], manifestHeaderSize)
	binary.LittleEndian.PutUint64(encoded[12:20], m.Generation)
	binary.LittleEndian.PutUint64(encoded[20:28], uint64(len(payload)))
	binary.LittleEndian.PutUint32(encoded[28:32], ailego.CRC32C(payload))
	copy(encoded[manifestHeaderSize:], payload)
	return encoded, nil
}

// UnmarshalManifest verifies and decodes one complete manifest file.
func UnmarshalManifest(encoded []byte) (Manifest, error) {
	if len(encoded) < manifestHeaderSize {
		return Manifest{}, fmt.Errorf("%w: file is shorter than the header", ErrManifestCorrupt)
	}
	if !bytes.Equal(encoded[:8], manifestMagic[:]) {
		return Manifest{}, fmt.Errorf("%w: invalid magic", ErrManifestCorrupt)
	}
	formatVersion := binary.LittleEndian.Uint16(encoded[8:10])
	if uint32(formatVersion) != DiskFormatVersion {
		return Manifest{}, fmt.Errorf("%w: got %d, support %d", ErrUnsupportedFormatVersion, formatVersion, DiskFormatVersion)
	}
	if headerSize := binary.LittleEndian.Uint16(encoded[10:12]); headerSize != manifestHeaderSize {
		return Manifest{}, fmt.Errorf("%w: invalid header size %d", ErrManifestCorrupt, headerSize)
	}
	generation := binary.LittleEndian.Uint64(encoded[12:20])
	payloadLength := binary.LittleEndian.Uint64(encoded[20:28])
	if payloadLength > maxManifestSize || payloadLength > uint64(len(encoded)-manifestHeaderSize) {
		return Manifest{}, fmt.Errorf("%w: invalid payload length %d", ErrManifestCorrupt, payloadLength)
	}
	if uint64(len(encoded)-manifestHeaderSize) != payloadLength {
		return Manifest{}, fmt.Errorf("%w: trailing or truncated payload", ErrManifestCorrupt)
	}
	payload := encoded[manifestHeaderSize:]
	expectedCRC := binary.LittleEndian.Uint32(encoded[28:32])
	if actualCRC := ailego.CRC32C(payload); actualCRC != expectedCRC {
		return Manifest{}, fmt.Errorf("%w: checksum got %08x, want %08x", ErrManifestCorrupt, actualCRC, expectedCRC)
	}

	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var manifest Manifest
	if err := decoder.Decode(&manifest); err != nil {
		return Manifest{}, fmt.Errorf("%w: decode payload: %v", ErrManifestCorrupt, err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return Manifest{}, fmt.Errorf("%w: decode payload: %v", ErrManifestCorrupt, err)
	}
	if manifest.Generation != generation {
		return Manifest{}, fmt.Errorf("%w: header generation %d differs from payload %d", ErrManifestCorrupt, generation, manifest.Generation)
	}
	if err := manifest.Validate(); err != nil {
		return Manifest{}, err
	}
	return manifest.Clone(), nil
}

func validateSegment(segment SegmentMetadata) error {
	if segment.DocCount == 0 {
		if segment.MinDocID != 0 || segment.MaxDocID != 0 {
			return errors.New("empty segment has a document range")
		}
	} else {
		if segment.MaxDocID < segment.MinDocID {
			return errors.New("maximum document ID is below minimum")
		}
		if segment.DocCount-1 > segment.MaxDocID-segment.MinDocID {
			return errors.New("document count exceeds the document ID range")
		}
	}
	files := make(map[string]struct{}, len(segment.Files))
	for _, name := range segment.Files {
		if err := validatePortableRelativePath(name); err != nil {
			return err
		}
		if _, exists := files[name]; exists {
			return fmt.Errorf("duplicate file %q", name)
		}
		files[name] = struct{}{}
	}
	return nil
}

func cloneSegments(segments []SegmentMetadata) []SegmentMetadata {
	if segments == nil {
		return nil
	}
	cloned := make([]SegmentMetadata, len(segments))
	for index := range segments {
		cloned[index] = cloneSegment(segments[index])
	}
	return cloned
}

func cloneSegment(segment SegmentMetadata) SegmentMetadata {
	segment.Files = slices.Clone(segment.Files)
	return segment
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("multiple JSON values")
	}
	return err
}
