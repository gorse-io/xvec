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
	"slices"
	"sync"
)

// FetchResult preserves request order. A nil Document with nil Err means the
// primary key is absent or logically deleted, matching the pinned baseline.
type FetchResult struct {
	PrimaryKey string
	Document   *StoredDocument
	Err        error
}

// SegmentManager owns sorted immutable segments, one optional write segment,
// the primary-key map, and the logical deletion snapshot.
type SegmentManager struct {
	mu         sync.RWMutex
	immutable  map[uint64]*ImmutableSegment
	writing    *WriteSegment
	primaryKey *PrimaryKeyMap
	deletes    *DeleteStore
}

// NewSegmentManager constructs an empty manager. Nil stores are replaced with
// empty instances.
func NewSegmentManager(primaryKey *PrimaryKeyMap, deletes *DeleteStore) *SegmentManager {
	if primaryKey == nil {
		primaryKey = NewPrimaryKeyMap()
	}
	if deletes == nil {
		deletes = NewDeleteStore()
	}
	return &SegmentManager{
		immutable:  make(map[uint64]*ImmutableSegment),
		primaryKey: primaryKey, deletes: deletes,
	}
}

// PrimaryKeys returns the manager's shared primary-key map.
func (m *SegmentManager) PrimaryKeys() *PrimaryKeyMap {
	if m == nil {
		return nil
	}
	return m.primaryKey
}

// Deletes returns the manager's shared logical deletion set.
func (m *SegmentManager) Deletes() *DeleteStore {
	if m == nil {
		return nil
	}
	return m.deletes
}

// SetWriting installs a write segment after checking ID and document-range
// conflicts.
func (m *SegmentManager) SetWriting(segment *WriteSegment) error {
	if m == nil {
		return errors.New("db: nil segment manager")
	}
	if segment == nil {
		return errors.New("db: nil write segment")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.writing != nil {
		return fmt.Errorf("db: writing segment %d already exists", m.writing.ID())
	}
	if _, exists := m.immutable[segment.ID()]; exists {
		return fmt.Errorf("db: segment ID %d already exists", segment.ID())
	}
	reservedMin, reservedMax := segment.ReservedRange()
	for id, immutable := range m.immutable {
		metadata := immutable.Metadata()
		if metadata.DocCount > 0 && reservedMin <= metadata.MaxDocID && metadata.MinDocID <= reservedMax {
			return fmt.Errorf("db: writing segment %d reserved range overlaps segment %d", segment.ID(), id)
		}
	}
	m.writing = segment
	return nil
}

// Writing returns the current write segment.
func (m *SegmentManager) Writing() *WriteSegment {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.writing
}

// ClearWriting removes and returns the current write segment.
func (m *SegmentManager) ClearWriting() *WriteSegment {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	segment := m.writing
	m.writing = nil
	return segment
}

// RotateWriting atomically replaces the current write segment with its
// immutable snapshot and installs the next empty write segment. The caller
// must have already durably published matching manifest metadata.
func (m *SegmentManager) RotateWriting(currentID uint64, immutable *ImmutableSegment, next *WriteSegment) error {
	if m == nil {
		return errors.New("db: nil segment manager")
	}
	if immutable == nil || next == nil {
		return errors.New("db: nil segment rotation input")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.writing == nil || m.writing.ID() != currentID {
		return fmt.Errorf("db: writing segment changed during rotation")
	}
	if immutable.ID() != currentID {
		return fmt.Errorf("db: immutable segment ID %d does not match writing segment %d", immutable.ID(), currentID)
	}
	if next.ID() == currentID {
		return fmt.Errorf("db: next writing segment reuses ID %d", currentID)
	}
	if _, exists := m.immutable[currentID]; exists {
		return fmt.Errorf("db: segment ID %d already exists", currentID)
	}
	if _, exists := m.immutable[next.ID()]; exists {
		return fmt.Errorf("db: next segment ID %d already exists", next.ID())
	}
	metadata := immutable.Metadata()
	if err := m.checkRangeAgainstImmutableLocked(metadata, currentID); err != nil {
		return err
	}
	nextMin, nextMax := next.ReservedRange()
	if metadata.DocCount > 0 && nextMin <= metadata.MaxDocID && metadata.MinDocID <= nextMax {
		return fmt.Errorf("db: next writing segment %d overlaps flushed segment %d", next.ID(), currentID)
	}
	for id, segment := range m.immutable {
		other := segment.Metadata()
		if other.DocCount > 0 && nextMin <= other.MaxDocID && other.MinDocID <= nextMax {
			return fmt.Errorf("db: next writing segment %d overlaps segment %d", next.ID(), id)
		}
	}
	m.immutable[currentID] = immutable
	m.writing = next
	return nil
}

// AddImmutable adds a verified immutable segment.
func (m *SegmentManager) AddImmutable(segment *ImmutableSegment) error {
	if m == nil {
		return errors.New("db: nil segment manager")
	}
	if segment == nil {
		return errors.New("db: nil immutable segment")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.immutable[segment.ID()]; exists || m.writing != nil && m.writing.ID() == segment.ID() {
		return fmt.Errorf("db: segment ID %d already exists", segment.ID())
	}
	if err := m.checkRangeLocked(segment.Metadata(), segment.ID()); err != nil {
		return err
	}
	m.immutable[segment.ID()] = segment
	return nil
}

// RemoveImmutable removes and returns a segment without deleting its files.
func (m *SegmentManager) RemoveImmutable(segmentID uint64) (*ImmutableSegment, error) {
	if m == nil {
		return nil, errors.New("db: nil segment manager")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	segment, found := m.immutable[segmentID]
	if !found {
		return nil, ErrSegmentNotFound
	}
	delete(m.immutable, segmentID)
	return segment, nil
}

// ImmutableSegments returns segments sorted by minimum document ID and ID.
func (m *SegmentManager) ImmutableSegments() []*ImmutableSegment {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	segments := make([]*ImmutableSegment, 0, len(m.immutable))
	for _, segment := range m.immutable {
		segments = append(segments, segment)
	}
	slices.SortFunc(segments, func(left, right *ImmutableSegment) int {
		leftMeta, rightMeta := left.Metadata(), right.Metadata()
		if leftMeta.MinDocID < rightMeta.MinDocID {
			return -1
		}
		if leftMeta.MinDocID > rightMeta.MinDocID {
			return 1
		}
		return cmpUint64(leftMeta.ID, rightMeta.ID)
	})
	return segments
}

// ImmutableMetadata returns independent metadata sorted like segments.
func (m *SegmentManager) ImmutableMetadata() []SegmentMetadata {
	segments := m.ImmutableSegments()
	metadata := make([]SegmentMetadata, len(segments))
	for index := range segments {
		metadata[index] = segments[index].Metadata()
	}
	return metadata
}

// Document returns a live document by global ID.
func (m *SegmentManager) Document(docID uint64) (StoredDocument, bool) {
	if m == nil || m.deletes.IsDeleted(docID) {
		return StoredDocument{}, false
	}
	m.mu.RLock()
	writing := m.writing
	segments := make([]*ImmutableSegment, 0, len(m.immutable))
	for _, segment := range m.immutable {
		segments = append(segments, segment)
	}
	m.mu.RUnlock()
	if writing != nil {
		if doc, found := writing.Document(docID); found {
			return doc, true
		}
	}
	for _, segment := range segments {
		metadata := segment.Metadata()
		if docID >= metadata.MinDocID && docID <= metadata.MaxDocID {
			return segment.Document(docID)
		}
	}
	return StoredDocument{}, false
}

// DocumentByPrimaryKey resolves the key map and verifies its segment location.
func (m *SegmentManager) DocumentByPrimaryKey(key string) (StoredDocument, bool) {
	if m == nil {
		return StoredDocument{}, false
	}
	location, found := m.primaryKey.Get(key)
	if !found || m.deletes.IsDeleted(location.DocID) {
		return StoredDocument{}, false
	}
	m.mu.RLock()
	writing := m.writing
	segment := m.immutable[location.SegmentID]
	m.mu.RUnlock()
	var document StoredDocument
	if writing != nil && writing.ID() == location.SegmentID {
		document, found = writing.Document(location.DocID)
	} else if segment != nil {
		document, found = segment.Document(location.DocID)
	} else {
		return StoredDocument{}, false
	}
	return document, found && document.PrimaryKey == key
}

// Fetch resolves primary keys in input order. Missing keys are successful nil
// results; context cancellation is returned at batch level and attached to all
// unprocessed entries.
func (m *SegmentManager) Fetch(ctx context.Context, primaryKeys []string) ([]FetchResult, error) {
	if m == nil {
		return nil, errors.New("db: nil segment manager")
	}
	if ctx == nil {
		return nil, errors.New("db: nil fetch context")
	}
	results := make([]FetchResult, len(primaryKeys))
	for index, key := range primaryKeys {
		results[index].PrimaryKey = key
		if err := ctx.Err(); err != nil {
			results[index].Err = err
			for remaining := index + 1; remaining < len(primaryKeys); remaining++ {
				results[remaining] = FetchResult{PrimaryKey: primaryKeys[remaining], Err: err}
			}
			return results, err
		}
		if document, found := m.DocumentByPrimaryKey(key); found {
			copy := document.Clone()
			results[index].Document = &copy
		}
	}
	return results, nil
}

func (m *SegmentManager) checkRangeLocked(candidate SegmentMetadata, candidateID uint64) error {
	if candidate.DocCount == 0 {
		return nil
	}
	for id, segment := range m.immutable {
		if id == candidateID {
			continue
		}
		if rangesOverlap(candidate, segment.Metadata()) {
			return fmt.Errorf("db: segment %d document range overlaps segment %d", candidateID, id)
		}
	}
	if m.writing != nil && m.writing.ID() != candidateID {
		reservedMin, reservedMax := m.writing.ReservedRange()
		if candidate.MinDocID <= reservedMax && reservedMin <= candidate.MaxDocID {
			return fmt.Errorf("db: segment %d document range overlaps writing segment %d reserved range", candidateID, m.writing.ID())
		}
	}
	return nil
}

func (m *SegmentManager) checkRangeAgainstImmutableLocked(candidate SegmentMetadata, candidateID uint64) error {
	if candidate.DocCount == 0 {
		return nil
	}
	for id, segment := range m.immutable {
		if id != candidateID && rangesOverlap(candidate, segment.Metadata()) {
			return fmt.Errorf("db: segment %d document range overlaps segment %d", candidateID, id)
		}
	}
	return nil
}

func rangesOverlap(left, right SegmentMetadata) bool {
	if left.DocCount == 0 || right.DocCount == 0 {
		return false
	}
	return left.MinDocID <= right.MaxDocID && right.MinDocID <= left.MaxDocID
}

func cmpUint64(left, right uint64) int {
	if left < right {
		return -1
	}
	if left > right {
		return 1
	}
	return 0
}
