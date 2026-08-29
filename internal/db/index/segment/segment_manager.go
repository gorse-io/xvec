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

package segment

import (
	"context"
	"errors"
	"fmt"
	"github.com/gorse-io/xvec/internal/db/index/common"
	"slices"
	"sort"
	"sync"
)

// FetchResult preserves request order. A nil Document with nil Err means the
// primary key is absent or logically deleted, matching the pinned baseline.
type FetchResult struct {
	PrimaryKey string
	Document   *StoredDocument
	Err        error
}

// StorageStats describes retained segment data without allocating document
// copies. MemoryUsageBytes counts encoded record headers, keys, payloads, and
// logical-deletion IDs.
type StorageStats struct {
	ImmutableSegmentCount uint64
	MutableDocumentCount  uint64
	DeletedDocumentCount  uint64
	MemoryUsageBytes      uint64
}

// SegmentManager owns sorted immutable segments, one optional write segment,
// the primary-key map, and the logical deletion snapshot.
type SegmentManager struct {
	mu         sync.RWMutex
	immutable  map[uint64]*ImmutableSegment
	ordered    []*ImmutableSegment
	writing    *WriteSegment
	primaryKey *common.PrimaryKeyMap
	deletes    *common.DeleteStore
}

// NewSegmentManager constructs an empty manager. Nil stores are replaced with
// empty instances.
func NewSegmentManager(primaryKey *common.PrimaryKeyMap, deletes *common.DeleteStore) *SegmentManager {
	if primaryKey == nil {
		primaryKey = common.NewPrimaryKeyMap()
	}
	if deletes == nil {
		deletes = common.NewDeleteStore()
	}
	return &SegmentManager{
		immutable:  make(map[uint64]*ImmutableSegment),
		primaryKey: primaryKey, deletes: deletes,
	}
}

// PrimaryKeys returns the manager's shared primary-key map.
func (m *SegmentManager) PrimaryKeys() *common.PrimaryKeyMap {
	if m == nil {
		return nil
	}
	return m.primaryKey
}

// ReplacePrimaryKeys swaps the logical IDMap while the collection-level lock
// excludes readers and writers. It is used only after CURRENT commits a new
// immutable checkpoint.
func (m *SegmentManager) ReplacePrimaryKeys(primaryKey *common.PrimaryKeyMap) error {
	if m == nil {
		return errors.New("db: nil segment manager")
	}
	if primaryKey == nil {
		return errors.New("db: nil replacement IDMap")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.primaryKey = primaryKey
	return nil
}

// Deletes returns the manager's shared logical deletion set.
func (m *SegmentManager) Deletes() *common.DeleteStore {
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
	if metadata.DocCount == 0 {
		return errors.New("db: cannot rotate an empty immutable segment")
	}
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
	m.insertOrderedLocked(immutable)
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
	if segment.Metadata().DocCount == 0 {
		return errors.New("db: empty immutable segment")
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
	m.insertOrderedLocked(segment)
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
	for index, candidate := range m.ordered {
		if candidate.ID() == segmentID {
			m.ordered = append(m.ordered[:index], m.ordered[index+1:]...)
			break
		}
	}
	return segment, nil
}

// ImmutableSegments returns segments sorted by minimum document ID and ID.
func (m *SegmentManager) ImmutableSegments() []*ImmutableSegment {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return slices.Clone(m.ordered)
}

// ImmutableMetadata returns independent metadata sorted like segments.
func (m *SegmentManager) ImmutableMetadata() []common.SegmentMetadata {
	segments := m.ImmutableSegments()
	metadata := make([]common.SegmentMetadata, len(segments))
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
	return m.RetainedDocument(docID)
}

// RetainedDocument returns a stored version by global ID without consulting
// the current logical-deletion set. Snapshot iterators use it to keep versions
// that were live when the iterator was created visible after later writes.
func (m *SegmentManager) RetainedDocument(docID uint64) (StoredDocument, bool) {
	if m == nil {
		return StoredDocument{}, false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	writing := m.writing
	if writing != nil {
		if doc, found := writing.Document(docID); found {
			return doc, true
		}
	}
	index, found := locateImmutableSegment(m.ordered, docID)
	if found {
		return m.ordered[index].Document(docID)
	}
	return StoredDocument{}, false
}

// DocumentByPrimaryKey resolves the IDMap and verifies the target document's
// primary key so a corrupt mapping cannot return another document.
func (m *SegmentManager) DocumentByPrimaryKey(key string) (StoredDocument, bool, error) {
	if m == nil {
		return StoredDocument{}, false, errors.New("db: nil segment manager")
	}
	docID, found, err := m.primaryKey.Get(key)
	if err != nil {
		return StoredDocument{}, false, err
	}
	if !found || m.deletes.IsDeleted(docID) {
		return StoredDocument{}, false, nil
	}
	document, found := m.Document(docID)
	return document, found && document.PrimaryKey == key, nil
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
		document, found, err := m.DocumentByPrimaryKey(key)
		if err != nil {
			results[index].Err = err
			for remaining := index + 1; remaining < len(primaryKeys); remaining++ {
				results[remaining] = FetchResult{PrimaryKey: primaryKeys[remaining], Err: err}
			}
			return results, err
		}
		if found {
			copy := document.Clone()
			results[index].Document = &copy
		}
	}
	return results, nil
}

// LiveDocuments returns independent copies of every current document in
// ascending global document-ID order. Superseded and deleted versions are
// omitted. The caller is responsible for excluding concurrent multi-step
// writes when it needs a transactionally stable query snapshot.
func (m *SegmentManager) LiveDocuments(ctx context.Context) ([]StoredDocument, error) {
	if m == nil {
		return nil, errors.New("db: nil segment manager")
	}
	if ctx == nil {
		return nil, errors.New("db: nil live-document context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	segments := m.ImmutableSegments()
	m.mu.RLock()
	writing := m.writing
	m.mu.RUnlock()

	result := make([]StoredDocument, 0, m.primaryKey.Count())
	appendLive := func(documents []StoredDocument) error {
		for _, document := range documents {
			if err := ctx.Err(); err != nil {
				return err
			}
			if m.deletes.IsDeleted(document.DocID) {
				continue
			}
			docID, found, err := m.primaryKey.Get(document.PrimaryKey)
			if err != nil {
				return err
			}
			if !found || docID != document.DocID {
				continue
			}
			result = append(result, document.Clone())
		}
		return nil
	}
	for _, segment := range segments {
		if err := appendLive(segment.Documents()); err != nil {
			return nil, err
		}
	}
	if writing != nil {
		if err := appendLive(writing.Documents()); err != nil {
			return nil, err
		}
	}
	if result == nil {
		return []StoredDocument{}, nil
	}
	return result, nil
}

// LiveDocumentPrimaryKeys returns the authoritative primary key for every
// live document ID without cloning stored document payloads.
func (m *SegmentManager) LiveDocumentPrimaryKeys(ctx context.Context) (map[uint64]string, error) {
	if m == nil {
		return nil, errors.New("db: nil segment manager")
	}
	if ctx == nil {
		return nil, errors.New("db: nil live-document context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make(map[uint64]string, m.primaryKey.Count())
	err := m.primaryKey.ForEach(ctx, func(primaryKey string, docID uint64) error {
		if previous, found := result[docID]; found {
			return fmt.Errorf("db: document ID %d is assigned to primary keys %q and %q", docID, previous, primaryKey)
		}
		result[docID] = primaryKey
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// StorageStats returns a stable snapshot of retained segment resources.
func (m *SegmentManager) StorageStats() StorageStats {
	if m == nil {
		return StorageStats{}
	}
	m.mu.RLock()
	writing := m.writing
	segments := make([]*ImmutableSegment, 0, len(m.immutable))
	for _, segment := range m.immutable {
		segments = append(segments, segment)
	}
	m.mu.RUnlock()
	stats := StorageStats{
		ImmutableSegmentCount: uint64(len(segments)),
		DeletedDocumentCount:  uint64(m.deletes.Count()),
	}
	stats.MemoryUsageBytes = saturatingMultiplyByEight(stats.DeletedDocumentCount)
	for _, segment := range segments {
		stats.MemoryUsageBytes = saturatingAdd(stats.MemoryUsageBytes, segment.MemoryUsageBytes())
	}
	if writing != nil {
		stats.MutableDocumentCount = writing.Metadata().DocCount
		stats.MemoryUsageBytes = saturatingAdd(stats.MemoryUsageBytes, writing.MemoryUsageBytes())
	}
	return stats
}

func saturatingAdd(left, right uint64) uint64 {
	if right > ^uint64(0)-left {
		return ^uint64(0)
	}
	return left + right
}

func saturatingMultiplyByEight(value uint64) uint64 {
	if value > ^uint64(0)/8 {
		return ^uint64(0)
	}
	return value * 8
}

func (m *SegmentManager) checkRangeLocked(candidate common.SegmentMetadata, candidateID uint64) error {
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

func (m *SegmentManager) checkRangeAgainstImmutableLocked(candidate common.SegmentMetadata, candidateID uint64) error {
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

func rangesOverlap(left, right common.SegmentMetadata) bool {
	if left.DocCount == 0 || right.DocCount == 0 {
		return false
	}
	return left.MinDocID <= right.MaxDocID && right.MinDocID <= left.MaxDocID
}

func (m *SegmentManager) insertOrderedLocked(segment *ImmutableSegment) {
	metadata := segment.Metadata()
	index, _ := slices.BinarySearchFunc(m.ordered, metadata.MinDocID, func(candidate *ImmutableSegment, docID uint64) int {
		return cmpUint64(candidate.Metadata().MinDocID, docID)
	})
	m.ordered = append(m.ordered, nil)
	copy(m.ordered[index+1:], m.ordered[index:])
	m.ordered[index] = segment
}

// locateImmutableSegment uses binary search over non-overlapping document-ID
// ranges maintained in ascending order.
func locateImmutableSegment(segments []*ImmutableSegment, docID uint64) (int, bool) {
	index := sort.Search(len(segments), func(index int) bool {
		return segments[index].Metadata().MaxDocID >= docID
	})
	if index == len(segments) {
		return 0, false
	}
	metadata := segments[index].Metadata()
	return index, metadata.DocCount > 0 && metadata.MinDocID <= docID
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
