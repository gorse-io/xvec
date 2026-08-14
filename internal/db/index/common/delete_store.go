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

package common

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
)

var deleteSnapshotMagic = [8]byte{'Z', 'V', 'E', 'C', 'D', 'E', 'L', 0}

// DeleteStore tracks global document IDs hidden by logical deletion.
type DeleteStore struct {
	mu      sync.RWMutex
	deleted map[uint64]struct{}
}

// NewDeleteStore returns an empty logical deletion set.
func NewDeleteStore() *DeleteStore {
	return &DeleteStore{deleted: make(map[uint64]struct{})}
}

// MarkDeleted marks docID and reports whether the set changed.
func (s *DeleteStore) MarkDeleted(ctx context.Context, docID uint64) (bool, error) {
	if s == nil {
		return false, errors.New("db: nil delete store")
	}
	if ctx == nil {
		return false, errors.New("db: nil delete context")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_, existed := s.deleted[docID]
	s.deleted[docID] = struct{}{}
	return !existed, nil
}

// Restore clears a logical deletion and reports whether the set changed.
func (s *DeleteStore) Restore(ctx context.Context, docID uint64) (bool, error) {
	if s == nil {
		return false, errors.New("db: nil delete store")
	}
	if ctx == nil {
		return false, errors.New("db: nil restore context")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_, existed := s.deleted[docID]
	delete(s.deleted, docID)
	return existed, nil
}

// IsDeleted reports whether docID is logically deleted.
func (s *DeleteStore) IsDeleted(docID uint64) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, deleted := s.deleted[docID]
	return deleted
}

func (d *DeleteStore) Range(yield func(docID uint64) bool) {
	if d == nil || yield == nil {
		return
	}
	d.mu.RLock()
	ids := make([]uint64, 0, len(d.deleted))
	for docID := range d.deleted {
		ids = append(ids, docID)
	}
	d.mu.RUnlock()
	slices.Sort(ids)
	for _, docID := range ids {
		if !yield(docID) {
			return
		}
	}
}

// Count returns the number of logically deleted document IDs.
func (s *DeleteStore) Count() uint64 {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return uint64(len(s.deleted))
}

// RangeCount returns deletions in the inclusive document-ID interval.
func (s *DeleteStore) RangeCount(minDocID, maxDocID uint64) uint64 {
	if s == nil || maxDocID < minDocID {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	var count uint64
	for docID := range s.deleted {
		if docID >= minDocID && docID <= maxDocID {
			count++
		}
	}
	return count
}

// Clone returns an independent delete set.
func (s *DeleteStore) Clone() *DeleteStore {
	clone := NewDeleteStore()
	if s == nil {
		return clone
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for docID := range s.deleted {
		clone.deleted[docID] = struct{}{}
	}
	return clone
}

// WriteSnapshot writes sorted fixed-width document IDs atomically.
func (s *DeleteStore) WriteSnapshot(ctx context.Context, name string) error {
	if s == nil {
		return errors.New("db: nil delete store")
	}
	s.mu.RLock()
	ids := make([]uint64, 0, len(s.deleted))
	for docID := range s.deleted {
		ids = append(ids, docID)
	}
	s.mu.RUnlock()
	slices.Sort(ids)
	if len(ids) > maxSnapshotPayload/8 {
		return errors.New("db: delete snapshot is too large")
	}
	payload := make([]byte, len(ids)*8)
	for index, docID := range ids {
		binary.LittleEndian.PutUint64(payload[index*8:index*8+8], docID)
	}
	encoded, err := encodeSnapshot(deleteSnapshotMagic, uint64(len(ids)), payload)
	if err != nil {
		return err
	}
	return WriteImmutableSnapshot(ctx, name, encoded)
}

// LoadDeleteStore reads and validates a logical-deletion snapshot.
func LoadDeleteStore(ctx context.Context, name string) (*DeleteStore, error) {
	encoded, err := readSnapshotFile(ctx, name)
	if err != nil {
		return nil, err
	}
	count, payload, err := decodeSnapshot(encoded, deleteSnapshotMagic)
	if err != nil {
		return nil, err
	}
	if count > uint64(math.MaxInt/8) || count*8 != uint64(len(payload)) {
		return nil, fmt.Errorf("%w: delete count %d does not match payload", ErrSnapshotCorrupt, count)
	}
	result := NewDeleteStore()
	var previous uint64
	for index := uint64(0); index < count; index++ {
		docID := binary.LittleEndian.Uint64(payload[index*8 : index*8+8])
		if index > 0 && docID <= previous {
			return nil, fmt.Errorf("%w: deleted IDs are not strictly sorted", ErrSnapshotCorrupt)
		}
		result.deleted[docID] = struct{}{}
		previous = docID
	}
	return result, nil
}
