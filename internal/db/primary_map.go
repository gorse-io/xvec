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
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"unicode/utf8"
)

const maxPrimaryKeyBytes = 64 << 10

var primaryMapMagic = [8]byte{'Z', 'V', 'E', 'C', 'P', 'K', 0, 0}

// DocumentLocation identifies one document within a segment.
type DocumentLocation struct {
	SegmentID uint64
	DocID     uint64
}

// PrimaryKeyMap maps unique UTF-8 primary keys to global document IDs.
type PrimaryKeyMap struct {
	mu      sync.RWMutex
	entries map[string]DocumentLocation
}

// NewPrimaryKeyMap returns an empty primary-key map.
func NewPrimaryKeyMap() *PrimaryKeyMap {
	return &PrimaryKeyMap{entries: make(map[string]DocumentLocation)}
}

// Put adds or replaces key and returns its prior location, if any.
func (m *PrimaryKeyMap) Put(ctx context.Context, key string, location DocumentLocation) (DocumentLocation, bool, error) {
	if m == nil {
		return DocumentLocation{}, false, errors.New("db: nil primary-key map")
	}
	if ctx == nil {
		return DocumentLocation{}, false, errors.New("db: nil primary-key write context")
	}
	if err := ctx.Err(); err != nil {
		return DocumentLocation{}, false, err
	}
	if err := validatePrimaryKey(key); err != nil {
		return DocumentLocation{}, false, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for existingKey, existingLocation := range m.entries {
		if existingKey != key && existingLocation == location {
			return DocumentLocation{}, false, fmt.Errorf("db: document location is already mapped by primary key %q", existingKey)
		}
	}
	previous, replaced := m.entries[key]
	m.entries[key] = location
	return previous, replaced, nil
}

// Delete removes key and returns its prior location, if any.
func (m *PrimaryKeyMap) Delete(ctx context.Context, key string) (DocumentLocation, bool, error) {
	if m == nil {
		return DocumentLocation{}, false, errors.New("db: nil primary-key map")
	}
	if ctx == nil {
		return DocumentLocation{}, false, errors.New("db: nil primary-key delete context")
	}
	if err := ctx.Err(); err != nil {
		return DocumentLocation{}, false, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	previous, found := m.entries[key]
	if found {
		delete(m.entries, key)
	}
	return previous, found, nil
}

// Get returns the current location for key.
func (m *PrimaryKeyMap) Get(key string) (DocumentLocation, bool) {
	if m == nil {
		return DocumentLocation{}, false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	location, found := m.entries[key]
	return location, found
}

// MultiGet returns locations and found flags in input order.
func (m *PrimaryKeyMap) MultiGet(keys []string) ([]DocumentLocation, []bool) {
	locations := make([]DocumentLocation, len(keys))
	found := make([]bool, len(keys))
	if m == nil {
		return locations, found
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	for index, key := range keys {
		locations[index], found[index] = m.entries[key]
	}
	return locations, found
}

// Count returns the number of live primary keys.
func (m *PrimaryKeyMap) Count() int {
	if m == nil {
		return 0
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entries)
}

// Clone returns an independent in-memory map.
func (m *PrimaryKeyMap) Clone() *PrimaryKeyMap {
	clone := NewPrimaryKeyMap()
	if m == nil {
		return clone
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	for key, location := range m.entries {
		clone.entries[key] = location
	}
	return clone
}

// WriteSnapshot writes an immutable, deterministic primary-key snapshot.
func (m *PrimaryKeyMap) WriteSnapshot(ctx context.Context, name string) error {
	if m == nil {
		return errors.New("db: nil primary-key map")
	}
	m.mu.RLock()
	keys := make([]string, 0, len(m.entries))
	locations := make(map[string]DocumentLocation, len(m.entries))
	for key, location := range m.entries {
		keys = append(keys, key)
		locations[key] = location
	}
	m.mu.RUnlock()
	sort.Strings(keys)

	payloadSize := 0
	for _, key := range keys {
		if 20+len(key) > maxSnapshotPayload-payloadSize {
			return errors.New("db: primary-key snapshot is too large")
		}
		payloadSize += 20 + len(key)
	}
	payload := make([]byte, 0, payloadSize)
	var fixed [20]byte
	for _, key := range keys {
		binary.LittleEndian.PutUint32(fixed[:4], uint32(len(key)))
		binary.LittleEndian.PutUint64(fixed[4:12], locations[key].SegmentID)
		binary.LittleEndian.PutUint64(fixed[12:20], locations[key].DocID)
		payload = append(payload, fixed[:]...)
		payload = append(payload, key...)
	}
	encoded, err := encodeSnapshot(primaryMapMagic, uint64(len(keys)), payload)
	if err != nil {
		return err
	}
	return writeImmutableSnapshot(ctx, name, encoded)
}

// LoadPrimaryKeyMap reads and validates an immutable snapshot.
func LoadPrimaryKeyMap(ctx context.Context, name string) (*PrimaryKeyMap, error) {
	encoded, err := readSnapshotFile(ctx, name)
	if err != nil {
		return nil, err
	}
	count, payload, err := decodeSnapshot(encoded, primaryMapMagic)
	if err != nil {
		return nil, err
	}
	if count > uint64(len(payload)/20) || count > uint64(math.MaxInt) {
		return nil, fmt.Errorf("%w: impossible primary-key count %d", ErrSnapshotCorrupt, count)
	}
	result := NewPrimaryKeyMap()
	locations := make(map[DocumentLocation]string, int(count))
	offset := 0
	previousKey := ""
	for index := uint64(0); index < count; index++ {
		if len(payload)-offset < 20 {
			return nil, fmt.Errorf("%w: truncated primary-key entry %d", ErrSnapshotCorrupt, index)
		}
		keyLength := int(binary.LittleEndian.Uint32(payload[offset : offset+4]))
		location := DocumentLocation{
			SegmentID: binary.LittleEndian.Uint64(payload[offset+4 : offset+12]),
			DocID:     binary.LittleEndian.Uint64(payload[offset+12 : offset+20]),
		}
		offset += 20
		if keyLength == 0 || keyLength > maxPrimaryKeyBytes || keyLength > len(payload)-offset {
			return nil, fmt.Errorf("%w: invalid primary-key length %d", ErrSnapshotCorrupt, keyLength)
		}
		key := string(payload[offset : offset+keyLength])
		offset += keyLength
		if err := validatePrimaryKey(key); err != nil {
			return nil, fmt.Errorf("%w: entry %d: %v", ErrSnapshotCorrupt, index, err)
		}
		if index > 0 && key <= previousKey {
			return nil, fmt.Errorf("%w: primary keys are not strictly sorted", ErrSnapshotCorrupt)
		}
		if prior, exists := locations[location]; exists {
			return nil, fmt.Errorf("%w: keys %q and %q share a document location", ErrSnapshotCorrupt, prior, key)
		}
		result.entries[key] = location
		locations[location] = key
		previousKey = key
	}
	if offset != len(payload) {
		return nil, fmt.Errorf("%w: trailing primary-key payload", ErrSnapshotCorrupt)
	}
	return result, nil
}

func validatePrimaryKey(key string) error {
	if key == "" {
		return errors.New("db: primary key is empty")
	}
	if len(key) > maxPrimaryKeyBytes {
		return fmt.Errorf("db: primary key is %d bytes, maximum %d", len(key), maxPrimaryKeyBytes)
	}
	if !utf8.ValidString(key) {
		return errors.New("db: primary key is not valid UTF-8")
	}
	return nil
}
