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
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unicode/utf8"

	"github.com/cockroachdb/pebble/v2"
	"github.com/cockroachdb/pebble/v2/vfs"
)

const maxPrimaryKeyBytes = 64 << 10

var ErrIDMapCorrupt = errors.New("db: corrupt IDMap")

type primaryKeyDelta struct {
	docID   uint64
	deleted bool
}

// PrimaryKeyMap is the collection IDMap: a Pebble point map from a primary key
// directly to a collection-global document ID. Writable maps are disposable
// working state whose durability comes from the outer WAL. Read-only maps use
// an immutable Pebble checkpoint plus an in-memory replay overlay.
type PrimaryKeyMap struct {
	mu      sync.RWMutex
	db      *pebble.DB
	count   int
	overlay map[string]primaryKeyDelta

	// Tests use these hooks to exercise fail-stop behavior after an outer-WAL
	// append. Production maps leave them nil.
	setPoint    func(key, value []byte) error
	deletePoint func(key []byte) error
}

// NewPrimaryKeyMap creates an in-memory writable IDMap for unit-level users.
// Collection lifecycle paths use the filesystem constructors below.
func NewPrimaryKeyMap() *PrimaryKeyMap {
	database, err := pebble.Open("idmap", &pebble.Options{
		FS:                 vfs.NewMem(),
		DisableWAL:         true,
		FormatMajorVersion: pebble.FormatNewest,
	})
	if err != nil {
		panic(fmt.Sprintf("db: create in-memory IDMap: %v", err))
	}
	return &PrimaryKeyMap{db: database}
}

// CreatePrimaryKeyMap creates a disposable writable Pebble IDMap at path.
func CreatePrimaryKeyMap(ctx context.Context, name string) (*PrimaryKeyMap, error) {
	if ctx == nil {
		return nil, errors.New("db: nil IDMap create context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if name == "" {
		return nil, errors.New("db: empty IDMap path")
	}
	if err := ensureDirectorySynced(filepath.Dir(name)); err != nil {
		return nil, fmt.Errorf("db: create IDMap parent: %w", err)
	}
	if err := validateIDMapParentDirectory(name); err != nil {
		return nil, err
	}
	if _, err := os.Lstat(name); err == nil {
		return nil, os.ErrExist
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("db: inspect IDMap path: %w", err)
	}
	database, err := pebble.Open(name, writableIDMapOptions(true))
	if err != nil {
		return nil, fmt.Errorf("db: create IDMap: %w", err)
	}
	if err := syncDirectory(filepath.Dir(name)); err != nil {
		closeErr := database.Close()
		return nil, errors.Join(fmt.Errorf("db: sync IDMap parent: %w", err), closeErr)
	}
	return &PrimaryKeyMap{db: database}, nil
}

// OpenPrimaryKeyMap copies checkpoint into a new disposable working directory
// and opens the copy for replay and mutations. Existing working directories are
// never reused, so the outer WAL remains the sole recovery authority.
func OpenPrimaryKeyMap(ctx context.Context, checkpoint, working string) (*PrimaryKeyMap, error) {
	if ctx == nil {
		return nil, errors.New("db: nil IDMap open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if checkpoint == "" || working == "" {
		return nil, errors.New("db: empty IDMap checkpoint or working path")
	}
	if err := validateIDMapDirectory(checkpoint); err != nil {
		return nil, err
	}
	if err := ensureDirectorySynced(filepath.Dir(working)); err != nil {
		return nil, fmt.Errorf("db: create IDMap working parent: %w", err)
	}
	if err := validateIDMapParentDirectory(working); err != nil {
		return nil, err
	}
	if _, err := os.Lstat(working); err == nil {
		return nil, os.ErrExist
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("db: inspect IDMap working path: %w", err)
	}

	cloned, err := vfs.Clone(vfs.Default, vfs.Default, checkpoint, working, vfs.CloneSync)
	if err != nil || !cloned {
		_ = os.RemoveAll(working)
		if err == nil {
			err = os.ErrNotExist
		}
		return nil, fmt.Errorf("db: copy IDMap checkpoint: %w", err)
	}

	database, err := pebble.Open(working, writableIDMapOptions(false))
	if err != nil {
		_ = os.RemoveAll(working)
		return nil, fmt.Errorf("db: open IDMap working copy: %w", err)
	}
	result := &PrimaryKeyMap{db: database}
	result.count, err = result.scanCount(ctx)
	if err != nil {
		closeErr := database.Close()
		_ = os.RemoveAll(working)
		return nil, errors.Join(err, closeErr)
	}
	return result, nil
}

// OpenPrimaryKeyMapReadOnly opens checkpoint without creating files and keeps
// WAL replay changes in memory.
func OpenPrimaryKeyMapReadOnly(ctx context.Context, checkpoint string) (*PrimaryKeyMap, error) {
	if ctx == nil {
		return nil, errors.New("db: nil read-only IDMap open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if checkpoint == "" {
		return nil, errors.New("db: empty IDMap checkpoint path")
	}
	if err := validateIDMapDirectory(checkpoint); err != nil {
		return nil, err
	}
	memoryFS := vfs.NewMem()
	const memoryPath = "idmap"
	cloned, err := vfs.Clone(vfs.Default, memoryFS, checkpoint, memoryPath)
	if err != nil || !cloned {
		if err == nil {
			err = os.ErrNotExist
		}
		return nil, fmt.Errorf("db: copy read-only IDMap checkpoint: %w", err)
	}
	options := readOnlyIDMapOptions()
	options.FS = memoryFS
	database, err := pebble.Open(memoryPath, options)
	if err != nil {
		return nil, fmt.Errorf("db: open read-only IDMap: %w", err)
	}
	result := &PrimaryKeyMap{db: database, overlay: make(map[string]primaryKeyDelta)}
	result.count, err = result.scanCount(ctx)
	if err != nil {
		return nil, errors.Join(err, database.Close())
	}
	return result, nil
}

func writableIDMapOptions(errorIfExists bool) *pebble.Options {
	return &pebble.Options{
		DisableWAL:         true,
		ErrorIfExists:      errorIfExists,
		FormatMajorVersion: pebble.FormatNewest,
	}
}

func readOnlyIDMapOptions() *pebble.Options {
	return &pebble.Options{
		ReadOnly:           true,
		ErrorIfNotExists:   true,
		FormatMajorVersion: pebble.FormatNewest,
	}
}

// Put adds or replaces key and returns its prior global document ID, if any.
func (m *PrimaryKeyMap) Put(ctx context.Context, key string, docID uint64) (uint64, bool, error) {
	if m == nil {
		return 0, false, errors.New("db: nil IDMap")
	}
	if ctx == nil {
		return 0, false, errors.New("db: nil IDMap write context")
	}
	if err := ctx.Err(); err != nil {
		return 0, false, err
	}
	if err := validatePrimaryKey(key); err != nil {
		return 0, false, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.db == nil {
		return 0, false, errors.New("db: closed IDMap")
	}
	previous, found, err := m.getLocked(key)
	if err != nil {
		return 0, false, err
	}
	if m.overlay != nil {
		m.overlay[key] = primaryKeyDelta{docID: docID}
	} else {
		var encoded [8]byte
		binary.BigEndian.PutUint64(encoded[:], docID)
		if m.setPoint != nil {
			err = m.setPoint([]byte(key), encoded[:])
		} else {
			err = m.db.Set([]byte(key), encoded[:], pebble.NoSync)
		}
		if err != nil {
			return 0, false, fmt.Errorf("db: write IDMap point: %w", err)
		}
	}
	if !found {
		m.count++
	}
	return previous, found, nil
}

// Delete removes key and returns its prior global document ID, if any.
func (m *PrimaryKeyMap) Delete(ctx context.Context, key string) (uint64, bool, error) {
	if m == nil {
		return 0, false, errors.New("db: nil IDMap")
	}
	if ctx == nil {
		return 0, false, errors.New("db: nil IDMap delete context")
	}
	if err := ctx.Err(); err != nil {
		return 0, false, err
	}
	if err := validatePrimaryKey(key); err != nil {
		return 0, false, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.db == nil {
		return 0, false, errors.New("db: closed IDMap")
	}
	previous, found, err := m.getLocked(key)
	if err != nil || !found {
		return previous, found, err
	}
	if m.overlay != nil {
		m.overlay[key] = primaryKeyDelta{deleted: true}
	} else {
		if m.deletePoint != nil {
			err = m.deletePoint([]byte(key))
		} else {
			err = m.db.Delete([]byte(key), pebble.NoSync)
		}
		if err != nil {
			return 0, false, fmt.Errorf("db: delete IDMap point: %w", err)
		}
	}
	m.count--
	return previous, true, nil
}

// Get performs a point lookup and never converts a Pebble error into absence.
func (m *PrimaryKeyMap) Get(key string) (uint64, bool, error) {
	if m == nil {
		return 0, false, errors.New("db: nil IDMap")
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.db == nil {
		return 0, false, errors.New("db: closed IDMap")
	}
	return m.getLocked(key)
}

func (m *PrimaryKeyMap) getLocked(key string) (uint64, bool, error) {
	if delta, found := m.overlay[key]; found {
		return delta.docID, !delta.deleted, nil
	}
	return getPrimaryDocID(m.db, key)
}

// MultiGet returns document IDs and found flags in input order.
func (m *PrimaryKeyMap) MultiGet(keys []string) ([]uint64, []bool, error) {
	docIDs := make([]uint64, len(keys))
	found := make([]bool, len(keys))
	for index, key := range keys {
		var err error
		docIDs[index], found[index], err = m.Get(key)
		if err != nil {
			return nil, nil, err
		}
	}
	return docIDs, found, nil
}

// Count returns the exact logical key count maintained across mutations and a
// read-only overlay. Corrupt checkpoint values are rejected while opening.
func (m *PrimaryKeyMap) Count() int {
	if m == nil {
		return 0
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.count
}

// Checkpoint flushes disposable working state and creates an immutable Pebble
// directory. The checkpoint is not visible until an outer manifest naming it
// is published through CURRENT.
func (m *PrimaryKeyMap) Checkpoint(ctx context.Context, target string) error {
	if m == nil {
		return errors.New("db: nil IDMap")
	}
	if ctx == nil {
		return errors.New("db: nil IDMap checkpoint context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if target == "" {
		return errors.New("db: empty IDMap checkpoint path")
	}
	parent := filepath.Dir(target)
	if err := ensureDirectorySynced(parent); err != nil {
		return fmt.Errorf("db: create IDMap checkpoint parent: %w", err)
	}
	if err := validateIDMapParentDirectory(target); err != nil {
		return err
	}
	if _, err := os.Lstat(target); err == nil {
		return os.ErrExist
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("db: inspect IDMap checkpoint: %w", err)
	}

	temp, err := os.MkdirTemp(parent, ".checkpoint-*.tmp")
	if err != nil {
		return fmt.Errorf("db: reserve IDMap checkpoint path: %w", err)
	}
	if err := os.Remove(temp); err != nil {
		return fmt.Errorf("db: prepare IDMap checkpoint path: %w", err)
	}
	defer func() { _ = os.RemoveAll(temp) }()

	m.mu.Lock()
	if m.db == nil {
		m.mu.Unlock()
		return errors.New("db: closed IDMap")
	}
	if m.overlay != nil {
		m.mu.Unlock()
		return ErrReadOnly
	}
	if err := m.db.Flush(); err != nil {
		m.mu.Unlock()
		return fmt.Errorf("db: flush IDMap working state: %w", err)
	}
	err = m.db.Checkpoint(temp)
	m.mu.Unlock()
	if err != nil {
		return fmt.Errorf("db: checkpoint IDMap: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := os.Rename(temp, target); err != nil {
		return fmt.Errorf("db: install IDMap checkpoint: %w", err)
	}
	if err := syncDirectory(parent); err != nil {
		return fmt.Errorf("db: sync IDMap checkpoint parent: %w", err)
	}
	return nil
}

// Close releases Pebble resources. It is idempotent.
func (m *PrimaryKeyMap) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.db == nil {
		return nil
	}
	err := m.db.Close()
	m.db = nil
	m.overlay = nil
	return err
}

func (m *PrimaryKeyMap) scanCount(ctx context.Context) (count int, err error) {
	iterator, err := m.db.NewIter(nil)
	if err != nil {
		return 0, fmt.Errorf("db: scan IDMap: %w", err)
	}
	defer func() { err = errors.Join(err, iterator.Close()) }()
	for valid := iterator.First(); valid; valid = iterator.Next() {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		if err := validatePrimaryKey(string(iterator.Key())); err != nil {
			return 0, fmt.Errorf("%w: invalid primary key: %v", ErrIDMapCorrupt, err)
		}
		if len(iterator.Value()) != 8 {
			return 0, fmt.Errorf("%w: document ID value has %d bytes", ErrIDMapCorrupt, len(iterator.Value()))
		}
		count++
	}
	if err := iterator.Error(); err != nil {
		return 0, fmt.Errorf("db: scan IDMap: %w", err)
	}
	return count, nil
}

func (m *PrimaryKeyMap) forEach(ctx context.Context, visit func(string, uint64) error) (err error) {
	if m == nil {
		return errors.New("db: nil IDMap")
	}
	if ctx == nil {
		return errors.New("db: nil IDMap iteration context")
	}
	if visit == nil {
		return errors.New("db: nil IDMap visitor")
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.db == nil {
		return errors.New("db: closed IDMap")
	}
	iterator, err := m.db.NewIter(nil)
	if err != nil {
		return fmt.Errorf("db: iterate IDMap: %w", err)
	}
	defer func() { err = errors.Join(err, iterator.Close()) }()
	seenOverlay := make(map[string]struct{}, len(m.overlay))
	for valid := iterator.First(); valid; valid = iterator.Next() {
		if err := ctx.Err(); err != nil {
			return err
		}
		key := string(iterator.Key())
		if err := validatePrimaryKey(key); err != nil {
			return fmt.Errorf("%w: invalid primary key: %v", ErrIDMapCorrupt, err)
		}
		value := iterator.Value()
		if len(value) != 8 {
			return fmt.Errorf("%w: document ID value has %d bytes", ErrIDMapCorrupt, len(value))
		}
		docID := binary.BigEndian.Uint64(value)
		if delta, found := m.overlay[key]; found {
			seenOverlay[key] = struct{}{}
			if delta.deleted {
				continue
			}
			docID = delta.docID
		}
		if err := visit(key, docID); err != nil {
			return err
		}
	}
	if err := iterator.Error(); err != nil {
		return fmt.Errorf("db: iterate IDMap: %w", err)
	}
	for key, delta := range m.overlay {
		if _, found := seenOverlay[key]; found || delta.deleted {
			continue
		}
		if err := visit(key, delta.docID); err != nil {
			return err
		}
	}
	return nil
}

func getPrimaryDocID(database *pebble.DB, key string) (docID uint64, found bool, err error) {
	value, closer, err := database.Get([]byte(key))
	if errors.Is(err, pebble.ErrNotFound) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, fmt.Errorf("db: read IDMap point: %w", err)
	}
	defer func() { err = errors.Join(err, closer.Close()) }()
	if len(value) != 8 {
		return 0, false, fmt.Errorf("%w: document ID value has %d bytes", ErrIDMapCorrupt, len(value))
	}
	return binary.BigEndian.Uint64(value), true, nil
}

func validateIDMapDirectory(name string) error {
	if err := validateIDMapParentDirectory(name); err != nil {
		return err
	}
	info, err := os.Lstat(name)
	if err != nil {
		return fmt.Errorf("db: inspect IDMap directory: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return errors.New("db: IDMap directory is a symlink")
	}
	if !info.IsDir() {
		return errors.New("db: IDMap path is not a directory")
	}
	return filepath.WalkDir(name, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("db: IDMap contains symlink %q", path)
		}
		if !entry.IsDir() && !entry.Type().IsRegular() {
			return fmt.Errorf("db: IDMap contains non-regular entry %q", path)
		}
		return nil
	})
}

func validateIDMapParentDirectory(name string) error {
	parent := filepath.Dir(name)
	info, err := os.Lstat(parent)
	if err != nil {
		return fmt.Errorf("db: inspect IDMap parent directory: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return errors.New("db: IDMap parent directory is a symlink")
	}
	if !info.IsDir() {
		return errors.New("db: IDMap parent path is not a directory")
	}
	return nil
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
