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
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	collectionLockName         = ".collection.lock"
	DefaultSegmentMaxDocuments = uint64(65_536)
)

var (
	ErrCollectionClosed  = errors.New("db: collection is closed")
	ErrCollectionCorrupt = errors.New("db: corrupt collection")
	ErrReadOnly          = errors.New("db: collection is read-only")
)

// CollectionOptions controls the native storage lifecycle. SegmentMaxDocuments
// is persisted at creation; zero selects DefaultSegmentMaxDocuments. ReadOnly
// is an open-handle property and is never persisted.
type CollectionOptions struct {
	ReadOnly            bool
	EnableMmap          bool
	SegmentMaxDocuments uint64
	WAL                 WALOptions
}

// CollectionStore owns one consistent manifest, WAL, and segment view. A
// writable handle holds the exclusive collection lock; any number of read-only
// handles can hold the shared lock together.
type CollectionStore struct {
	mu       sync.RWMutex
	dir      string
	readOnly bool
	closed   bool
	lock     *ailego.FileLock
	versions *VersionManager
	manager  *SegmentManager
	engine   *WriteEngine
	wal      *WAL
}

// CreateCollection creates a native Go collection and returns its sole writer.
func CreateCollection(ctx context.Context, dir string, schema json.RawMessage, options CollectionOptions) (*CollectionStore, error) {
	if ctx == nil {
		return nil, errors.New("db: nil create collection context")
	}
	if dir == "" {
		return nil, errors.New("db: empty collection directory")
	}
	if options.ReadOnly {
		return nil, ErrReadOnly
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	capacity := options.SegmentMaxDocuments
	if capacity == 0 {
		capacity = DefaultSegmentMaxDocuments
	}
	if err := validateSchemaJSON(schema); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("db: create collection directory: %w", err)
	}
	lock, err := ailego.AcquireFileLock(ctx, filepath.Join(dir, collectionLockName), ailego.LockExclusive)
	if err != nil {
		return nil, fmt.Errorf("db: lock collection creation: %w", err)
	}
	fail := func(err error, wal *WAL, created ...string) (*CollectionStore, error) {
		if wal != nil {
			_ = wal.Close()
		}
		for _, name := range created {
			_ = os.Remove(name)
		}
		_ = lock.Close()
		return nil, err
	}
	if _, err := OpenVersionManager(ctx, dir); err == nil {
		return fail(ErrManifestExists, nil)
	} else if !errors.Is(err, ErrManifestNotFound) {
		return fail(err, nil)
	}

	walRelative := walFileName(0, 1)
	walPath := collectionPath(dir, walRelative)
	if err := ensureDirectorySynced(filepath.Dir(walPath)); err != nil {
		return fail(fmt.Errorf("db: create WAL directory: %w", err), nil)
	}
	wal, err := CreateWAL(ctx, walPath, options.WAL)
	if err != nil {
		return fail(err, nil)
	}
	primaryPath := collectionPath(dir, primarySnapshotName(1))
	deletesPath := collectionPath(dir, deleteSnapshotName(1))
	primary := NewPrimaryKeyMap()
	deletes := NewDeleteStore()
	if err := primary.WriteSnapshot(ctx, primaryPath); err != nil {
		return fail(err, wal, walPath)
	}
	if err := deletes.WriteSnapshot(ctx, deletesPath); err != nil {
		return fail(err, wal, walPath, primaryPath)
	}
	writing, err := NewWriteSegment(0, 0, capacity)
	if err != nil {
		return fail(err, wal, walPath, primaryPath, deletesPath)
	}
	manager := NewSegmentManager(primary, deletes)
	if err := manager.SetWriting(writing); err != nil {
		return fail(err, wal, walPath, primaryPath, deletesPath)
	}
	manifest := Manifest{
		FormatVersion: DiskFormatVersion, Schema: slices.Clone(schema),
		EnableMmap: options.EnableMmap, SegmentMaxDocuments: capacity,
		WritingSegment:  &SegmentMetadata{ID: 0, Files: []string{walRelative}},
		IDMapGeneration: 1, DeleteSnapshotGeneration: 1, NextSegmentID: 1,
	}
	versions, err := CreateVersionManager(ctx, dir, manifest)
	if err != nil {
		return fail(err, wal, walPath, primaryPath, deletesPath)
	}
	engine, err := NewWriteEngine(manager, wal)
	if err != nil {
		return fail(err, wal)
	}
	return &CollectionStore{
		dir: dir, lock: lock, versions: versions, manager: manager,
		engine: engine, wal: wal,
	}, nil
}

// OpenCollection opens the exact version named by CURRENT and replays the
// complete WAL prefix. Read-only recovery never modifies an incomplete tail.
func OpenCollection(ctx context.Context, dir string, options CollectionOptions) (*CollectionStore, error) {
	if ctx == nil {
		return nil, errors.New("db: nil open collection context")
	}
	if dir == "" {
		return nil, errors.New("db: empty collection directory")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := os.Stat(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrManifestNotFound
		}
		return nil, fmt.Errorf("db: stat collection directory: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("db: collection path %q is not a directory", dir)
	}
	mode := ailego.LockExclusive
	if options.ReadOnly {
		mode = ailego.LockShared
	}
	lock, err := ailego.AcquireFileLock(ctx, filepath.Join(dir, collectionLockName), mode)
	if err != nil {
		return nil, fmt.Errorf("db: lock collection open: %w", err)
	}
	fail := func(err error, wal *WAL) (*CollectionStore, error) {
		if wal != nil {
			_ = wal.Close()
		}
		_ = lock.Close()
		return nil, err
	}
	versions, err := OpenVersionManager(ctx, dir)
	if err != nil {
		return fail(err, nil)
	}
	manifest := versions.Current()
	if err := validateLifecycleManifest(manifest); err != nil {
		return fail(err, nil)
	}
	primary, err := LoadPrimaryKeyMap(ctx, collectionPath(dir, primarySnapshotName(manifest.IDMapGeneration)))
	if err != nil {
		return fail(fmt.Errorf("%w: load primary-key snapshot: %v", ErrCollectionCorrupt, err), nil)
	}
	deletes, err := LoadDeleteStore(ctx, collectionPath(dir, deleteSnapshotName(manifest.DeleteSnapshotGeneration)))
	if err != nil {
		return fail(fmt.Errorf("%w: load delete snapshot: %v", ErrCollectionCorrupt, err), nil)
	}
	manager := NewSegmentManager(primary, deletes)
	nextDocID := uint64(0)
	for _, metadata := range manifest.PersistedSegments {
		segment, err := OpenImmutableSegment(ctx, dir, metadata)
		if err != nil {
			return fail(fmt.Errorf("%w: open segment %d: %v", ErrCollectionCorrupt, metadata.ID, err), nil)
		}
		if err := manager.AddImmutable(segment); err != nil {
			return fail(fmt.Errorf("%w: add segment %d: %v", ErrCollectionCorrupt, metadata.ID, err), nil)
		}
		if metadata.DocCount > 0 {
			if metadata.MaxDocID == math.MaxUint64 {
				return fail(fmt.Errorf("%w: document ID space is exhausted", ErrCollectionCorrupt), nil)
			}
			nextDocID = max(nextDocID, metadata.MaxDocID+1)
		}
	}
	writing, err := NewWriteSegment(manifest.WritingSegment.ID, nextDocID, manifest.SegmentMaxDocuments)
	if err != nil {
		return fail(fmt.Errorf("%w: create writing segment: %v", ErrCollectionCorrupt, err), nil)
	}
	if err := manager.SetWriting(writing); err != nil {
		return fail(fmt.Errorf("%w: install writing segment: %v", ErrCollectionCorrupt, err), nil)
	}
	walPath := collectionPath(dir, manifest.WritingSegment.Files[0])
	var wal *WAL
	if options.ReadOnly {
		wal, err = OpenWALReadOnly(ctx, walPath)
	} else {
		wal, err = OpenWAL(ctx, walPath, options.WAL)
	}
	if err != nil {
		return fail(fmt.Errorf("%w: open writing WAL: %v", ErrCollectionCorrupt, err), nil)
	}
	if err := replayWriteWAL(ctx, wal, manager); err != nil {
		return fail(err, wal)
	}
	if err := validateCollectionState(manager); err != nil {
		return fail(err, wal)
	}
	store := &CollectionStore{
		dir: dir, readOnly: options.ReadOnly, lock: lock, versions: versions,
		manager: manager, wal: wal,
	}
	if !options.ReadOnly {
		store.engine, err = NewWriteEngine(manager, wal)
		if err != nil {
			return fail(err, wal)
		}
	}
	return store, nil
}

// Insert delegates a durable batch insert to the current WAL writer.
func (c *CollectionStore) Insert(ctx context.Context, inputs []WriteInput) ([]WriteResult, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireWritableLocked(); err != nil {
		return nil, err
	}
	return c.engine.Insert(ctx, inputs)
}

// Upsert delegates a durable batch upsert to the current WAL writer.
func (c *CollectionStore) Upsert(ctx context.Context, inputs []WriteInput) ([]WriteResult, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireWritableLocked(); err != nil {
		return nil, err
	}
	return c.engine.Upsert(ctx, inputs)
}

// Update delegates a durable batch update to the current WAL writer.
func (c *CollectionStore) Update(ctx context.Context, inputs []WriteInput) ([]WriteResult, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireWritableLocked(); err != nil {
		return nil, err
	}
	return c.engine.Update(ctx, inputs)
}

// Delete delegates a durable primary-key batch delete to the current writer.
func (c *CollectionStore) Delete(ctx context.Context, primaryKeys []string) ([]WriteResult, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireWritableLocked(); err != nil {
		return nil, err
	}
	return c.engine.Delete(ctx, primaryKeys)
}

// Fetch resolves primary keys against the stable in-memory version.
func (c *CollectionStore) Fetch(ctx context.Context, primaryKeys []string) ([]FetchResult, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, ErrCollectionClosed
	}
	return c.manager.Fetch(ctx, primaryKeys)
}

// Flush atomically turns the non-empty write segment into an immutable segment,
// snapshots key/deletion state, publishes a new manifest, and rotates the WAL.
func (c *CollectionStore) Flush(ctx context.Context) error {
	if c == nil {
		return errors.New("db: nil collection")
	}
	if ctx == nil {
		return errors.New("db: nil flush context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireWritableLocked(); err != nil {
		return err
	}
	if err := c.wal.Sync(ctx); err != nil {
		return err
	}
	writing := c.manager.Writing()
	documents := writing.Documents()
	if len(documents) == 0 {
		return nil
	}
	current := c.versions.Current()
	if current.NextSegmentID == math.MaxUint64 {
		return errors.New("db: segment ID space is exhausted")
	}
	lastDocID := documents[len(documents)-1].DocID
	if lastDocID == math.MaxUint64 {
		return errors.New("db: document ID space is exhausted")
	}
	nextWriting, err := NewWriteSegment(current.NextSegmentID, lastDocID+1, current.SegmentMaxDocuments)
	if err != nil {
		return err
	}

	if current.Generation == math.MaxUint64 {
		return errors.New("db: manifest generation space is exhausted")
	}
	artifactGeneration := current.Generation + 1
	segmentRelative, err := c.availableArtifact(func(generation uint64) string {
		return segmentFileName(writing.ID(), generation)
	}, artifactGeneration)
	if err != nil {
		return err
	}
	immutable, err := writing.Snapshot(ctx, c.dir, segmentRelative)
	if err != nil {
		return fmt.Errorf("db: snapshot writing segment: %w", err)
	}
	created := []string{collectionPath(c.dir, segmentRelative)}
	cleanup := func() {
		for _, name := range created {
			_ = os.Remove(name)
		}
	}

	previousSnapshotGeneration := max(current.IDMapGeneration, current.DeleteSnapshotGeneration)
	if previousSnapshotGeneration == math.MaxUint64 {
		cleanup()
		return errors.New("db: snapshot generation space is exhausted")
	}
	snapshotGeneration := previousSnapshotGeneration + 1
	for c.snapshotGenerationExists(snapshotGeneration) {
		snapshotGeneration++
		if snapshotGeneration == 0 {
			cleanup()
			return errors.New("db: snapshot generation space is exhausted")
		}
	}
	primaryPath := collectionPath(c.dir, primarySnapshotName(snapshotGeneration))
	if err := c.manager.PrimaryKeys().WriteSnapshot(ctx, primaryPath); err != nil {
		cleanup()
		return fmt.Errorf("db: write primary-key snapshot: %w", err)
	}
	created = append(created, primaryPath)
	deletesPath := collectionPath(c.dir, deleteSnapshotName(snapshotGeneration))
	if err := c.manager.Deletes().WriteSnapshot(ctx, deletesPath); err != nil {
		cleanup()
		return fmt.Errorf("db: write delete snapshot: %w", err)
	}
	created = append(created, deletesPath)

	walRelative, err := c.availableArtifact(func(generation uint64) string {
		return walFileName(current.NextSegmentID, generation)
	}, artifactGeneration)
	if err != nil {
		cleanup()
		return err
	}
	walPath := collectionPath(c.dir, walRelative)
	if err := ensureDirectorySynced(filepath.Dir(walPath)); err != nil {
		cleanup()
		return fmt.Errorf("db: create next WAL directory: %w", err)
	}
	nextWAL, err := CreateWAL(ctx, walPath, c.wal.options)
	if err != nil {
		cleanup()
		return fmt.Errorf("db: create next WAL: %w", err)
	}
	created = append(created, walPath)

	nextManifest := current.Clone()
	nextManifest.PersistedSegments = append(nextManifest.PersistedSegments, immutable.Metadata())
	nextManifest.WritingSegment = &SegmentMetadata{ID: current.NextSegmentID, Files: []string{walRelative}}
	nextManifest.IDMapGeneration = snapshotGeneration
	nextManifest.DeleteSnapshotGeneration = snapshotGeneration
	nextManifest.NextSegmentID++
	published, publishErr := c.versions.Publish(ctx, nextManifest)
	committed := c.versions.Current().Generation != current.Generation
	if !committed {
		_ = nextWAL.Close()
		cleanup()
		return publishErr
	}
	if err := c.manager.RotateWriting(writing.ID(), immutable, nextWriting); err != nil {
		_ = nextWAL.Close()
		return errors.Join(publishErr, fmt.Errorf("db: apply committed segment rotation at generation %d: %w", published.Generation, err))
	}
	oldWAL := c.wal
	c.wal = nextWAL
	c.engine, err = NewWriteEngine(c.manager, nextWAL)
	if err != nil {
		return errors.Join(publishErr, err, oldWAL.Close())
	}
	return errors.Join(publishErr, oldWAL.Close())
}

// Manifest returns an independent copy of the current published metadata.
func (c *CollectionStore) Manifest() Manifest {
	if c == nil {
		return Manifest{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.versions.Current()
}

// ReadOnly reports whether this handle rejects mutations.
func (c *CollectionStore) ReadOnly() bool {
	if c == nil {
		return false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.readOnly
}

// Close releases the WAL and collection lock. WAL-backed writes are already
// durable and will be replayed even when Flush was not called.
func (c *CollectionStore) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return errors.Join(c.wal.Close(), c.lock.Close())
}

func (c *CollectionStore) requireWritableLocked() error {
	if c == nil {
		return errors.New("db: nil collection")
	}
	if c.closed {
		return ErrCollectionClosed
	}
	if c.readOnly {
		return ErrReadOnly
	}
	return nil
}

func replayWriteWAL(ctx context.Context, wal *WAL, manager *SegmentManager) error {
	return wal.Replay(ctx, func(record WALRecord) error {
		operation, err := decodeWriteOperation(record.Payload)
		if err != nil {
			return fmt.Errorf("%w: decode operation: %v", ErrWALCorrupt, err)
		}
		if err := applyRecoveredOperation(ctx, manager, operation); err != nil {
			return fmt.Errorf("%w: apply operation: %v", ErrWALCorrupt, err)
		}
		return nil
	})
}

func applyRecoveredOperation(ctx context.Context, manager *SegmentManager, operation writeOperation) error {
	writing := manager.Writing()
	if operation.Type != writeOperationDelete {
		if writing == nil || operation.SegmentID != writing.ID() {
			return fmt.Errorf("operation targets writing segment %d, current is %d", operation.SegmentID, writing.ID())
		}
	}
	switch operation.Type {
	case writeOperationInsert:
		if _, exists := manager.PrimaryKeys().Get(operation.PrimaryKey); exists {
			return ErrPrimaryKeyExists
		}
		if _, err := writing.AppendExpected(ctx, operation.DocID, operation.PrimaryKey, operation.Payload); err != nil {
			return err
		}
		_, _, err := manager.PrimaryKeys().Put(ctx, operation.PrimaryKey, DocumentLocation{SegmentID: operation.SegmentID, DocID: operation.DocID})
		return err
	case writeOperationUpsert, writeOperationUpdate:
		previous, existed := manager.PrimaryKeys().Get(operation.PrimaryKey)
		if operation.Type == writeOperationUpdate && !existed {
			return ErrPrimaryKeyNotFound
		}
		if _, err := writing.AppendExpected(ctx, operation.DocID, operation.PrimaryKey, operation.Payload); err != nil {
			return err
		}
		if existed {
			if _, err := manager.Deletes().MarkDeleted(ctx, previous.DocID); err != nil {
				return err
			}
		}
		_, _, err := manager.PrimaryKeys().Put(ctx, operation.PrimaryKey, DocumentLocation{SegmentID: operation.SegmentID, DocID: operation.DocID})
		return err
	case writeOperationDelete:
		if len(operation.Payload) != 0 {
			return errors.New("delete operation contains a payload")
		}
		location, existed := manager.PrimaryKeys().Get(operation.PrimaryKey)
		if !existed || location != (DocumentLocation{SegmentID: operation.SegmentID, DocID: operation.DocID}) {
			return ErrPrimaryKeyNotFound
		}
		if _, err := manager.Deletes().MarkDeleted(ctx, operation.DocID); err != nil {
			return err
		}
		_, found, err := manager.PrimaryKeys().Delete(ctx, operation.PrimaryKey)
		if err != nil || !found {
			return errors.Join(err, ErrPrimaryKeyNotFound)
		}
		return nil
	default:
		return errors.New("unknown write operation")
	}
}

func validateCollectionState(manager *SegmentManager) error {
	live := make(map[DocumentLocation]string)
	known := make(map[uint64]struct{})
	segments := manager.ImmutableSegments()
	for _, segment := range segments {
		for _, document := range segment.Documents() {
			known[document.DocID] = struct{}{}
			if !manager.Deletes().IsDeleted(document.DocID) {
				live[DocumentLocation{SegmentID: segment.ID(), DocID: document.DocID}] = document.PrimaryKey
			}
		}
	}
	if writing := manager.Writing(); writing != nil {
		for _, document := range writing.Documents() {
			known[document.DocID] = struct{}{}
			if !manager.Deletes().IsDeleted(document.DocID) {
				live[DocumentLocation{SegmentID: writing.ID(), DocID: document.DocID}] = document.PrimaryKey
			}
		}
	}
	manager.deletes.mu.RLock()
	for docID := range manager.deletes.deleted {
		if _, exists := known[docID]; !exists {
			manager.deletes.mu.RUnlock()
			return fmt.Errorf("%w: deletion references missing document %d", ErrCollectionCorrupt, docID)
		}
	}
	manager.deletes.mu.RUnlock()
	manager.primaryKey.mu.RLock()
	defer manager.primaryKey.mu.RUnlock()
	if len(manager.primaryKey.entries) != len(live) {
		return fmt.Errorf("%w: primary-key count %d differs from live document count %d", ErrCollectionCorrupt, len(manager.primaryKey.entries), len(live))
	}
	for key, location := range manager.primaryKey.entries {
		if liveKey, exists := live[location]; !exists || liveKey != key {
			return fmt.Errorf("%w: primary key %q has invalid location", ErrCollectionCorrupt, key)
		}
	}
	return nil
}

func validateLifecycleManifest(manifest Manifest) error {
	if manifest.SegmentMaxDocuments == 0 || manifest.IDMapGeneration == 0 || manifest.DeleteSnapshotGeneration == 0 {
		return fmt.Errorf("%w: invalid lifecycle generations or capacity", ErrCollectionCorrupt)
	}
	if manifest.WritingSegment == nil || manifest.WritingSegment.DocCount != 0 || len(manifest.WritingSegment.Files) != 1 {
		return fmt.Errorf("%w: invalid writing segment metadata", ErrCollectionCorrupt)
	}
	return nil
}

func validateSchemaJSON(schema json.RawMessage) error {
	if !json.Valid(schema) {
		return fmt.Errorf("%w: schema is not valid JSON", ErrManifestCorrupt)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(schema, &object); err != nil || object == nil {
		return fmt.Errorf("%w: schema must be a JSON object", ErrManifestCorrupt)
	}
	return nil
}

func (c *CollectionStore) availableArtifact(name func(uint64) string, generation uint64) (string, error) {
	for {
		relative := name(generation)
		if _, err := os.Stat(collectionPath(c.dir, relative)); errors.Is(err, os.ErrNotExist) {
			return relative, nil
		} else if err != nil {
			return "", fmt.Errorf("db: inspect artifact %q: %w", relative, err)
		}
		if generation == math.MaxUint64 {
			return "", errors.New("db: artifact generation space is exhausted")
		}
		generation++
	}
}

func (c *CollectionStore) snapshotGenerationExists(generation uint64) bool {
	for _, relative := range []string{primarySnapshotName(generation), deleteSnapshotName(generation)} {
		if _, err := os.Stat(collectionPath(c.dir, relative)); err == nil || !errors.Is(err, os.ErrNotExist) {
			return true
		}
	}
	return false
}

func collectionPath(dir, relative string) string {
	return filepath.Join(dir, filepath.FromSlash(relative))
}

func segmentFileName(segmentID, generation uint64) string {
	return fmt.Sprintf("segments/%020d/data-%020d.seg", segmentID, generation)
}

func walFileName(segmentID, generation uint64) string {
	return fmt.Sprintf("wal/%020d-%020d.wal", segmentID, generation)
}

func primarySnapshotName(generation uint64) string {
	return fmt.Sprintf("snapshots/primary-%020d.snap", generation)
}

func deleteSnapshotName(generation uint64) string {
	return fmt.Sprintf("snapshots/delete-%020d.snap", generation)
}
