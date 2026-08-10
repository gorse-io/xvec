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
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gofrs/flock"
)

const (
	collectionLockName         = ".collection.lock"
	DefaultSegmentMaxDocuments = uint64(65_536)
	fileLockRetryDelay         = 10 * time.Millisecond
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

// CollectionStats describes current live keys and retained segment resources.
type CollectionStats struct {
	DocumentCount         uint64
	ImmutableSegmentCount uint64
	MutableDocumentCount  uint64
	DeletedDocumentCount  uint64
	MemoryUsageBytes      uint64
}

// CollectionStore owns one consistent manifest, WAL, and segment view. A
// writable handle holds the exclusive collection lock; any number of read-only
// handles can hold the shared lock together.
type CollectionStore struct {
	mu           sync.RWMutex
	dir          string
	readOnly     bool
	closed       bool
	poisoned     error
	lock         *flock.Flock
	versions     *VersionManager
	manager      *SegmentManager
	engine       *WriteEngine
	wal          *WAL
	idMapWorking string
}

// SegmentSnapshot is an owned, stable view used by the collection index layer.
// Immutable snapshots correspond to PersistedSegments; the final mutable
// snapshot corresponds to the current WAL-backed writing segment.
type SegmentSnapshot struct {
	Metadata  SegmentMetadata
	Documents []StoredDocument
	Mutable   bool
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
	lock := flock.New(filepath.Join(dir, collectionLockName))
	locked, err := lock.TryLockContext(ctx, fileLockRetryDelay)
	if err != nil {
		return nil, fmt.Errorf("db: lock collection creation: %w", err)
	}
	if !locked {
		return nil, errors.New("db: collection creation lock unavailable")
	}
	var primary *PrimaryKeyMap
	var idMapWorking string
	fail := func(failure error, wal *WAL, created ...string) (*CollectionStore, error) {
		var cleanupErrors []error
		if wal != nil {
			cleanupErrors = append(cleanupErrors, wal.Close())
		}
		if primary != nil {
			cleanupErrors = append(cleanupErrors, primary.Close())
			primary = nil
		}
		if idMapWorking != "" {
			cleanupErrors = append(cleanupErrors, removeIDMapDirectory(idMapWorking))
		}
		for _, name := range created {
			cleanupErrors = append(cleanupErrors, removeCollectionArtifact(name))
		}
		cleanupErrors = append(cleanupErrors, lock.Close())
		return nil, errors.Join(failure, errors.Join(cleanupErrors...))
	}
	if _, err := OpenVersionManager(ctx, dir); err == nil {
		return fail(ErrManifestExists, nil)
	} else if !errors.Is(err, ErrManifestNotFound) {
		return fail(err, nil)
	}

	idMapWorking = newIDMapWorkingPath(dir)
	primary, err = CreatePrimaryKeyMap(ctx, idMapWorking)
	if err != nil {
		return fail(err, nil)
	}
	idMapRelative := idMapCheckpointName(1)
	idMapPath := collectionPath(dir, idMapRelative)
	deletesPath := collectionPath(dir, deleteSnapshotName(1))
	deletes := NewDeleteStore()
	if err := primary.Checkpoint(ctx, idMapPath); err != nil {
		return fail(err, nil)
	}
	if err := deletes.WriteSnapshot(ctx, deletesPath); err != nil {
		return fail(err, nil, idMapPath)
	}
	walRelative := walFileName(0, 1)
	walPath := collectionPath(dir, walRelative)
	if err := ensureDirectorySynced(filepath.Dir(walPath)); err != nil {
		return fail(fmt.Errorf("db: create WAL directory: %w", err), nil, idMapPath, deletesPath)
	}
	wal, err := CreateWAL(ctx, walPath, options.WAL)
	if err != nil {
		return fail(err, nil, idMapPath, deletesPath)
	}
	writing, err := NewWriteSegment(0, 0, capacity)
	if err != nil {
		return fail(err, wal, walPath, idMapPath, deletesPath)
	}
	manager := NewSegmentManager(primary, deletes)
	if err := manager.SetWriting(writing); err != nil {
		return fail(err, wal, walPath, idMapPath, deletesPath)
	}
	manifest := Manifest{
		FormatVersion: DiskFormatVersion, Schema: slices.Clone(schema),
		EnableMmap: options.EnableMmap, SegmentMaxDocuments: capacity,
		WritingSegment: &SegmentMetadata{ID: 0, Files: []string{walRelative}},
		IDMap:          idMapRelative, DeleteSnapshotGeneration: 1, NextSegmentID: 1,
	}
	versions, err := CreateVersionManager(ctx, dir, manifest)
	if err != nil {
		if versions == nil {
			return fail(err, wal, walPath, walPath+".lock", idMapPath, deletesPath)
		}
	}
	createPublishErr := err
	engine, err := NewWriteEngine(manager, wal)
	if err != nil {
		return fail(err, wal, walPath, walPath+".lock", idMapPath, deletesPath)
	}
	store := &CollectionStore{
		dir: dir, lock: lock, versions: versions, manager: manager,
		engine: engine, wal: wal, idMapWorking: idMapWorking,
	}
	primary = nil
	return store, createPublishErr
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
	lock := flock.New(filepath.Join(dir, collectionLockName))
	var locked bool
	if options.ReadOnly {
		locked, err = lock.TryRLockContext(ctx, fileLockRetryDelay)
	} else {
		locked, err = lock.TryLockContext(ctx, fileLockRetryDelay)
	}
	if err != nil {
		return nil, fmt.Errorf("db: lock collection open: %w", err)
	}
	if !locked {
		return nil, errors.New("db: collection open lock unavailable")
	}
	var primary *PrimaryKeyMap
	var wal *WAL
	var idMapWorking string
	fail := func(failure error) (*CollectionStore, error) {
		var cleanupErrors []error
		if wal != nil {
			cleanupErrors = append(cleanupErrors, wal.Close())
		}
		if primary != nil {
			cleanupErrors = append(cleanupErrors, primary.Close())
		}
		if idMapWorking != "" {
			cleanupErrors = append(cleanupErrors, removeIDMapDirectory(idMapWorking))
		}
		cleanupErrors = append(cleanupErrors, lock.Close())
		return nil, errors.Join(failure, errors.Join(cleanupErrors...))
	}
	versions, err := OpenVersionManager(ctx, dir)
	if err != nil {
		return fail(err)
	}
	manifest := versions.Current()
	if err := validateLifecycleManifest(manifest); err != nil {
		return fail(err)
	}
	idMapCheckpoint := collectionPath(dir, manifest.IDMap)
	if options.ReadOnly {
		primary, err = OpenPrimaryKeyMapReadOnly(ctx, idMapCheckpoint)
	} else {
		idMapWorking = newIDMapWorkingPath(dir)
		primary, err = OpenPrimaryKeyMap(ctx, idMapCheckpoint, idMapWorking)
	}
	if err != nil {
		return fail(fmt.Errorf("%w: open IDMap checkpoint: %v", ErrCollectionCorrupt, err))
	}
	deletes, err := LoadDeleteStore(ctx, collectionPath(dir, deleteSnapshotName(manifest.DeleteSnapshotGeneration)))
	if err != nil {
		return fail(fmt.Errorf("%w: load delete snapshot: %v", ErrCollectionCorrupt, err))
	}
	manager := NewSegmentManager(primary, deletes)
	for _, metadata := range manifest.PersistedSegments {
		segment, err := OpenImmutableSegment(ctx, dir, metadata)
		if err != nil {
			return fail(fmt.Errorf("%w: open segment %d: %v", ErrCollectionCorrupt, metadata.ID, err))
		}
		if err := manager.AddImmutable(segment); err != nil {
			return fail(fmt.Errorf("%w: add segment %d: %v", ErrCollectionCorrupt, metadata.ID, err))
		}
	}
	nextDocID := manifest.WritingSegmentStartDocID
	writing, err := NewWriteSegment(manifest.WritingSegment.ID, nextDocID, manifest.SegmentMaxDocuments)
	if err != nil {
		return fail(fmt.Errorf("%w: create writing segment: %v", ErrCollectionCorrupt, err))
	}
	if err := manager.SetWriting(writing); err != nil {
		return fail(fmt.Errorf("%w: install writing segment: %v", ErrCollectionCorrupt, err))
	}
	walPath := collectionPath(dir, manifest.WritingSegment.Files[0])
	if options.ReadOnly {
		wal, err = OpenWALReadOnly(ctx, walPath)
	} else {
		wal, err = OpenWAL(ctx, walPath, options.WAL)
	}
	if err != nil {
		return fail(fmt.Errorf("%w: open writing WAL: %v", ErrCollectionCorrupt, err))
	}
	if err := replayWriteWAL(ctx, wal, manager); err != nil {
		return fail(err)
	}
	if err := validateCollectionState(manager); err != nil {
		return fail(err)
	}
	store := &CollectionStore{
		dir: dir, readOnly: options.ReadOnly, lock: lock, versions: versions,
		manager: manager, wal: wal, idMapWorking: idMapWorking,
	}
	if !options.ReadOnly {
		store.engine, err = NewWriteEngine(manager, wal)
		if err != nil {
			return fail(err)
		}
	}
	return store, nil
}

// Insert delegates a batch insert to the current WAL writer.
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

// Upsert delegates a batch upsert to the current WAL writer.
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

// Update delegates a batch update to the current WAL writer.
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

// Delete delegates a primary-key batch delete to the current WAL writer.
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

// LiveDocuments returns a stable collection-level view while excluding a
// concurrent flush. Public query orchestration additionally serializes writes
// so multi-step upserts cannot be observed halfway through application.
func (c *CollectionStore) LiveDocuments(ctx context.Context) ([]StoredDocument, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, ErrCollectionClosed
	}
	return c.manager.LiveDocuments(ctx)
}

// SegmentSnapshots returns retained documents grouped by physical segment.
// Logical deletions and superseded versions remain present so immutable index
// artifacts never need rewriting; query-time live masks exclude them.
func (c *CollectionStore) SegmentSnapshots(ctx context.Context) ([]SegmentSnapshot, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	if ctx == nil {
		return nil, errors.New("db: nil segment-snapshot context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, ErrCollectionClosed
	}
	segments := c.manager.ImmutableSegments()
	snapshots := make([]SegmentSnapshot, 0, len(segments)+1)
	for _, segment := range segments {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		snapshots = append(snapshots, SegmentSnapshot{Metadata: segment.Metadata(), Documents: segment.Documents()})
	}
	if writing := c.manager.Writing(); writing != nil {
		snapshots = append(snapshots, SegmentSnapshot{Metadata: writing.Metadata(), Documents: writing.Documents(), Mutable: true})
	}
	return snapshots, nil
}

// DocumentCount returns the number of live primary keys in memory.
func (c *CollectionStore) DocumentCount() uint64 {
	return c.Stats().DocumentCount
}

// Stats returns a point-in-time storage snapshot without cloning documents.
func (c *CollectionStore) Stats() CollectionStats {
	if c == nil {
		return CollectionStats{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.manager == nil || c.manager.PrimaryKeys() == nil {
		return CollectionStats{}
	}
	storage := c.manager.StorageStats()
	return CollectionStats{
		DocumentCount:         uint64(c.manager.PrimaryKeys().Count()),
		ImmutableSegmentCount: storage.ImmutableSegmentCount,
		MutableDocumentCount:  storage.MutableDocumentCount,
		DeletedDocumentCount:  storage.DeletedDocumentCount,
		MemoryUsageBytes:      storage.MemoryUsageBytes,
	}
}

// OptimizationNeeded reports whether rewriting would flush mutable documents,
// remove deleted versions, or reduce the immutable layout to the canonical
// contiguous-ID runs bounded by SegmentMaxDocuments.
func (c *CollectionStore) OptimizationNeeded(ctx context.Context) (bool, error) {
	if c == nil {
		return false, errors.New("db: nil collection")
	}
	if ctx == nil {
		return false, errors.New("db: nil optimization context")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return false, ErrCollectionClosed
	}
	if writing := c.manager.Writing(); writing != nil && len(writing.Documents()) != 0 {
		return true, nil
	}
	if c.manager.Deletes().Count() != 0 {
		return true, nil
	}
	documents, err := c.manager.LiveDocuments(ctx)
	if err != nil {
		return false, err
	}
	expected := rewriteDocumentRuns(documents, c.versions.Current().SegmentMaxDocuments)
	actual := c.manager.ImmutableMetadata()
	if len(expected) != len(actual) {
		return true, nil
	}
	for index := range expected {
		run := expected[index]
		metadata := actual[index]
		if metadata.MinDocID != run[0].DocID || metadata.MaxDocID != run[len(run)-1].DocID || metadata.DocCount != uint64(len(run)) {
			return true, nil
		}
	}
	return false, nil
}

// PruneObsoleteArtifacts removes only storage artifacts owned by this package
// that are no longer referenced by the current manifest. CURRENT is already
// the commit point, so an interrupted prune is harmless and can be retried.
// Unknown files and manifest generations are deliberately left untouched.
func (c *CollectionStore) PruneObsoleteArtifacts(ctx context.Context) error {
	if c == nil {
		return errors.New("db: nil collection")
	}
	if ctx == nil {
		return errors.New("db: nil artifact prune context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireWritableLocked(); err != nil {
		return err
	}

	manifest := c.versions.Current()
	keep := make(map[string]struct{})
	keepRelative := func(relative string) {
		if relative == "" {
			return
		}
		keep[filepath.Clean(collectionPath(c.dir, relative))] = struct{}{}
	}
	for _, segment := range manifest.PersistedSegments {
		for _, relative := range segment.Files {
			keepRelative(relative)
		}
	}
	if manifest.WritingSegment != nil {
		for _, relative := range manifest.WritingSegment.Files {
			keepRelative(relative)
			keep[filepath.Clean(collectionPath(c.dir, relative)+".lock")] = struct{}{}
		}
	}
	for _, snapshot := range manifest.SegmentIndexSnapshots {
		for _, artifact := range snapshot.Artifacts {
			keepRelative(artifact.File)
		}
	}
	keepRelative(manifest.IDMap)
	keepRelative(deleteSnapshotName(manifest.DeleteSnapshotGeneration))
	if c.idMapWorking != "" {
		keep[filepath.Clean(c.idMapWorking)] = struct{}{}
	}

	segmentRoot := filepath.Join(c.dir, "segments")
	segmentDirectories, err := ownedSubdirectories(segmentRoot)
	if err != nil {
		return err
	}
	candidates := make([]string, 0)
	for _, directory := range segmentDirectories {
		matches, matchErr := ownedFiles(directory, "data-*.seg")
		if matchErr != nil {
			return matchErr
		}
		candidates = append(candidates, matches...)
	}
	idMapCandidates, err := ownedIDMapDirectories(filepath.Join(c.dir, "idmap"))
	if err != nil {
		return err
	}
	candidates = append(candidates, idMapCandidates...)
	for _, specification := range []struct {
		directory string
		patterns  []string
	}{
		{filepath.Join(c.dir, "wal"), []string{"*.wal.lock", "*.wal"}},
		{filepath.Join(c.dir, "snapshots"), []string{"delete-*.snap"}},
		{filepath.Join(c.dir, "indexes"), []string{"*.zvi", "*.pebble"}},
	} {
		matches, matchErr := ownedFiles(specification.directory, specification.patterns...)
		if matchErr != nil {
			return matchErr
		}
		candidates = append(candidates, matches...)
	}
	touchedDirectories := make(map[string]struct{})
	for _, name := range candidates {
		if err := ctx.Err(); err != nil {
			return err
		}
		name = filepath.Clean(name)
		info, err := os.Lstat(name)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("db: inspect obsolete artifact %q: %w", name, err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("db: refuse to prune non-regular artifact %q", name)
		}
		directoryArtifact := filepath.Ext(name) == ".pebble"
		if directoryArtifact && !info.IsDir() || !directoryArtifact && !info.Mode().IsRegular() {
			return fmt.Errorf("db: refuse to prune invalid artifact %q", name)
		}
		if _, retained := keep[name]; retained {
			continue
		}
		if directoryArtifact {
			err = os.RemoveAll(name)
		} else {
			err = os.Remove(name)
		}
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("db: remove obsolete artifact %q: %w", name, err)
		}
		touchedDirectories[filepath.Dir(name)] = struct{}{}
	}

	// Persist file removals before attempting the cosmetic removal of empty
	// segment directories. Directory fsync is a no-op where unsupported.
	for directory := range touchedDirectories {
		if err := syncDirectory(directory); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("db: sync pruned artifact directory %q: %w", directory, err)
		}
	}
	removedDirectory := false
	for _, directory := range segmentDirectories {
		if err := ctx.Err(); err != nil {
			return err
		}
		info, err := os.Lstat(directory)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("db: inspect segment directory %q: %w", directory, err)
		}
		if !info.IsDir() {
			continue
		}
		if err := os.Remove(directory); err == nil {
			removedDirectory = true
		} else if !errors.Is(err, os.ErrNotExist) {
			// A non-empty directory contains either the current segment or an
			// unknown file that this conservative prune must retain.
			continue
		}
	}
	if removedDirectory {
		if err := syncDirectory(segmentRoot); err != nil {
			return fmt.Errorf("db: sync segment directory: %w", err)
		}
	}
	return nil
}

func ownedSubdirectories(directory string) ([]string, error) {
	entries, err := ownedDirectoryEntries(directory)
	if err != nil {
		return nil, err
	}
	result := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.Type()&os.ModeSymlink == 0 && entry.IsDir() {
			result = append(result, filepath.Join(directory, entry.Name()))
		}
	}
	return result, nil
}

func ownedFiles(directory string, patterns ...string) ([]string, error) {
	entries, err := ownedDirectoryEntries(directory)
	if err != nil {
		return nil, err
	}
	result := make([]string, 0)
	for _, entry := range entries {
		for _, pattern := range patterns {
			matched, matchErr := filepath.Match(pattern, entry.Name())
			if matchErr != nil {
				return nil, fmt.Errorf("db: match artifact pattern %q: %w", pattern, matchErr)
			}
			if matched {
				result = append(result, filepath.Join(directory, entry.Name()))
				break
			}
		}
	}
	return result, nil
}

func ownedIDMapDirectories(directory string) ([]string, error) {
	entries, err := ownedDirectoryEntries(directory)
	if err != nil {
		return nil, err
	}
	result := make([]string, 0, len(entries))
	for _, entry := range entries {
		if isOwnedIDMapName(entry.Name()) {
			result = append(result, filepath.Join(directory, entry.Name()))
		}
	}
	return result, nil
}

func isOwnedIDMapName(name string) bool {
	if strings.HasPrefix(name, "idmap-") && strings.HasSuffix(name, ".pebble") {
		digits := strings.TrimSuffix(strings.TrimPrefix(name, "idmap-"), ".pebble")
		if len(digits) == 20 {
			generation, err := strconv.ParseUint(digits, 10, 64)
			return err == nil && generation > 0
		}
		return false
	}
	if !strings.HasPrefix(name, ".working-") || !strings.HasSuffix(name, ".pebble") {
		return false
	}
	parts := strings.Split(strings.TrimSuffix(strings.TrimPrefix(name, ".working-"), ".pebble"), "-")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return false
	}
	_, firstErr := strconv.ParseUint(parts[0], 10, 64)
	_, secondErr := strconv.ParseUint(parts[1], 10, 64)
	return firstErr == nil && secondErr == nil
}

func ownedDirectoryEntries(directory string) ([]os.DirEntry, error) {
	info, err := os.Lstat(directory)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("db: inspect artifact directory %q: %w", directory, err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return nil, fmt.Errorf("db: refuse to scan non-directory artifact path %q", directory)
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, fmt.Errorf("db: read artifact directory %q: %w", directory, err)
	}
	return entries, nil
}

// Flush atomically turns the non-empty write segment into an immutable segment,
// checkpoints IDMap/deletion state, publishes a new manifest, and rotates the
// outer WAL and disposable IDMap working copy.
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
	var nextPrimary *PrimaryKeyMap
	var nextWorking string
	cleanup := func() {
		if nextPrimary != nil {
			_ = nextPrimary.Close()
			nextPrimary = nil
		}
		if nextWorking != "" {
			_ = removeIDMapDirectory(nextWorking)
		}
		for _, name := range created {
			_ = removeCollectionArtifact(name)
		}
	}

	idMapRelative, err := c.availableArtifact(idMapCheckpointName, artifactGeneration)
	if err != nil {
		cleanup()
		return err
	}
	idMapPath := collectionPath(c.dir, idMapRelative)
	if err := c.manager.PrimaryKeys().Checkpoint(ctx, idMapPath); err != nil {
		cleanup()
		return fmt.Errorf("db: checkpoint IDMap: %w", err)
	}
	created = append(created, idMapPath)
	nextWorking = newIDMapWorkingPath(c.dir)
	nextPrimary, err = OpenPrimaryKeyMap(ctx, idMapPath, nextWorking)
	if err != nil {
		cleanup()
		return fmt.Errorf("db: open next IDMap working copy: %w", err)
	}

	snapshotGeneration, err := c.nextDeleteSnapshotGeneration(current.DeleteSnapshotGeneration)
	if err != nil {
		cleanup()
		return err
	}
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
	created = append(created, walPath, walPath+".lock")
	nextEngine, err := NewWriteEngine(c.manager, nextWAL)
	if err != nil {
		_ = nextWAL.Close()
		cleanup()
		return err
	}

	nextManifest := current.Clone()
	nextManifest.PersistedSegments = append(nextManifest.PersistedSegments, immutable.Metadata())
	nextManifest.WritingSegment = &SegmentMetadata{ID: current.NextSegmentID, Files: []string{walRelative}}
	nextManifest.WritingSegmentStartDocID = lastDocID + 1
	nextManifest.IDMap = idMapRelative
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
		cleanupErr := nextPrimary.Close()
		nextPrimary = nil
		_ = removeIDMapDirectory(nextWorking)
		c.poisoned = fmt.Errorf("db: reopen required after committed generation %d failed to rotate: %w", published.Generation, err)
		return errors.Join(publishErr, c.poisoned, cleanupErr)
	}
	oldWAL := c.wal
	oldPrimary := c.manager.PrimaryKeys()
	oldWorking := c.idMapWorking
	if err := c.manager.ReplacePrimaryKeys(nextPrimary); err != nil {
		c.poisoned = fmt.Errorf("db: reopen required after committed generation %d failed to swap IDMap: %w", published.Generation, err)
		closeErr := errors.Join(nextWAL.Close(), nextPrimary.Close())
		nextPrimary = nil
		removeErr := removeIDMapDirectory(nextWorking)
		return errors.Join(publishErr, c.poisoned, closeErr, removeErr)
	}
	c.wal = nextWAL
	c.engine = nextEngine
	c.idMapWorking = nextWorking
	nextPrimary = nil
	return errors.Join(
		publishErr,
		oldWAL.Close(),
		oldPrimary.Close(),
		removeIDMapDirectory(oldWorking),
	)
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

// PublishSchema atomically installs a validated schema payload in a new
// manifest generation. committed is true once CURRENT names that generation,
// including the rare case where a post-commit directory sync reports an
// error. Callers must update their in-memory schema whenever committed is true.
func (c *CollectionStore) PublishSchema(ctx context.Context, schema json.RawMessage) (committed bool, err error) {
	if c == nil {
		return false, errors.New("db: nil collection")
	}
	if ctx == nil {
		return false, errors.New("db: nil publish schema context")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if err := validateSchemaJSON(schema); err != nil {
		return false, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireWritableLocked(); err != nil {
		return false, err
	}
	current := c.versions.Current()
	if bytes.Equal(current.Schema, schema) {
		return false, nil
	}
	next := current.Clone()
	next.Schema = slices.Clone(schema)
	_, publishErr := c.versions.Publish(ctx, next)
	committed = c.versions.Current().Generation != current.Generation
	return committed, publishErr
}

// PublishSegmentIndexSnapshots atomically installs immutable per-segment index
// metadata. Vector artifacts must already exist as regular files and FTS/INVERT
// artifacts as Pebble directories below the collection directory.
func (c *CollectionStore) PublishSegmentIndexSnapshots(ctx context.Context, snapshots []SegmentIndexSnapshotMetadata) (committed bool, err error) {
	if c == nil {
		return false, errors.New("db: nil collection")
	}
	if ctx == nil {
		return false, errors.New("db: nil publish segment indexes context")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	snapshots = cloneSegmentIndexSnapshots(snapshots)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireWritableLocked(); err != nil {
		return false, err
	}
	current := c.versions.Current()
	if reflect.DeepEqual(current.SegmentIndexSnapshots, snapshots) {
		return false, nil
	}
	next := current.Clone()
	next.SegmentIndexSnapshots = snapshots
	if err := next.Validate(); err != nil {
		return false, err
	}
	for _, snapshot := range snapshots {
		for _, artifact := range snapshot.Artifacts {
			path := collectionPath(c.dir, artifact.File)
			info, statErr := os.Lstat(path)
			if statErr != nil {
				return false, fmt.Errorf("db: inspect segment index artifact %q: %w", artifact.File, statErr)
			}
			if info.Mode()&os.ModeSymlink != 0 {
				return false, fmt.Errorf("db: segment index artifact %q is a symlink", artifact.File)
			}
			pebbleDirectory := artifact.Kind == "fts" || artifact.Kind == "invert"
			if pebbleDirectory && !info.IsDir() {
				return false, fmt.Errorf("db: segment index artifact %q is not a Pebble directory", artifact.File)
			}
			if !pebbleDirectory && !info.Mode().IsRegular() {
				return false, fmt.Errorf("db: segment index artifact %q is not a regular file", artifact.File)
			}
		}
	}
	_, publishErr := c.versions.Publish(ctx, next)
	committed = c.versions.Current().Generation != current.Generation
	return committed, publishErr
}

// RewriteDocuments atomically replaces every live document payload together
// with the collection schema. Document IDs and primary keys must exactly match
// the current live snapshot in ascending document-ID order. Superseded and
// deleted versions are reclaimed by the rewrite, while the next document ID
// remains monotonic. committed reports whether CURRENT reached the new
// generation even if a post-commit sync or old-WAL close reports an error.
func (c *CollectionStore) RewriteDocuments(ctx context.Context, schema json.RawMessage, documents []StoredDocument) (committed bool, err error) {
	if c == nil {
		return false, errors.New("db: nil collection")
	}
	if ctx == nil {
		return false, errors.New("db: nil rewrite context")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if err := validateSchemaJSON(schema); err != nil {
		return false, err
	}
	documents = cloneDocuments(documents)

	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireWritableLocked(); err != nil {
		return false, err
	}
	currentLive, err := c.manager.LiveDocuments(ctx)
	if err != nil {
		return false, err
	}
	if len(documents) != len(currentLive) {
		return false, fmt.Errorf("db: rewrite has %d documents, current snapshot has %d", len(documents), len(currentLive))
	}
	keys := make(map[string]struct{}, len(documents))
	for index := range documents {
		document := &documents[index]
		current := &currentLive[index]
		if document.DocID != current.DocID || document.PrimaryKey != current.PrimaryKey {
			return false, fmt.Errorf("db: rewrite document %d does not match current snapshot", index)
		}
		if err := validatePrimaryKey(document.PrimaryKey); err != nil {
			return false, err
		}
		if len(document.Payload) > MaxDocumentPayloadSize {
			return false, fmt.Errorf("db: rewrite document %d payload is too large", document.DocID)
		}
		if _, exists := keys[document.PrimaryKey]; exists {
			return false, fmt.Errorf("db: rewrite contains duplicate primary key %q", document.PrimaryKey)
		}
		keys[document.PrimaryKey] = struct{}{}
	}
	if err := c.wal.Sync(ctx); err != nil {
		return false, err
	}

	nextDocID, err := c.rewriteNextDocumentID()
	if err != nil {
		return false, err
	}
	if len(documents) > 0 && documents[len(documents)-1].DocID >= nextDocID {
		return false, fmt.Errorf("%w: rewrite document IDs reach or exceed the next writable ID", ErrCollectionCorrupt)
	}
	current := c.versions.Current()
	runs := rewriteDocumentRuns(documents, current.SegmentMaxDocuments)
	if uint64(len(runs)) > math.MaxUint64-current.NextSegmentID {
		return false, errors.New("db: segment ID space is exhausted")
	}
	writingID := current.NextSegmentID + uint64(len(runs))
	if writingID == math.MaxUint64 {
		return false, errors.New("db: segment ID space is exhausted")
	}
	if current.Generation == math.MaxUint64 {
		return false, errors.New("db: manifest generation space is exhausted")
	}

	created := make([]string, 0, len(runs)+3)
	primaryWorking := newIDMapWorkingPath(c.dir)
	primary, err := CreatePrimaryKeyMap(ctx, primaryWorking)
	if err != nil {
		return false, fmt.Errorf("db: create rewritten IDMap working copy: %w", err)
	}
	candidateOwned := true
	cleanup := func() {
		if candidateOwned {
			_ = primary.Close()
			_ = removeIDMapDirectory(primaryWorking)
		}
		for _, name := range created {
			_ = removeCollectionArtifact(name)
		}
	}
	artifactGeneration := current.Generation + 1
	deletes := NewDeleteStore()
	nextManager := NewSegmentManager(primary, deletes)
	segmentMetadata := make([]SegmentMetadata, 0, len(runs))
	for runIndex, run := range runs {
		segmentID := current.NextSegmentID + uint64(runIndex)
		writing, createErr := NewWriteSegment(segmentID, run[0].DocID, uint64(len(run)))
		if createErr != nil {
			cleanup()
			return false, createErr
		}
		for index := range run {
			document := &run[index]
			if _, appendErr := writing.AppendExpected(ctx, document.DocID, document.PrimaryKey, document.Payload); appendErr != nil {
				cleanup()
				return false, appendErr
			}
		}
		segmentRelative, nameErr := c.availableArtifact(func(generation uint64) string {
			return segmentFileName(segmentID, generation)
		}, artifactGeneration)
		if nameErr != nil {
			cleanup()
			return false, nameErr
		}
		segmentPath := collectionPath(c.dir, segmentRelative)
		created = append(created, segmentPath)
		immutable, snapshotErr := writing.Snapshot(ctx, c.dir, segmentRelative)
		if snapshotErr != nil {
			cleanup()
			return false, fmt.Errorf("db: snapshot rewritten segment: %w", snapshotErr)
		}
		if addErr := nextManager.AddImmutable(immutable); addErr != nil {
			cleanup()
			return false, addErr
		}
		for index := range run {
			document := &run[index]
			if _, replaced, putErr := primary.Put(ctx, document.PrimaryKey, document.DocID); putErr != nil {
				cleanup()
				return false, fmt.Errorf("db: build rewritten IDMap: %w", putErr)
			} else if replaced {
				cleanup()
				return false, fmt.Errorf("db: rewrite contains duplicate primary key %q", document.PrimaryKey)
			}
		}
		segmentMetadata = append(segmentMetadata, immutable.Metadata())
	}

	nextWriting, err := NewWriteSegment(writingID, nextDocID, current.SegmentMaxDocuments)
	if err != nil {
		cleanup()
		return false, err
	}
	if err := nextManager.SetWriting(nextWriting); err != nil {
		cleanup()
		return false, err
	}
	if err := validateCollectionState(nextManager); err != nil {
		cleanup()
		return false, err
	}

	idMapRelative, err := c.availableArtifact(idMapCheckpointName, artifactGeneration)
	if err != nil {
		cleanup()
		return false, err
	}
	idMapPath := collectionPath(c.dir, idMapRelative)
	created = append(created, idMapPath)
	if err := primary.Checkpoint(ctx, idMapPath); err != nil {
		cleanup()
		return false, fmt.Errorf("db: checkpoint rewritten IDMap: %w", err)
	}
	snapshotGeneration, err := c.nextDeleteSnapshotGeneration(current.DeleteSnapshotGeneration)
	if err != nil {
		cleanup()
		return false, err
	}
	deletesPath := collectionPath(c.dir, deleteSnapshotName(snapshotGeneration))
	created = append(created, deletesPath)
	if err := deletes.WriteSnapshot(ctx, deletesPath); err != nil {
		cleanup()
		return false, fmt.Errorf("db: write rewritten delete snapshot: %w", err)
	}

	walRelative, err := c.availableArtifact(func(generation uint64) string {
		return walFileName(writingID, generation)
	}, artifactGeneration)
	if err != nil {
		cleanup()
		return false, err
	}
	walPath := collectionPath(c.dir, walRelative)
	if err := ensureDirectorySynced(filepath.Dir(walPath)); err != nil {
		cleanup()
		return false, fmt.Errorf("db: create rewritten WAL directory: %w", err)
	}
	created = append(created, walPath, walPath+".lock")
	nextWAL, err := CreateWAL(ctx, walPath, c.wal.options)
	if err != nil {
		cleanup()
		return false, fmt.Errorf("db: create rewritten WAL: %w", err)
	}
	nextEngine, err := NewWriteEngine(nextManager, nextWAL)
	if err != nil {
		_ = nextWAL.Close()
		cleanup()
		return false, err
	}

	nextManifest := current.Clone()
	nextManifest.Schema = slices.Clone(schema)
	nextManifest.PersistedSegments = segmentMetadata
	nextManifest.SegmentIndexSnapshots = nil
	nextManifest.WritingSegment = &SegmentMetadata{ID: writingID, Files: []string{walRelative}}
	nextManifest.WritingSegmentStartDocID = nextDocID
	nextManifest.IDMap = idMapRelative
	nextManifest.DeleteSnapshotGeneration = snapshotGeneration
	nextManifest.NextSegmentID = writingID + 1
	_, publishErr := c.versions.Publish(ctx, nextManifest)
	committed = c.versions.Current().Generation != current.Generation
	if !committed {
		_ = nextWAL.Close()
		cleanup()
		return false, publishErr
	}

	oldWAL := c.wal
	oldPrimary := c.manager.PrimaryKeys()
	oldWorking := c.idMapWorking
	c.manager = nextManager
	c.wal = nextWAL
	c.engine = nextEngine
	c.idMapWorking = primaryWorking
	candidateOwned = false
	return true, errors.Join(
		publishErr,
		oldWAL.Close(),
		oldPrimary.Close(),
		removeIDMapDirectory(oldWorking),
	)
}

func (c *CollectionStore) rewriteNextDocumentID() (uint64, error) {
	writing := c.manager.Writing()
	if writing == nil {
		return 0, fmt.Errorf("%w: collection has no writing segment", ErrCollectionCorrupt)
	}
	next, err := writing.NextDocumentID()
	if err == nil {
		return next, nil
	}
	if !errors.Is(err, ErrSegmentFull) {
		return 0, err
	}
	_, maximum := writing.ReservedRange()
	if maximum == math.MaxUint64 {
		return 0, errors.New("db: document ID space is exhausted")
	}
	return maximum + 1, nil
}

func (c *CollectionStore) nextDeleteSnapshotGeneration(generation uint64) (uint64, error) {
	for {
		if generation == math.MaxUint64 {
			return 0, errors.New("db: snapshot generation space is exhausted")
		}
		generation++
		if _, err := os.Lstat(collectionPath(c.dir, deleteSnapshotName(generation))); errors.Is(err, os.ErrNotExist) {
			return generation, nil
		} else if err != nil {
			return 0, fmt.Errorf("db: inspect delete snapshot generation: %w", err)
		}
	}
}

func rewriteDocumentRuns(documents []StoredDocument, maximum uint64) [][]StoredDocument {
	if len(documents) == 0 {
		return nil
	}
	runs := make([][]StoredDocument, 0, 1)
	for start := 0; start < len(documents); {
		end := start + 1
		for end < len(documents) && uint64(end-start) < maximum && documents[end].DocID == documents[end-1].DocID+1 {
			end++
		}
		runs = append(runs, documents[start:end])
		start = end
	}
	return runs
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

// Close releases the outer WAL and IDMap before the collection lock. Working
// IDMap state is disposable because the outer WAL can recreate it.
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
	var errs []error
	if c.wal != nil {
		errs = append(errs, c.wal.Close())
	}
	if c.manager != nil && c.manager.PrimaryKeys() != nil {
		errs = append(errs, c.manager.PrimaryKeys().Close())
	}
	if c.idMapWorking != "" {
		errs = append(errs, removeIDMapDirectory(c.idMapWorking))
		c.idMapWorking = ""
	}
	if c.lock != nil {
		errs = append(errs, c.lock.Close())
	}
	return errors.Join(errs...)
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
	if c.poisoned != nil {
		return c.poisoned
	}
	if c.engine != nil {
		if err := c.engine.Err(); err != nil {
			return err
		}
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
		if writing == nil {
			return errors.New("operation requires a writing segment")
		}
	}
	switch operation.Type {
	case writeOperationInsert:
		if _, exists, err := manager.PrimaryKeys().Get(operation.PrimaryKey); err != nil {
			return err
		} else if exists {
			return ErrPrimaryKeyExists
		}
		if _, err := writing.AppendExpected(ctx, operation.DocID, operation.PrimaryKey, operation.Payload); err != nil {
			return err
		}
		_, _, err := manager.PrimaryKeys().Put(ctx, operation.PrimaryKey, operation.DocID)
		return err
	case writeOperationUpsert, writeOperationUpdate:
		previous, existed, err := manager.PrimaryKeys().Get(operation.PrimaryKey)
		if err != nil {
			return err
		}
		if operation.Type == writeOperationUpdate && !existed {
			return ErrPrimaryKeyNotFound
		}
		if _, err := writing.AppendExpected(ctx, operation.DocID, operation.PrimaryKey, operation.Payload); err != nil {
			return err
		}
		if existed {
			if _, err := manager.Deletes().MarkDeleted(ctx, previous); err != nil {
				return err
			}
		}
		_, _, err = manager.PrimaryKeys().Put(ctx, operation.PrimaryKey, operation.DocID)
		return err
	case writeOperationDelete:
		if len(operation.Payload) != 0 {
			return errors.New("delete operation contains a payload")
		}
		docID, existed, err := manager.PrimaryKeys().Get(operation.PrimaryKey)
		if err != nil {
			return err
		}
		if !existed || docID != operation.DocID {
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
	live := make(map[uint64]string)
	known := make(map[uint64]struct{})
	segments := manager.ImmutableSegments()
	for _, segment := range segments {
		for _, document := range segment.Documents() {
			known[document.DocID] = struct{}{}
			if !manager.Deletes().IsDeleted(document.DocID) {
				live[document.DocID] = document.PrimaryKey
			}
		}
	}
	if writing := manager.Writing(); writing != nil {
		for _, document := range writing.Documents() {
			known[document.DocID] = struct{}{}
			if !manager.Deletes().IsDeleted(document.DocID) {
				live[document.DocID] = document.PrimaryKey
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
	if manager.primaryKey.Count() != len(live) {
		return fmt.Errorf("%w: IDMap count %d differs from live document count %d", ErrCollectionCorrupt, manager.primaryKey.Count(), len(live))
	}
	if err := manager.primaryKey.forEach(context.Background(), func(key string, docID uint64) error {
		if liveKey, exists := live[docID]; !exists || liveKey != key {
			return fmt.Errorf("%w: primary key %q has invalid document ID %d", ErrCollectionCorrupt, key, docID)
		}
		return nil
	}); err != nil {
		if errors.Is(err, ErrCollectionCorrupt) {
			return err
		}
		return fmt.Errorf("%w: validate IDMap: %v", ErrCollectionCorrupt, err)
	}
	return nil
}

func validateLifecycleManifest(manifest Manifest) error {
	if manifest.SegmentMaxDocuments == 0 || manifest.IDMap == "" || manifest.DeleteSnapshotGeneration == 0 {
		return fmt.Errorf("%w: invalid lifecycle generations or capacity", ErrCollectionCorrupt)
	}
	if manifest.WritingSegment == nil || manifest.WritingSegment.DocCount != 0 || len(manifest.WritingSegment.Files) != 1 {
		return fmt.Errorf("%w: invalid writing segment metadata", ErrCollectionCorrupt)
	}
	for _, segment := range manifest.PersistedSegments {
		if segment.DocCount > 0 && manifest.WritingSegmentStartDocID <= segment.MaxDocID {
			return fmt.Errorf("%w: writing segment starts at document %d, not after persisted document %d", ErrCollectionCorrupt, manifest.WritingSegmentStartDocID, segment.MaxDocID)
		}
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

func collectionPath(dir, relative string) string {
	return filepath.Join(dir, filepath.FromSlash(relative))
}

func segmentFileName(segmentID, generation uint64) string {
	return fmt.Sprintf("segments/%020d/data-%020d.seg", segmentID, generation)
}

func walFileName(segmentID, generation uint64) string {
	return fmt.Sprintf("wal/%020d-%020d.wal", segmentID, generation)
}

func idMapCheckpointName(generation uint64) string {
	return fmt.Sprintf("idmap/idmap-%020d.pebble", generation)
}

func deleteSnapshotName(generation uint64) string {
	return fmt.Sprintf("snapshots/delete-%020d.snap", generation)
}

func newIDMapWorkingPath(dir string) string {
	return filepath.Join(dir, "idmap", fmt.Sprintf(".working-%d-%d.pebble", os.Getpid(), time.Now().UnixNano()))
}

func removeIDMapDirectory(name string) error {
	if name == "" {
		return nil
	}
	info, err := os.Lstat(name)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("db: inspect IDMap directory for removal: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return fmt.Errorf("db: refuse to remove invalid IDMap directory %q", name)
	}
	if err := os.RemoveAll(name); err != nil {
		return fmt.Errorf("db: remove IDMap directory %q: %w", name, err)
	}
	if err := syncDirectory(filepath.Dir(name)); err != nil {
		return fmt.Errorf("db: sync removed IDMap directory %q: %w", name, err)
	}
	return nil
}

func removeCollectionArtifact(name string) error {
	if name == "" {
		return nil
	}
	info, err := os.Lstat(name)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("db: refuse to remove symlink artifact %q", name)
	}
	if info.IsDir() {
		return removeIDMapDirectory(name)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("db: refuse to remove non-regular artifact %q", name)
	}
	if err := os.Remove(name); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}
