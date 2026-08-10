// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
package db

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cockroachdb/pebble/v2"
	"github.com/stretchr/testify/require"
)

func TestPrimaryKeyMapPersistsGlobalDocumentIDsInPebbleCheckpoint(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	working := filepath.Join(root, "working.pebble")
	checkpoint := filepath.Join(root, "idmap-1.pebble")
	primary, err := CreatePrimaryKeyMap(ctx, working)
	require.NoError(t, err)

	previous, replaced, err := primary.Put(ctx, "alpha", 41)
	require.NoError(t, err)
	require.False(t, replaced)
	require.Zero(t, previous)
	previous, replaced, err = primary.Put(ctx, "alpha", 42)
	require.NoError(t, err)
	require.True(t, replaced)
	require.Equal(t, uint64(41), previous)
	require.NoError(t, primary.Checkpoint(ctx, checkpoint))
	require.NoError(t, primary.Close())

	reopened, err := OpenPrimaryKeyMap(ctx, checkpoint, filepath.Join(root, "replay.pebble"))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, reopened.Close()) })
	docID, found, err := reopened.Get("alpha")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(42), docID)
	info, err := os.Stat(checkpoint)
	require.NoError(t, err)
	require.True(t, info.IsDir())
}

func TestPrimaryKeyMapDeleteIsPointMutation(t *testing.T) {
	ctx := context.Background()
	primary, err := CreatePrimaryKeyMap(ctx, filepath.Join(t.TempDir(), "working.pebble"))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, primary.Close()) })
	_, _, err = primary.Put(ctx, "alpha", 7)
	require.NoError(t, err)
	removed, found, err := primary.Delete(ctx, "alpha")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(7), removed)
	_, found, err = primary.Get("alpha")
	require.NoError(t, err)
	require.False(t, found)
	require.Zero(t, primary.Count())
}

func TestPrimaryKeyMapReadOnlyOverlayDoesNotChangeCheckpoint(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	working := filepath.Join(root, "working.pebble")
	checkpoint := filepath.Join(root, "checkpoint.pebble")
	primary, err := CreatePrimaryKeyMap(ctx, working)
	require.NoError(t, err)
	_, _, err = primary.Put(ctx, "alpha", 10)
	require.NoError(t, err)
	_, _, err = primary.Put(ctx, "beta", 20)
	require.NoError(t, err)
	require.NoError(t, primary.Checkpoint(ctx, checkpoint))
	require.NoError(t, primary.Close())

	before := snapshotFileTree(t, checkpoint)
	readOnly, err := OpenPrimaryKeyMapReadOnly(ctx, checkpoint)
	require.NoError(t, err)
	require.Equal(t, 2, readOnly.Count())

	previous, found, err := readOnly.Put(ctx, "alpha", 11)
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(10), previous)
	require.Equal(t, 2, readOnly.Count())

	previous, found, err = readOnly.Delete(ctx, "alpha")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(11), previous)
	require.Equal(t, 1, readOnly.Count())
	_, found, err = readOnly.Delete(ctx, "alpha")
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, 1, readOnly.Count())

	_, found, err = readOnly.Put(ctx, "alpha", 12)
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, 2, readOnly.Count())
	_, found, err = readOnly.Put(ctx, "gamma", 30)
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, 3, readOnly.Count())
	previous, found, err = readOnly.Put(ctx, "gamma", 31)
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(30), previous)
	require.Equal(t, 3, readOnly.Count())
	_, found, err = readOnly.Delete(ctx, "gamma")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, 2, readOnly.Count())
	_, found, err = readOnly.Delete(ctx, "gamma")
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, 2, readOnly.Count())
	_, found, err = readOnly.Put(ctx, "gamma", 32)
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, 3, readOnly.Count())
	_, found, err = readOnly.Delete(ctx, "beta")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, 2, readOnly.Count())

	docID, found, err := readOnly.Get("alpha")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(12), docID)
	docID, found, err = readOnly.Get("gamma")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(32), docID)
	_, found, err = readOnly.Get("beta")
	require.NoError(t, err)
	require.False(t, found)

	visited := make(map[string]uint64)
	require.NoError(t, readOnly.forEach(ctx, func(key string, docID uint64) error {
		visited[key] = docID
		return nil
	}))
	require.Equal(t, map[string]uint64{"alpha": 12, "gamma": 32}, visited)
	require.ErrorIs(t, readOnly.Checkpoint(ctx, filepath.Join(root, "forbidden.pebble")), ErrReadOnly)
	require.NoError(t, readOnly.Close())
	require.Equal(t, before, snapshotFileTree(t, checkpoint))
}

func TestPrimaryKeyMapRejectsMalformedPebbleValue(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	working := filepath.Join(root, "working.pebble")
	checkpoint := filepath.Join(root, "checkpoint.pebble")
	primary, err := CreatePrimaryKeyMap(ctx, working)
	require.NoError(t, err)
	require.NoError(t, primary.Checkpoint(ctx, checkpoint))
	require.NoError(t, primary.Close())

	database, err := pebble.Open(checkpoint, &pebble.Options{FormatMajorVersion: pebble.FormatNewest})
	require.NoError(t, err)
	require.NoError(t, database.Set([]byte("bad"), []byte{1}, pebble.Sync))
	require.NoError(t, database.Flush())
	require.NoError(t, database.Close())

	_, err = OpenPrimaryKeyMapReadOnly(ctx, checkpoint)
	require.ErrorIs(t, err, ErrIDMapCorrupt)
}

func TestPrimaryKeyMapLifecycleFailuresPreserveDurableState(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	working := filepath.Join(root, "working.pebble")
	checkpoint := filepath.Join(root, "checkpoint.pebble")
	primary, err := CreatePrimaryKeyMap(ctx, working)
	require.NoError(t, err)
	_, _, err = primary.Put(ctx, "alpha", 10)
	require.NoError(t, err)
	_, _, err = primary.Put(ctx, "beta", 20)
	require.NoError(t, err)

	docIDs, found, err := primary.MultiGet([]string{"beta", "missing", "alpha"})
	require.NoError(t, err)
	require.Equal(t, []uint64{20, 0, 10}, docIDs)
	require.Equal(t, []bool{true, false, true}, found)

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	canceledCheckpoint := filepath.Join(root, "canceled.pebble")
	require.ErrorIs(t, primary.Checkpoint(canceled, canceledCheckpoint), context.Canceled)
	_, err = os.Stat(canceledCheckpoint)
	require.ErrorIs(t, err, os.ErrNotExist)

	require.NoError(t, primary.Checkpoint(ctx, checkpoint))
	checkpointTree := snapshotFileTree(t, checkpoint)
	require.ErrorIs(t, primary.Checkpoint(ctx, checkpoint), os.ErrExist)
	require.Equal(t, checkpointTree, snapshotFileTree(t, checkpoint))

	existingWorking := filepath.Join(root, "existing.pebble")
	require.NoError(t, os.Mkdir(existingWorking, 0o700))
	marker := filepath.Join(existingWorking, "marker")
	require.NoError(t, os.WriteFile(marker, []byte("keep"), 0o600))
	_, err = OpenPrimaryKeyMap(ctx, checkpoint, existingWorking)
	require.ErrorIs(t, err, os.ErrExist)
	require.FileExists(t, marker)

	missingWorking := filepath.Join(root, "missing-working.pebble")
	_, err = OpenPrimaryKeyMap(ctx, filepath.Join(root, "missing-checkpoint.pebble"), missingWorking)
	require.Error(t, err)
	_, statErr := os.Stat(missingWorking)
	require.ErrorIs(t, statErr, os.ErrNotExist)

	reopened, err := OpenPrimaryKeyMap(ctx, checkpoint, filepath.Join(root, "reopened.pebble"))
	require.NoError(t, err)
	docIDs, found, err = reopened.MultiGet([]string{"alpha", "beta"})
	require.NoError(t, err)
	require.Equal(t, []uint64{10, 20}, docIDs)
	require.Equal(t, []bool{true, true}, found)
	require.NoError(t, reopened.Close())
	require.NoError(t, reopened.Close())
	_, _, err = reopened.Get("alpha")
	require.Error(t, err)
	_, _, err = reopened.Put(ctx, "gamma", 30)
	require.Error(t, err)
	_, _, err = reopened.Delete(ctx, "alpha")
	require.Error(t, err)
	closedCheckpoint := filepath.Join(root, "closed.pebble")
	require.Error(t, reopened.Checkpoint(ctx, closedCheckpoint))
	_, err = os.Stat(closedCheckpoint)
	require.ErrorIs(t, err, os.ErrNotExist)

	visitorErr := errors.New("stop iteration")
	require.ErrorIs(t, primary.forEach(ctx, func(string, uint64) error { return visitorErr }), visitorErr)
	require.ErrorIs(t, primary.forEach(canceled, func(string, uint64) error { return nil }), context.Canceled)
	require.NoError(t, primary.db.Set([]byte("beta"), []byte{1}, pebble.NoSync))
	docIDs, found, err = primary.MultiGet([]string{"alpha", "beta", "later"})
	require.ErrorIs(t, err, ErrIDMapCorrupt)
	require.Nil(t, docIDs)
	require.Nil(t, found)
	require.ErrorIs(t, primary.forEach(ctx, func(string, uint64) error { return nil }), ErrIDMapCorrupt)
	require.NoError(t, primary.Close())
}

func TestPrimaryKeyMapEnforcesKeyEncodingBoundariesWithoutMutation(t *testing.T) {
	ctx := context.Background()
	primary := NewPrimaryKeyMap()
	t.Cleanup(func() { require.NoError(t, primary.Close()) })
	maximum := strings.Repeat("k", maxPrimaryKeyBytes)
	_, found, err := primary.Put(ctx, maximum, 1)
	require.NoError(t, err)
	require.False(t, found)
	require.Equal(t, 1, primary.Count())

	_, _, err = primary.Put(ctx, maximum+"k", 2)
	require.Error(t, err)
	_, _, err = primary.Put(ctx, string([]byte{0xff}), 3)
	require.Error(t, err)
	require.Equal(t, 1, primary.Count())
	docID, found, err := primary.Get(maximum)
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(1), docID)
}

func TestOpenPrimaryKeyMapRejectsCorruptCheckpointWithoutWorkingState(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	primary, err := CreatePrimaryKeyMap(ctx, filepath.Join(root, "source.pebble"))
	require.NoError(t, err)
	checkpoint := filepath.Join(root, "checkpoint.pebble")
	require.NoError(t, primary.Checkpoint(ctx, checkpoint))
	require.NoError(t, primary.Close())

	database, err := pebble.Open(checkpoint, &pebble.Options{FormatMajorVersion: pebble.FormatNewest})
	require.NoError(t, err)
	require.NoError(t, database.Set([]byte("broken"), []byte{1}, pebble.Sync))
	require.NoError(t, database.Close())

	working := filepath.Join(root, "working.pebble")
	_, err = OpenPrimaryKeyMap(ctx, checkpoint, working)
	require.ErrorIs(t, err, ErrIDMapCorrupt)
	_, statErr := os.Stat(working)
	require.ErrorIs(t, statErr, os.ErrNotExist)
}

func TestManifestOwnsPebbleIDMapArtifact(t *testing.T) {
	manifest := sampleManifest(1)
	manifest.IDMap = "idmap/idmap-00000000000000000001.pebble"
	require.NoError(t, manifest.Validate())
	encoded, err := MarshalManifest(manifest)
	require.NoError(t, err)
	require.Contains(t, string(encoded), `"id_map":"idmap/idmap-00000000000000000001.pebble"`)
}

func TestCollectionPublishesOnlyCheckpointNamedByCurrent(t *testing.T) {
	ctx := context.Background()
	dir := filepath.Join(t.TempDir(), "collection")
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 2})
	require.NoError(t, err)
	_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "a"}, {PrimaryKey: "b"}})
	require.NoError(t, err)
	require.NoError(t, store.Flush(ctx))
	manifest := store.Manifest()
	require.Equal(t, uint32(3), manifest.FormatVersion)
	require.Equal(t, idMapCheckpointName(manifest.Generation), manifest.IDMap)
	info, err := os.Stat(collectionPath(dir, manifest.IDMap))
	require.NoError(t, err)
	require.True(t, info.IsDir())
	require.NoError(t, store.Close())
}
