// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
package db

import (
	"context"
	"os"
	"path/filepath"
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
	_, _, err = primary.Put(ctx, "base", 10)
	require.NoError(t, err)
	require.NoError(t, primary.Checkpoint(ctx, checkpoint))
	require.NoError(t, primary.Close())

	before := snapshotFileTree(t, checkpoint)
	readOnly, err := OpenPrimaryKeyMapReadOnly(ctx, checkpoint)
	require.NoError(t, err)
	_, _, err = readOnly.Put(ctx, "replayed", 11)
	require.NoError(t, err)
	_, _, err = readOnly.Delete(ctx, "base")
	require.NoError(t, err)
	require.Equal(t, 1, readOnly.Count())
	docID, found, err := readOnly.Get("replayed")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, uint64(11), docID)
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
