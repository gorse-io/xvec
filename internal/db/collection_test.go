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
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gofrs/flock"
	"github.com/stretchr/testify/require"
)

var testCollectionSchema = json.RawMessage(`{"name":"books","fields":[]}`)

func TestCollectionPublishSegmentIndexSnapshotsAndPrune(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 4})
	require.NoError(t, err)
	_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "a", Payload: []byte("a")}})
	require.NoError(t, err)
	require.NoError(t, store.Flush(ctx))

	indexDirectory := filepath.Join(dir, "indexes")
	require.NoError(t, os.MkdirAll(indexDirectory, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(indexDirectory, "segment.zvi"), []byte("segment"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(indexDirectory, "obsolete.zvi"), []byte("obsolete"), 0o600))
	require.NoError(t, os.Mkdir(filepath.Join(indexDirectory, "segment.pebble"), 0o700))
	require.NoError(t, os.Mkdir(filepath.Join(indexDirectory, "obsolete.pebble"), 0o700))
	segment := store.Manifest().PersistedSegments[0]
	snapshots := []SegmentIndexSnapshotMetadata{{
		SegmentID: segment.ID, SchemaSHA256: strings.Repeat("a", 64),
		DocumentCount: segment.DocCount, MinDocumentID: segment.MinDocID, MaxDocumentID: segment.MaxDocID,
		Artifacts: []IndexArtifactMetadata{
			{Field: "embedding", Kind: "vector-2", File: "indexes/segment.zvi"},
			{Field: "rating", Kind: "invert", File: "indexes/segment.pebble"},
		},
	}}
	committed, err := store.PublishSegmentIndexSnapshots(ctx, snapshots)
	require.NoError(t, err)
	require.True(t, committed)
	require.Equal(t, snapshots, store.Manifest().SegmentIndexSnapshots)
	snapshots[0].Artifacts[0].File = "changed"
	require.Equal(t, "indexes/segment.zvi", store.Manifest().SegmentIndexSnapshots[0].Artifacts[0].File)

	committed, err = store.PublishSegmentIndexSnapshots(ctx, store.Manifest().SegmentIndexSnapshots)
	require.NoError(t, err)
	require.False(t, committed)
	require.NoError(t, store.PruneObsoleteArtifacts(ctx))
	_, err = os.Stat(filepath.Join(indexDirectory, "obsolete.zvi"))
	require.ErrorIs(t, err, os.ErrNotExist)
	_, err = os.Stat(filepath.Join(indexDirectory, "segment.zvi"))
	require.NoError(t, err)
	_, err = os.Stat(filepath.Join(indexDirectory, "obsolete.pebble"))
	require.ErrorIs(t, err, os.ErrNotExist)
	info, err := os.Stat(filepath.Join(indexDirectory, "segment.pebble"))
	require.NoError(t, err)
	require.True(t, info.IsDir())
	require.NoError(t, store.Close())
}

func TestCollectionCreateRecoverFlushAndContinue(t *testing.T) {
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 4})
	require.NoError(t, err)
	{
		manifest := store.Manifest()
		require.True(t, manifest.Generation == 1)
		require.True(t, manifest.SegmentMaxDocuments == 4)
		require.True(t, manifest.WritingSegment.ID == 0)
	}
	{
		_, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "a", Payload: []byte("a1")}, {PrimaryKey: "b", Payload: []byte("b1")}})
		require.NoError(t, err)
	}
	{
		_, err := store.Upsert(context.Background(), []WriteInput{{PrimaryKey: "a", Payload: []byte("a2")}})
		require.NoError(t, err)
	}
	{
		_, err := store.Delete(context.Background(), []string{"b"})
		require.NoError(t, err)
	}
	{
		err := store.Close()
		require.NoError(t, err)
	}

	store, err = OpenCollection(context.Background(), dir, CollectionOptions{})
	require.NoError(t, err)

	results, err := store.Fetch(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.NotNil(t, results[0].Document)
	require.True(t, results[0].Document.DocID == 2)
	require.True(t, string(results[0].Document.Payload) == "a2")
	require.Nil(t, results[1].Document)

	inserted, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "c", Payload: []byte("c1")}})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 3)
	{
		err := store.Flush(context.Background())
		require.NoError(t, err)
	}

	manifest := store.Manifest()
	require.True(t, manifest.Generation == 2)
	require.Len(t, manifest.PersistedSegments, 1)
	require.True(t, manifest.PersistedSegments[0].DocCount == 4)
	require.True(t, manifest.WritingSegment.ID == 1)
	require.True(t, manifest.NextSegmentID == 2)
	require.Equal(t, idMapCheckpointName(2), manifest.IDMap)
	require.True(t, manifest.DeleteSnapshotGeneration == 2)
	{
		err := store.Flush(context.Background())
		require.NoError(t, err)
	}
	{
		got := store.Manifest().Generation
		require.Equal(t, manifest.Generation, got)
	}

	second, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "d", Payload: []byte("d1")}})
	require.NoError(t, err)
	require.True(t, second[0].DocID == 4)
	{
		err := store.Flush(context.Background())
		require.NoError(t, err)
	}
	{
		err := store.Close()
		require.NoError(t, err)
	}

	readOnly, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)

	defer readOnly.Close()
	require.True(t, readOnly.ReadOnly())
	require.Len(t, readOnly.Manifest().PersistedSegments, 2)

	results, err = readOnly.Fetch(context.Background(), []string{"a", "c", "d"})
	require.NoError(t, err)
	require.NotNil(t, results[0].Document)
	require.NotNil(t, results[1].Document)
	require.NotNil(t, results[2].Document)
	{
		_, err := readOnly.Insert(context.Background(), []WriteInput{{PrimaryKey: "x"}})
		require.ErrorIs(t, err, ErrReadOnly)
	}
	{
		err := readOnly.Flush(context.Background())
		require.ErrorIs(t, err, ErrReadOnly)
	}
}

func TestCollectionAllowsManyReadersAndOneWriter(t *testing.T) {
	dir := t.TempDir()
	created, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)
	{
		err := created.Close()
		require.NoError(t, err)
	}

	first, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)

	second, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)

	deadline, cancel := context.WithTimeout(context.Background(), 75*time.Millisecond)
	defer cancel()
	{
		_, err := OpenCollection(deadline, dir, CollectionOptions{})
		require.ErrorIs(t, err, context.DeadlineExceeded)
	}
	{
		err := first.Close()
		require.NoError(t, err)
	}
	{
		err := second.Close()
		require.NoError(t, err)
	}

	writer, err := OpenCollection(context.Background(), dir, CollectionOptions{})
	require.NoError(t, err)

	deadline, cancel = context.WithTimeout(context.Background(), 75*time.Millisecond)
	defer cancel()
	{
		_, err := OpenCollection(deadline, dir, CollectionOptions{ReadOnly: true})
		require.ErrorIs(t, err, context.DeadlineExceeded)
	}
	{
		err := writer.Close()
		require.NoError(t, err)
	}
}

func TestCollectionReadOnlyOpenDoesNotMutateDirectory(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)
	_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "dirty", Payload: []byte("wal")}})
	require.NoError(t, err)
	require.NoError(t, store.Close())

	before := snapshotFileTree(t, dir)
	readOnly, err := OpenCollection(ctx, dir, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)
	results, err := readOnly.Fetch(ctx, []string{"dirty"})
	require.NoError(t, err)
	require.NotNil(t, results[0].Document)
	require.NoError(t, readOnly.Close())
	require.Equal(t, before, snapshotFileTree(t, dir))
}

func TestCollectionIDMapApplyFailurePoisonsUntilReopen(t *testing.T) {
	t.Run("put", func(t *testing.T) {
		ctx := context.Background()
		dir := t.TempDir()
		store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
		require.NoError(t, err)
		injected := errors.New("injected IDMap put failure")
		store.manager.PrimaryKeys().setPoint = func(_, _ []byte) error { return injected }

		_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "recover", Payload: []byte("from-wal")}})
		require.ErrorIs(t, err, injected)
		require.ErrorIs(t, err, ErrWriteEnginePoisoned)
		_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "rejected"}})
		require.ErrorIs(t, err, ErrWriteEnginePoisoned)
		require.ErrorIs(t, store.Flush(ctx), ErrWriteEnginePoisoned)
		require.NoError(t, store.Close())

		reopened, err := OpenCollection(ctx, dir, CollectionOptions{})
		require.NoError(t, err)
		defer reopened.Close()
		results, err := reopened.Fetch(ctx, []string{"recover", "rejected"})
		require.NoError(t, err)
		require.NotNil(t, results[0].Document)
		require.Equal(t, "from-wal", string(results[0].Document.Payload))
		require.Nil(t, results[1].Document)
	})

	t.Run("delete", func(t *testing.T) {
		ctx := context.Background()
		dir := t.TempDir()
		store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
		require.NoError(t, err)
		_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "deleted"}})
		require.NoError(t, err)
		injected := errors.New("injected IDMap delete failure")
		store.manager.PrimaryKeys().deletePoint = func(_ []byte) error { return injected }

		_, err = store.Delete(ctx, []string{"deleted"})
		require.ErrorIs(t, err, injected)
		require.ErrorIs(t, err, ErrWriteEnginePoisoned)
		require.NoError(t, store.Close())

		reopened, err := OpenCollection(ctx, dir, CollectionOptions{})
		require.NoError(t, err)
		defer reopened.Close()
		results, err := reopened.Fetch(ctx, []string{"deleted"})
		require.NoError(t, err)
		require.Nil(t, results[0].Document)
	})
}

func TestCollectionPruneIDMapDirectoriesSafely(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)
	activeWorking := store.idMapWorking
	staleWorking := filepath.Join(dir, "idmap", ".working-999-999.pebble")
	require.NoError(t, os.Mkdir(staleWorking, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(staleWorking, "stale"), []byte("stale"), 0o600))
	require.NoError(t, store.PruneObsoleteArtifacts(ctx))
	_, err = os.Stat(activeWorking)
	require.NoError(t, err)
	_, err = os.Stat(staleWorking)
	require.ErrorIs(t, err, os.ErrNotExist)

	target := filepath.Join(t.TempDir(), "outside")
	require.NoError(t, os.Mkdir(target, 0o700))
	symlink := filepath.Join(dir, "idmap", ".working-998-998.pebble")
	if err := os.Symlink(target, symlink); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	require.Error(t, store.PruneObsoleteArtifacts(ctx))
	_, err = os.Stat(target)
	require.NoError(t, err)
	require.NoError(t, os.Remove(symlink))
	require.NoError(t, store.Close())
}

func TestCollectionOpenRejectsSymlinkedIDMapRoot(t *testing.T) {
	ctx := context.Background()
	dir, _ := createClosedCollection(t, false)
	idMapRoot := filepath.Join(dir, "idmap")
	relocated := filepath.Join(t.TempDir(), "relocated-idmap")
	require.NoError(t, os.Rename(idMapRoot, relocated))
	if err := os.Symlink(relocated, idMapRoot); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	_, err := OpenCollection(ctx, dir, CollectionOptions{})
	require.ErrorIs(t, err, ErrCollectionCorrupt)
}

func TestCollectionFailedFlushLeavesPublishedStateAndWriterUsable(t *testing.T) {
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)

	defer store.Close()
	{
		_, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}})
		require.NoError(t, err)
	}

	versionLock := flock.New(filepath.Join(dir, versionLockName))
	locked, err := versionLock.TryLock()
	require.NoError(t, err)
	require.True(t, locked)

	deadline, cancel := context.WithTimeout(context.Background(), 75*time.Millisecond)
	err = store.Flush(deadline)
	cancel()
	require.ErrorIs(t, err, context.DeadlineExceeded)
	{
		err := versionLock.Close()
		require.NoError(t, err)
	}
	{
		got := store.Manifest().Generation
		require.True(t, got == 1)
	}

	result, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "two"}})
	require.NoError(t, err)
	require.True(t, result[0].DocID == 1)
	{
		err := store.Flush(context.Background())
		require.NoError(t, err)
	}
}

func TestCollectionRewriteDocumentsIsAtomicAndRecoverable(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 8})
	require.NoError(t, err)

	inserted, err := store.Insert(ctx, []WriteInput{
		{PrimaryKey: "a", Payload: []byte("a1")},
		{PrimaryKey: "b", Payload: []byte("b1")},
		{PrimaryKey: "c", Payload: []byte("c1")},
	})
	require.NoError(t, err)

	updated, err := store.Upsert(ctx, []WriteInput{{PrimaryKey: "b", Payload: []byte("b2")}})
	require.NoError(t, err)
	{
		_, err := store.Delete(ctx, []string{"c"})
		require.NoError(t, err)
	}

	ephemeral, err := store.Insert(ctx, []WriteInput{{PrimaryKey: "ephemeral", Payload: []byte("gone")}})
	require.NoError(t, err)
	require.True(t, ephemeral[0].DocID == 4)
	{
		_, err := store.Delete(ctx, []string{"ephemeral"})
		require.NoError(t, err)
	}

	live, err := store.LiveDocuments(ctx)
	require.NoError(t, err)
	require.Len(t, live, 2)
	require.Equal(t, inserted[0].DocID, live[0].DocID)
	require.Equal(t, updated[0].DocID, live[1].DocID)

	rewritten := cloneDocuments(live)
	for index := range rewritten {
		rewritten[index].Payload = append([]byte("rewritten-"), rewritten[index].Payload...)
	}
	nextSchema := json.RawMessage(`{"name":"books-v2","fields":[]}`)

	versionLock := flock.New(filepath.Join(dir, versionLockName))
	locked, err := versionLock.TryLock()
	require.NoError(t, err)
	require.True(t, locked)

	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	committed, err := store.RewriteDocuments(deadline, nextSchema, rewritten)
	cancel()
	require.False(t, committed)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	{
		err := versionLock.Close()
		require.NoError(t, err)
	}
	{
		manifest := store.Manifest()
		require.True(t, manifest.Generation == 1)
		require.Equal(t, string(testCollectionSchema), string(manifest.Schema))
	}

	for pattern, want := range map[string]int{
		filepath.Join(dir, "segments", "*", "*.seg"): 0,
		filepath.Join(dir, "snapshots", "*.snap"):    1,
		filepath.Join(dir, "wal", "*.wal"):           1,
	} {
		files, globErr := filepath.Glob(pattern)
		require.NoError(t, globErr)
		require.Len(t, files, want)
	}
	results, err := store.Fetch(ctx, []string{"a", "b", "c"})
	require.NoError(t, err)
	require.True(t, string(results[0].Document.Payload) == "a1")
	require.True(t, string(results[1].Document.Payload) == "b2")
	require.Nil(t, results[2].Document)

	committed, err = store.RewriteDocuments(ctx, nextSchema, rewritten)
	require.NoError(t, err)
	require.True(t, committed)

	manifest := store.Manifest()
	require.True(t, manifest.Generation == 2)
	require.Equal(t, string(nextSchema), string(manifest.Schema))
	require.Len(t, manifest.PersistedSegments, 2)
	require.Equal(t, inserted[0].DocID, manifest.PersistedSegments[0].MinDocID)
	require.Equal(t, updated[0].DocID, manifest.PersistedSegments[1].MinDocID)

	results, err = store.Fetch(ctx, []string{"a", "b", "c"})
	require.NoError(t, err)
	require.Equal(t, inserted[0].DocID, results[0].Document.DocID)
	require.Equal(t, updated[0].DocID, results[1].Document.DocID)
	require.True(t, string(results[0].Document.Payload) == "rewritten-a1")
	require.True(t, string(results[1].Document.Payload) == "rewritten-b2")
	require.Nil(t, results[2].Document)

	continued, err := store.Insert(ctx, []WriteInput{{PrimaryKey: "d", Payload: []byte("d1")}})
	require.NoError(t, err)
	require.True(t, continued[0].DocID == 5)
	{
		err := store.Close()
		require.NoError(t, err)
	}

	store, err = OpenCollection(ctx, dir, CollectionOptions{})
	require.NoError(t, err)

	defer store.Close()
	require.Equal(t, string(nextSchema), string(store.Manifest().Schema))

	results, err = store.Fetch(ctx, []string{"a", "b", "c", "d"})
	require.NoError(t, err)
	require.Equal(t, inserted[0].DocID, results[0].Document.DocID)
	require.Equal(t, updated[0].DocID, results[1].Document.DocID)
	require.Nil(t, results[2].Document)
	require.NotNil(t, results[3].Document)
	require.True(t, results[3].Document.DocID == 5)
}

func TestCollectionRewriteRejectsStaleSnapshot(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)

	defer store.Close()
	{
		_, err := store.Insert(ctx, []WriteInput{{PrimaryKey: "a", Payload: []byte("a1")}})
		require.NoError(t, err)
	}

	live, err := store.LiveDocuments(ctx)
	require.NoError(t, err)

	live[0].PrimaryKey = "different"
	{
		committed, err := store.RewriteDocuments(ctx, testCollectionSchema, live)
		require.False(t, committed)
		require.Error(t, err)
	}
	{
		got := store.Manifest().Generation
		require.True(t, got == 1)
	}
}

func TestRewriteDocumentRunsHonorsGapsAndCapacity(t *testing.T) {
	documents := []StoredDocument{
		{DocID: 0}, {DocID: 1}, {DocID: 2},
		{DocID: 4}, {DocID: 5}, {DocID: 8},
	}
	runs := rewriteDocumentRuns(documents, 2)
	want := [][]uint64{{0, 1}, {2}, {4, 5}, {8}}
	require.Len(t, runs, len(want))

	for runIndex := range runs {
		require.Len(t, runs[runIndex], len(want[runIndex]))

		for documentIndex := range runs[runIndex] {
			require.Equal(t, want[runIndex][documentIndex], runs[runIndex][documentIndex].DocID)
		}
	}
	{
		runs := rewriteDocumentRuns(nil, 2)
		require.Nil(t, runs)
	}
}

func TestCollectionReadOnlyRecoveryDoesNotRepairWAL(t *testing.T) {
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)
	{
		_, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}})
		require.NoError(t, err)
	}

	manifest := store.Manifest()
	{
		err := store.Close()
		require.NoError(t, err)
	}

	walPath := collectionPath(dir, manifest.WritingSegment.Files[0])
	file, err := os.OpenFile(walPath, os.O_APPEND|os.O_WRONLY, 0)
	require.NoError(t, err)

	partial := encodeWALRecord(2, []byte("partial"))[:walRecordHeaderSize+3]
	{
		_, err := file.Write(partial)
		require.NoError(t, err)
	}
	{
		err := file.Close()
		require.NoError(t, err)
	}

	withTail := fileSize(t, walPath)
	readOnly, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)
	{
		err := readOnly.Close()
		require.NoError(t, err)
	}
	{
		got := fileSize(t, walPath)
		require.Equal(t, withTail, got)
	}

	writer, err := OpenCollection(context.Background(), dir, CollectionOptions{})
	require.NoError(t, err)
	{
		err := writer.Close()
		require.NoError(t, err)
	}
	{
		got := fileSize(t, walPath)
		require.Equal(t, withTail-int64(len(partial)), got)
	}
}

func TestCollectionRecoveryRejectsDamagedReferencedFiles(t *testing.T) {
	t.Run("missing IDMap checkpoint", func(t *testing.T) {
		dir, manifest := createClosedCollection(t, false)
		{
			err := os.RemoveAll(collectionPath(dir, manifest.IDMap))
			require.NoError(t, err)
		}
		{
			_, err := OpenCollection(context.Background(), dir, CollectionOptions{})
			require.ErrorIs(t, err, ErrCollectionCorrupt)
		}
	})
	t.Run("corrupt segment", func(t *testing.T) {
		dir, manifest := createClosedCollection(t, true)
		name := collectionPath(dir, manifest.PersistedSegments[0].Files[0])
		contents, err := os.ReadFile(name)
		require.NoError(t, err)

		contents[len(contents)-1] ^= 1
		{
			err := os.WriteFile(name, contents, 0o600)
			require.NoError(t, err)
		}
		{
			_, err := OpenCollection(context.Background(), dir, CollectionOptions{})
			require.ErrorIs(t, err, ErrCollectionCorrupt)
		}
	})
	t.Run("corrupt WAL operation", func(t *testing.T) {
		dir, manifest := createClosedCollection(t, false)
		name := collectionPath(dir, manifest.WritingSegment.Files[0])
		contents, err := os.ReadFile(name)
		require.NoError(t, err)

		contents[len(contents)-1] ^= 1
		{
			err := os.WriteFile(name, contents, 0o600)
			require.NoError(t, err)
		}
		{
			_, err := OpenCollection(context.Background(), dir, CollectionOptions{})
			require.ErrorIs(t, err, ErrCollectionCorrupt)
		}
	})
}

func TestCollectionIgnoresUnpublishedArtifacts(t *testing.T) {
	dir, current := createClosedCollection(t, false)
	orphan := current.Clone()
	orphan.Generation = current.Generation + 100
	orphan.Schema = json.RawMessage(`{"name":"orphan"}`)
	encoded, err := MarshalManifest(orphan)
	require.NoError(t, err)
	{
		err := os.WriteFile(filepath.Join(dir, manifestFileName(orphan.Generation)), encoded, 0o600)
		require.NoError(t, err)
	}

	opened, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)

	defer opened.Close()
	{
		got := opened.Manifest()
		require.Equal(t, current.Generation, got.Generation)
		require.Equal(t, string(testCollectionSchema), string(got.Schema))
	}
}

func TestCollectionRecoversAfterProcessExit(t *testing.T) {
	if dir := os.Getenv("ZVEC_TEST_CRASH_DIR"); dir != "" {
		store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
		if err != nil {
			os.Exit(91)
		}
		if _, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "survivor", Payload: []byte("durable")}}); err != nil {
			os.Exit(92)
		}
		os.Exit(93)
	}
	dir := filepath.Join(t.TempDir(), "collection")
	command := exec.Command(os.Args[0], "-test.run=^TestCollectionRecoversAfterProcessExit$")
	command.Env = append(os.Environ(), "ZVEC_TEST_CRASH_DIR="+dir)
	err := command.Run()
	var exitError *exec.ExitError
	require.ErrorAs(t, err, &exitError)
	require.True(t, exitError.ExitCode() == 93)

	opened, err := OpenCollection(context.Background(), dir, CollectionOptions{})
	require.NoError(t, err)

	defer opened.Close()
	results, err := opened.Fetch(context.Background(), []string{"survivor"})
	require.NoError(t, err)
	require.NotNil(t, results[0].Document)
	require.True(t, string(results[0].Document.Payload) == "durable")
}

func TestCollectionArgumentsAndClose(t *testing.T) {
	var nilStore *CollectionStore
	{
		_, err := nilStore.Insert(context.Background(), []WriteInput{{PrimaryKey: "x"}})
		require.Error(t, err,
			"nil collection insert succeeded")
	}
	{
		err := nilStore.Flush(context.Background())
		require.Error(t, err,
			"nil collection flush succeeded")
	}
	{
		_, err := CreateCollection(nil, t.TempDir(), testCollectionSchema, CollectionOptions{})
		require.Error(t, err,
			"nil create context succeeded")
	}
	{
		_, err := CreateCollection(context.Background(), t.TempDir(), json.RawMessage(`[]`), CollectionOptions{})
		require.Error(t, err,
			"non-object schema succeeded")
	}
	{
		_, err := CreateCollection(context.Background(), t.TempDir(), testCollectionSchema, CollectionOptions{ReadOnly: true})
		require.ErrorIs(t, err, ErrReadOnly)
	}
	{
		_, err := OpenCollection(nil, t.TempDir(), CollectionOptions{})
		require.Error(t, err,
			"nil open context succeeded")
	}
	{
		_, err := OpenCollection(context.Background(), filepath.Join(t.TempDir(), "missing"), CollectionOptions{})
		require.ErrorIs(t, err, ErrManifestNotFound)
	}

	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)

	deadline, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	{
		_, err := CreateCollection(deadline, dir, testCollectionSchema, CollectionOptions{})
		require.ErrorIs(t, err, context.DeadlineExceeded)
	}
	{
		err := store.Close()
		require.NoError(t, err)
	}
	{
		err := store.Close()
		require.NoError(t, err)
	}
	{
		_, err := store.Fetch(context.Background(), []string{"x"})
		require.ErrorIs(t, err, ErrCollectionClosed)
	}
	{
		err := store.Flush(context.Background())
		require.ErrorIs(t, err, ErrCollectionClosed)
	}
}

func TestCollectionPublishSchemaIsAtomicAndDurable(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)

	initial := store.Manifest()
	_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "dirty-schema", Payload: []byte("wal")}})
	require.NoError(t, err)
	{
		committed, err := store.PublishSchema(ctx, json.RawMessage(`[`))
		require.Error(t, err)
		require.False(t, committed)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		committed, err := store.PublishSchema(canceled, json.RawMessage(`{"name":"canceled"}`))
		require.ErrorIs(t, err, context.Canceled)
		require.False(t, committed)
	}
	{
		current := store.Manifest()
		require.Equal(t, initial.Generation, current.Generation)
		require.Equal(t, string(initial.Schema), string(current.Schema))
	}

	updatedSchema := json.RawMessage(`{"name":"articles","fields":[]}`)
	committed, err := store.PublishSchema(ctx, updatedSchema)
	require.NoError(t, err)
	require.True(t, committed)

	updated := store.Manifest()
	require.True(t, updated.Generation > initial.Generation)
	require.Equal(t, string(updatedSchema), string(updated.Schema))
	require.Equal(t, initial.IDMap, updated.IDMap)
	require.NoError(t, store.PruneObsoleteArtifacts(ctx))
	_, err = os.Stat(collectionPath(dir, updated.IDMap))
	require.NoError(t, err)
	{
		committed, err := store.PublishSchema(ctx, updatedSchema)
		require.NoError(t, err)
		require.False(t, committed)
	}
	require.Equal(t, updated.Generation, store.Manifest().Generation,
		"idempotent publication advanced generation")
	{
		err := store.Close()
		require.NoError(t, err)
	}
	{
		committed, err := store.PublishSchema(ctx, testCollectionSchema)
		require.ErrorIs(t, err, ErrCollectionClosed)
		require.False(t, committed)
	}

	readOnly, err := OpenCollection(ctx, dir, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)

	defer readOnly.Close()
	{
		got := readOnly.Manifest()
		require.Equal(t, string(updatedSchema), string(got.Schema))
	}
	results, err := readOnly.Fetch(ctx, []string{"dirty-schema"})
	require.NoError(t, err)
	require.NotNil(t, results[0].Document)
	{
		committed, err := readOnly.PublishSchema(ctx, testCollectionSchema)
		require.ErrorIs(t, err, ErrReadOnly)
		require.False(t, committed)
	}

	var nilStore *CollectionStore
	{
		committed, err := nilStore.PublishSchema(ctx, testCollectionSchema)
		require.Error(t, err)
		require.False(t, committed)
	}
}

func createClosedCollection(t *testing.T, flush bool) (string, Manifest) {
	t.Helper()
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	require.NoError(t, err)
	{
		_, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "one", Payload: []byte("payload")}})
		require.NoError(t, err)
	}

	if flush {
		{
			err := store.Flush(context.Background())
			require.NoError(t, err)
		}
	}
	manifest := store.Manifest()
	{
		err := store.Close()
		require.NoError(t, err)
	}

	return dir, manifest
}

func snapshotFileTree(t *testing.T, root string) map[string]string {
	t.Helper()
	result := make(map[string]string)
	require.NoError(t, filepath.WalkDir(root, func(name string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(root, name)
		if err != nil {
			return err
		}
		if entry.IsDir() {
			result[relative] = "directory"
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(name)
			if err != nil {
				return err
			}
			result[relative] = "symlink:" + target
			return nil
		}
		contents, err := os.ReadFile(name)
		if err != nil {
			return err
		}
		result[relative] = string(contents)
		return nil
	}))
	return result
}
