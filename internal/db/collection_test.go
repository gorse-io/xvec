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
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
)

var testCollectionSchema = json.RawMessage(`{"name":"books","fields":[]}`)

func TestCollectionCreateRecoverFlushAndContinue(t *testing.T) {
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 4})
	if err != nil {
		t.Fatal(err)
	}
	if manifest := store.Manifest(); manifest.Generation != 1 || manifest.SegmentMaxDocuments != 4 || manifest.WritingSegment.ID != 0 {
		t.Fatalf("initial manifest = %#v", manifest)
	}
	if _, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "a", Payload: []byte("a1")}, {PrimaryKey: "b", Payload: []byte("b1")}}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Upsert(context.Background(), []WriteInput{{PrimaryKey: "a", Payload: []byte("a2")}}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Delete(context.Background(), []string{"b"}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenCollection(context.Background(), dir, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	results, err := store.Fetch(context.Background(), []string{"a", "b"})
	if err != nil || results[0].Document == nil || results[0].Document.DocID != 2 || string(results[0].Document.Payload) != "a2" || results[1].Document != nil {
		t.Fatalf("recovered fetch = %#v, %v", results, err)
	}
	inserted, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "c", Payload: []byte("c1")}})
	if err != nil || inserted[0].DocID != 3 {
		t.Fatalf("continued insert = %#v, %v", inserted, err)
	}
	if err := store.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	manifest := store.Manifest()
	if manifest.Generation != 2 || len(manifest.PersistedSegments) != 1 || manifest.PersistedSegments[0].DocCount != 4 || manifest.WritingSegment.ID != 1 || manifest.NextSegmentID != 2 {
		t.Fatalf("flushed manifest = %#v", manifest)
	}
	if manifest.IDMapGeneration != 2 || manifest.DeleteSnapshotGeneration != 2 {
		t.Fatalf("snapshot generations = %d/%d", manifest.IDMapGeneration, manifest.DeleteSnapshotGeneration)
	}
	if err := store.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := store.Manifest().Generation; got != manifest.Generation {
		t.Fatalf("empty flush advanced generation to %d", got)
	}
	second, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "d", Payload: []byte("d1")}})
	if err != nil || second[0].DocID != 4 {
		t.Fatalf("post-flush insert = %#v, %v", second, err)
	}
	if err := store.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	readOnly, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	defer readOnly.Close()
	if !readOnly.ReadOnly() || len(readOnly.Manifest().PersistedSegments) != 2 {
		t.Fatalf("read-only state = %v, %#v", readOnly.ReadOnly(), readOnly.Manifest())
	}
	results, err = readOnly.Fetch(context.Background(), []string{"a", "c", "d"})
	if err != nil || results[0].Document == nil || results[1].Document == nil || results[2].Document == nil {
		t.Fatalf("read-only fetch = %#v, %v", results, err)
	}
	if _, err := readOnly.Insert(context.Background(), []WriteInput{{PrimaryKey: "x"}}); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("read-only insert error = %v", err)
	}
	if err := readOnly.Flush(context.Background()); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("read-only flush error = %v", err)
	}
}

func TestCollectionAllowsManyReadersAndOneWriter(t *testing.T) {
	dir := t.TempDir()
	created, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := created.Close(); err != nil {
		t.Fatal(err)
	}
	first, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	second, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	deadline, cancel := context.WithTimeout(context.Background(), 75*time.Millisecond)
	defer cancel()
	if _, err := OpenCollection(deadline, dir, CollectionOptions{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("writer while readers are open = %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
	writer, err := OpenCollection(context.Background(), dir, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	deadline, cancel = context.WithTimeout(context.Background(), 75*time.Millisecond)
	defer cancel()
	if _, err := OpenCollection(deadline, dir, CollectionOptions{ReadOnly: true}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("reader while writer is open = %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCollectionFailedFlushLeavesPublishedStateAndWriterUsable(t *testing.T) {
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}}); err != nil {
		t.Fatal(err)
	}
	versionLock, err := ailego.AcquireFileLock(context.Background(), filepath.Join(dir, versionLockName), ailego.LockExclusive)
	if err != nil {
		t.Fatal(err)
	}
	deadline, cancel := context.WithTimeout(context.Background(), 75*time.Millisecond)
	err = store.Flush(deadline)
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("blocked flush error = %v", err)
	}
	if err := versionLock.Close(); err != nil {
		t.Fatal(err)
	}
	if got := store.Manifest().Generation; got != 1 {
		t.Fatalf("failed flush published generation %d", got)
	}
	result, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "two"}})
	if err != nil || result[0].DocID != 1 {
		t.Fatalf("insert after failed flush = %#v, %v", result, err)
	}
	if err := store.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestCollectionRewriteDocumentsIsAtomicAndRecoverable(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 8})
	if err != nil {
		t.Fatal(err)
	}
	inserted, err := store.Insert(ctx, []WriteInput{
		{PrimaryKey: "a", Payload: []byte("a1")},
		{PrimaryKey: "b", Payload: []byte("b1")},
		{PrimaryKey: "c", Payload: []byte("c1")},
	})
	if err != nil {
		t.Fatal(err)
	}
	updated, err := store.Upsert(ctx, []WriteInput{{PrimaryKey: "b", Payload: []byte("b2")}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Delete(ctx, []string{"c"}); err != nil {
		t.Fatal(err)
	}
	ephemeral, err := store.Insert(ctx, []WriteInput{{PrimaryKey: "ephemeral", Payload: []byte("gone")}})
	if err != nil || ephemeral[0].DocID != 4 {
		t.Fatalf("ephemeral insert = %#v, %v", ephemeral, err)
	}
	if _, err := store.Delete(ctx, []string{"ephemeral"}); err != nil {
		t.Fatal(err)
	}
	live, err := store.LiveDocuments(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(live) != 2 || live[0].DocID != inserted[0].DocID || live[1].DocID != updated[0].DocID {
		t.Fatalf("live documents = %#v", live)
	}
	rewritten := cloneDocuments(live)
	for index := range rewritten {
		rewritten[index].Payload = append([]byte("rewritten-"), rewritten[index].Payload...)
	}
	nextSchema := json.RawMessage(`{"name":"books-v2","fields":[]}`)

	versionLock, err := ailego.AcquireFileLock(ctx, filepath.Join(dir, versionLockName), ailego.LockExclusive)
	if err != nil {
		t.Fatal(err)
	}
	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	committed, err := store.RewriteDocuments(deadline, nextSchema, rewritten)
	cancel()
	if committed || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("blocked rewrite = committed %v, error %v", committed, err)
	}
	if err := versionLock.Close(); err != nil {
		t.Fatal(err)
	}
	if manifest := store.Manifest(); manifest.Generation != 1 || string(manifest.Schema) != string(testCollectionSchema) {
		t.Fatalf("manifest after failed rewrite = %#v", manifest)
	}
	for pattern, want := range map[string]int{
		filepath.Join(dir, "segments", "*", "*.seg"): 0,
		filepath.Join(dir, "snapshots", "*.snap"):    2,
		filepath.Join(dir, "wal", "*.wal"):           1,
	} {
		files, globErr := filepath.Glob(pattern)
		if globErr != nil || len(files) != want {
			t.Fatalf("artifacts %q = %v, %v; want %d files", pattern, files, globErr, want)
		}
	}
	results, err := store.Fetch(ctx, []string{"a", "b", "c"})
	if err != nil || string(results[0].Document.Payload) != "a1" || string(results[1].Document.Payload) != "b2" || results[2].Document != nil {
		t.Fatalf("fetch after failed rewrite = %#v, %v", results, err)
	}

	committed, err = store.RewriteDocuments(ctx, nextSchema, rewritten)
	if err != nil || !committed {
		t.Fatalf("rewrite = committed %v, error %v", committed, err)
	}
	manifest := store.Manifest()
	if manifest.Generation != 2 || string(manifest.Schema) != string(nextSchema) || len(manifest.PersistedSegments) != 2 {
		t.Fatalf("rewritten manifest = %#v", manifest)
	}
	if manifest.PersistedSegments[0].MinDocID != inserted[0].DocID || manifest.PersistedSegments[1].MinDocID != updated[0].DocID {
		t.Fatalf("rewritten ranges = %#v", manifest.PersistedSegments)
	}
	results, err = store.Fetch(ctx, []string{"a", "b", "c"})
	if err != nil || results[0].Document.DocID != inserted[0].DocID || results[1].Document.DocID != updated[0].DocID ||
		string(results[0].Document.Payload) != "rewritten-a1" || string(results[1].Document.Payload) != "rewritten-b2" || results[2].Document != nil {
		t.Fatalf("rewritten fetch = %#v, %v", results, err)
	}
	continued, err := store.Insert(ctx, []WriteInput{{PrimaryKey: "d", Payload: []byte("d1")}})
	if err != nil || continued[0].DocID != 5 {
		t.Fatalf("continued insert = %#v, %v", continued, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenCollection(ctx, dir, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if string(store.Manifest().Schema) != string(nextSchema) {
		t.Fatalf("reopened schema = %s", store.Manifest().Schema)
	}
	results, err = store.Fetch(ctx, []string{"a", "b", "c", "d"})
	if err != nil || results[0].Document.DocID != inserted[0].DocID || results[1].Document.DocID != updated[0].DocID ||
		results[2].Document != nil || results[3].Document == nil || results[3].Document.DocID != 5 {
		t.Fatalf("reopened fetch = %#v, %v", results, err)
	}
}

func TestCollectionRewriteRejectsStaleSnapshot(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Insert(ctx, []WriteInput{{PrimaryKey: "a", Payload: []byte("a1")}}); err != nil {
		t.Fatal(err)
	}
	live, err := store.LiveDocuments(ctx)
	if err != nil {
		t.Fatal(err)
	}
	live[0].PrimaryKey = "different"
	if committed, err := store.RewriteDocuments(ctx, testCollectionSchema, live); committed || err == nil {
		t.Fatalf("stale rewrite = committed %v, error %v", committed, err)
	}
	if got := store.Manifest().Generation; got != 1 {
		t.Fatalf("failed stale rewrite published generation %d", got)
	}
}

func TestRewriteDocumentRunsHonorsGapsAndCapacity(t *testing.T) {
	documents := []StoredDocument{
		{DocID: 0}, {DocID: 1}, {DocID: 2},
		{DocID: 4}, {DocID: 5}, {DocID: 8},
	}
	runs := rewriteDocumentRuns(documents, 2)
	want := [][]uint64{{0, 1}, {2}, {4, 5}, {8}}
	if len(runs) != len(want) {
		t.Fatalf("runs = %#v", runs)
	}
	for runIndex := range runs {
		if len(runs[runIndex]) != len(want[runIndex]) {
			t.Fatalf("run %d = %#v", runIndex, runs[runIndex])
		}
		for documentIndex := range runs[runIndex] {
			if runs[runIndex][documentIndex].DocID != want[runIndex][documentIndex] {
				t.Fatalf("run %d = %#v", runIndex, runs[runIndex])
			}
		}
	}
	if runs := rewriteDocumentRuns(nil, 2); runs != nil {
		t.Fatalf("empty runs = %#v", runs)
	}
}

func TestCollectionReadOnlyRecoveryDoesNotRepairWAL(t *testing.T) {
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "one"}}); err != nil {
		t.Fatal(err)
	}
	manifest := store.Manifest()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	walPath := collectionPath(dir, manifest.WritingSegment.Files[0])
	file, err := os.OpenFile(walPath, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	partial := encodeWALRecord(2, []byte("partial"))[:walRecordHeaderSize+3]
	if _, err := file.Write(partial); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	withTail := fileSize(t, walPath)
	readOnly, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := readOnly.Close(); err != nil {
		t.Fatal(err)
	}
	if got := fileSize(t, walPath); got != withTail {
		t.Fatalf("read-only collection changed WAL size to %d, want %d", got, withTail)
	}
	writer, err := OpenCollection(context.Background(), dir, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if got := fileSize(t, walPath); got != withTail-int64(len(partial)) {
		t.Fatalf("writer repaired WAL to %d, want %d", got, withTail-int64(len(partial)))
	}
}

func TestCollectionRecoveryRejectsDamagedReferencedFiles(t *testing.T) {
	t.Run("missing primary snapshot", func(t *testing.T) {
		dir, manifest := createClosedCollection(t, false)
		if err := os.Remove(collectionPath(dir, primarySnapshotName(manifest.IDMapGeneration))); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenCollection(context.Background(), dir, CollectionOptions{}); !errors.Is(err, ErrCollectionCorrupt) {
			t.Fatalf("open error = %v", err)
		}
	})
	t.Run("corrupt segment", func(t *testing.T) {
		dir, manifest := createClosedCollection(t, true)
		name := collectionPath(dir, manifest.PersistedSegments[0].Files[0])
		contents, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		contents[len(contents)-1] ^= 1
		if err := os.WriteFile(name, contents, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenCollection(context.Background(), dir, CollectionOptions{}); !errors.Is(err, ErrCollectionCorrupt) {
			t.Fatalf("open error = %v", err)
		}
	})
	t.Run("corrupt WAL operation", func(t *testing.T) {
		dir, manifest := createClosedCollection(t, false)
		name := collectionPath(dir, manifest.WritingSegment.Files[0])
		contents, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		contents[len(contents)-1] ^= 1
		if err := os.WriteFile(name, contents, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenCollection(context.Background(), dir, CollectionOptions{}); !errors.Is(err, ErrCollectionCorrupt) {
			t.Fatalf("open error = %v", err)
		}
	})
}

func TestCollectionIgnoresUnpublishedArtifacts(t *testing.T) {
	dir, current := createClosedCollection(t, false)
	orphan := current.Clone()
	orphan.Generation = current.Generation + 100
	orphan.Schema = json.RawMessage(`{"name":"orphan"}`)
	encoded, err := MarshalManifest(orphan)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, manifestFileName(orphan.Generation)), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	opened, err := OpenCollection(context.Background(), dir, CollectionOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if got := opened.Manifest(); got.Generation != current.Generation || string(got.Schema) != string(testCollectionSchema) {
		t.Fatalf("opened orphan manifest = %#v", got)
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
	if !errors.As(err, &exitError) || exitError.ExitCode() != 93 {
		t.Fatalf("child exit = %v", err)
	}
	opened, err := OpenCollection(context.Background(), dir, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	results, err := opened.Fetch(context.Background(), []string{"survivor"})
	if err != nil || results[0].Document == nil || string(results[0].Document.Payload) != "durable" {
		t.Fatalf("recovered child write = %#v, %v", results, err)
	}
}

func TestCollectionArgumentsAndClose(t *testing.T) {
	var nilStore *CollectionStore
	if _, err := nilStore.Insert(context.Background(), []WriteInput{{PrimaryKey: "x"}}); err == nil {
		t.Fatal("nil collection insert succeeded")
	}
	if err := nilStore.Flush(context.Background()); err == nil {
		t.Fatal("nil collection flush succeeded")
	}
	if _, err := CreateCollection(nil, t.TempDir(), testCollectionSchema, CollectionOptions{}); err == nil {
		t.Fatal("nil create context succeeded")
	}
	if _, err := CreateCollection(context.Background(), t.TempDir(), json.RawMessage(`[]`), CollectionOptions{}); err == nil {
		t.Fatal("non-object schema succeeded")
	}
	if _, err := CreateCollection(context.Background(), t.TempDir(), testCollectionSchema, CollectionOptions{ReadOnly: true}); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("read-only create error = %v", err)
	}
	if _, err := OpenCollection(nil, t.TempDir(), CollectionOptions{}); err == nil {
		t.Fatal("nil open context succeeded")
	}
	if _, err := OpenCollection(context.Background(), filepath.Join(t.TempDir(), "missing"), CollectionOptions{}); !errors.Is(err, ErrManifestNotFound) {
		t.Fatalf("missing collection error = %v", err)
	}
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	deadline, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := CreateCollection(deadline, dir, testCollectionSchema, CollectionOptions{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second create while writer is open = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Fetch(context.Background(), []string{"x"}); !errors.Is(err, ErrCollectionClosed) {
		t.Fatalf("closed fetch error = %v", err)
	}
	if err := store.Flush(context.Background()); !errors.Is(err, ErrCollectionClosed) {
		t.Fatalf("closed flush error = %v", err)
	}
}

func TestCollectionPublishSchemaIsAtomicAndDurable(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := CreateCollection(ctx, dir, testCollectionSchema, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	initial := store.Manifest()
	if committed, err := store.PublishSchema(ctx, json.RawMessage(`[`)); err == nil || committed {
		t.Fatalf("invalid schema = committed=%t error=%v", committed, err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if committed, err := store.PublishSchema(canceled, json.RawMessage(`{"name":"canceled"}`)); !errors.Is(err, context.Canceled) || committed {
		t.Fatalf("canceled schema = committed=%t error=%v", committed, err)
	}
	if current := store.Manifest(); current.Generation != initial.Generation || string(current.Schema) != string(initial.Schema) {
		t.Fatalf("failed publication changed manifest = %#v", current)
	}
	updatedSchema := json.RawMessage(`{"name":"articles","fields":[]}`)
	committed, err := store.PublishSchema(ctx, updatedSchema)
	if err != nil || !committed {
		t.Fatalf("publish schema = committed=%t error=%v", committed, err)
	}
	updated := store.Manifest()
	if updated.Generation <= initial.Generation || string(updated.Schema) != string(updatedSchema) {
		t.Fatalf("updated manifest = %#v", updated)
	}
	if committed, err := store.PublishSchema(ctx, updatedSchema); err != nil || committed {
		t.Fatalf("idempotent publication = committed=%t error=%v", committed, err)
	}
	if store.Manifest().Generation != updated.Generation {
		t.Fatal("idempotent publication advanced generation")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if committed, err := store.PublishSchema(ctx, testCollectionSchema); !errors.Is(err, ErrCollectionClosed) || committed {
		t.Fatalf("closed publication = committed=%t error=%v", committed, err)
	}
	readOnly, err := OpenCollection(ctx, dir, CollectionOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	defer readOnly.Close()
	if got := readOnly.Manifest(); string(got.Schema) != string(updatedSchema) {
		t.Fatalf("reopened schema = %s", got.Schema)
	}
	if committed, err := readOnly.PublishSchema(ctx, testCollectionSchema); !errors.Is(err, ErrReadOnly) || committed {
		t.Fatalf("read-only publication = committed=%t error=%v", committed, err)
	}
	var nilStore *CollectionStore
	if committed, err := nilStore.PublishSchema(ctx, testCollectionSchema); err == nil || committed {
		t.Fatalf("nil publication = committed=%t error=%v", committed, err)
	}
}

func createClosedCollection(t *testing.T, flush bool) (string, Manifest) {
	t.Helper()
	dir := t.TempDir()
	store, err := CreateCollection(context.Background(), dir, testCollectionSchema, CollectionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Insert(context.Background(), []WriteInput{{PrimaryKey: "one", Payload: []byte("payload")}}); err != nil {
		t.Fatal(err)
	}
	if flush {
		if err := store.Flush(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	manifest := store.Manifest()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return dir, manifest
}
