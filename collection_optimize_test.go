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

package zvec

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestOptimizeCompactsLiveDocumentsAndPrunesArtifacts(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{2, 0}),
		testPublicDocument("c", "charlie", "low", 3, 3, []float32{3, 0}),
		testPublicDocument("d", "delta", "high", 4, 4, []float32{4, 0}),
		testPublicDocument("e", "echo", "low", 5, 5, []float32{5, 0}),
		testPublicDocument("f", "foxtrot", "high", 6, 6, []float32{6, 0}),
	}
	wantIDs := make(map[string]uint64, len(documents))
	for start := 0; start < len(documents); start += 2 {
		results, insertErr := collection.Insert(ctx, documents[start:start+2])
		require.NoError(t, insertErr)

		for index := range results {
			wantIDs[results[index].PrimaryKey] = results[index].DocID
		}
		{
			err := collection.Flush(ctx)
			require.NoError(t, err)
		}
	}
	initial := collection.store.Manifest()
	require.Len(t, initial.PersistedSegments, 3)

	unknown := filepath.Join(path, "segments", "application", "note.txt")
	{
		err := os.MkdirAll(filepath.Dir(unknown), 0o755)
		require.NoError(t, err)
	}
	{
		err := os.WriteFile(unknown, []byte("retain me"), 0o644)
		require.NoError(t, err)
	}

	outside := t.TempDir()
	escapeTarget := filepath.Join(outside, "data-external.seg")
	{
		err := os.WriteFile(escapeTarget, []byte("external"), 0o644)
		require.NoError(t, err)
	}

	escapeLink := filepath.Join(path, "segments", "escape")
	symlinkCreated := os.Symlink(outside, escapeLink) == nil

	before, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 6,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 3})
		require.NoError(t, err)
	}

	optimized := collection.store.Manifest()
	require.True(t, optimized.Generation > initial.Generation)
	require.Len(t, optimized.PersistedSegments, 1)
	require.True(t, optimized.WritingSegmentStartDocID == 6)

	assertOptimizeArtifacts(t, path, 1)
	{
		content, err := os.ReadFile(unknown)
		require.NoError(t, err)
		require.True(t, string(content) == "retain me")
	}

	if symlinkCreated {
		{
			content, err := os.ReadFile(escapeTarget)
			require.NoError(t, err)
			require.True(t, string(content) == "external")
		}
	}
	after, err := collection.Query(ctx, VectorQuery{
		Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 6,
		Projection: Projection{OutputFields: []string{"title"}},
	})
	require.NoError(t, err)
	require.Equal(t, before, after)

	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)

	// A canonical collection is a manifest no-op, while prune remains safe to
	// retry for a process that stopped just after an earlier publication.
	generation := optimized.Generation
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 1})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, generation, got)
	}

	updated, err := collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"rating": int32(10)}}})
	require.NoError(t, err)

	wantIDs["a"] = updated[0].DocID
	{
		_, err := collection.Delete(ctx, []string{"e"})
		require.NoError(t, err)
	}

	delete(wantIDs, "e")
	temporary := testPublicDocument("temporary", "temporary", "low", 7, 7, []float32{7, 0})
	inserted, err := collection.Insert(ctx, []Document{temporary})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 7)
	{
		_, err := collection.Delete(ctx, []string{"temporary"})
		require.NoError(t, err)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: 2})
		require.NoError(t, err)
	}

	optimized = collection.store.Manifest()
	require.Len(t, optimized.PersistedSegments, 2)
	require.True(t, optimized.WritingSegmentStartDocID == 8)

	assertOptimizeArtifacts(t, path, 2)
	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)
	fetched, err := collection.Fetch(ctx, []string{"a", "e", "temporary"}, Projection{})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, int32(10), fetched[0].Fields["rating"])
	require.Nil(t, fetched[1])
	require.Nil(t, fetched[2])

	next := testPublicDocument("next", "next", "low", 8, 8, []float32{8, 0})
	inserted, err = collection.Insert(ctx, []Document{next})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 8)

	wantIDs["next"] = 8
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	assertOptimizeDocumentIDs(t, ctx, collection, wantIDs)
	require.Equal(t, uint64(len(wantIDs)), collection.Stats().DocumentCount)
}

func TestOptimizeFullyDeletedCollectionKeepsDocumentIDsMonotonic(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "optimize-empty")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	documents := []Document{
		testPublicDocument("a", "a", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "b", "high", 2, 2, []float32{2, 0}),
	}
	{
		_, err := collection.Insert(ctx, documents)
		require.NoError(t, err)
	}
	{
		err := collection.Flush(ctx)
		require.NoError(t, err)
	}
	{
		_, err := collection.Delete(ctx, []string{"a", "b"})
		require.NoError(t, err)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.NoError(t, err)
	}

	manifest := collection.store.Manifest()
	require.Len(t, manifest.PersistedSegments, 0)
	require.True(t, manifest.WritingSegmentStartDocID == 2)
	require.True(t, collection.Stats().DocumentCount == 0)

	assertOptimizeArtifacts(t, path, 0)
	{
		err := collection.Close()
		require.NoError(t, err)
	}

	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)

	defer collection.Close()
	inserted, err := collection.Insert(ctx, []Document{testPublicDocument("c", "c", "low", 3, 3, []float32{3, 0})})
	require.NoError(t, err)
	require.True(t, inserted[0].DocID == 2)
}

func TestOptimizeValidationAndRollback(t *testing.T) {
	ctx := context.Background()
	var nilCollection *Collection
	{
		err := nilCollection.Optimize(ctx, OptimizeOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	path := filepath.Join(t.TempDir(), "optimize-errors")
	collection, err := CreateAndOpen(ctx, path, testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	{
		err := collection.Optimize(nil, OptimizeOptions{})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{Concurrency: -1})
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	{
		err := collection.Optimize(canceled, OptimizeOptions{})
		require.ErrorIs(t, err, context.Canceled)
	}

	initialGeneration := collection.store.Manifest().Generation
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, initialGeneration, got)
	}
	{
		_, err := collection.Insert(ctx, []Document{testPublicDocument("a", "a", "low", 1, 1, []float32{1, 0})})
		require.NoError(t, err)
	}

	initialGeneration = collection.store.Manifest().Generation
	versionLock, err := ailego.AcquireFileLock(ctx, filepath.Join(path, ".version.lock"), ailego.LockExclusive)
	require.NoError(t, err)

	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	err = collection.Optimize(deadline, OptimizeOptions{Concurrency: 2})
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		_ = versionLock.Close()
	}
	require.ErrorIs(t, err, context.DeadlineExceeded)
	{
		err := versionLock.Close()
		require.NoError(t, err)
	}
	{
		got := collection.store.Manifest().Generation
		require.Equal(t, initialGeneration, got)
	}

	fetched, err := collection.Fetch(ctx, []string{"a"}, Projection{})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.True(t, fetched[0].DocID == 0)
	{
		err := collection.Close()
		require.NoError(t, err)
	}
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.ErrorIs(t, err, ErrFailedPrecondition)
	}

	readOnlyOptions := NewCollectionOptions()
	readOnlyOptions.ReadOnly = true
	collection, err = Open(ctx, path, readOnlyOptions)
	require.NoError(t, err)
	{
		err := collection.Optimize(ctx, OptimizeOptions{})
		require.ErrorIs(t, err, ErrPermissionDenied)
	}
	{
		err := collection.Close()
		require.NoError(t, err)
	}
}

func assertOptimizeDocumentIDs(t *testing.T, ctx context.Context, collection *Collection, want map[string]uint64) {
	t.Helper()
	keys := make([]string, 0, len(want))
	for key := range want {
		keys = append(keys, key)
	}
	fetched, err := collection.Fetch(ctx, keys, Projection{})
	require.NoError(t, err)

	for index, key := range keys {
		require.NotNil(t, fetched[index])
		require.Equal(t, want[key], fetched[index].DocID)
	}
}

func assertOptimizeArtifacts(t *testing.T, path string, segments int) {
	t.Helper()
	patterns := map[string]int{
		filepath.Join(path, "segments", "*", "data-*.seg"): segments,
		filepath.Join(path, "wal", "*.wal"):                1,
		filepath.Join(path, "wal", "*.wal.lock"):           1,
		filepath.Join(path, "snapshots", "primary-*.snap"): 1,
		filepath.Join(path, "snapshots", "delete-*.snap"):  1,
	}
	for pattern, want := range patterns {
		matches, err := filepath.Glob(pattern)
		require.NoError(t, err)

		filtered := matches[:0]
		for _, name := range matches {
			parent, statErr := os.Lstat(filepath.Dir(name))
			require.NoError(t, statErr)

			if parent.Mode()&os.ModeSymlink == 0 {
				filtered = append(filtered, name)
			}
		}
		matches = filtered
		require.Len(t, matches, want)
	}
}
