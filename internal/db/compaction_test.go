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
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gofrs/flock"
	"github.com/gorse-io/xvec/internal/db/index/common"
	"github.com/stretchr/testify/require"
)

func TestCompactionPreservesConcurrentMutationsAndFlush(t *testing.T) {
	for _, mapped := range []bool{false, true} {
		t.Run(fmt.Sprintf("mmap=%v", mapped), func(t *testing.T) {
			ctx := context.Background()
			path := t.TempDir()
			store, err := CreateCollection(ctx, path, testCollectionSchema, CollectionOptions{EnableMmap: mapped, SegmentMaxDocuments: 8})
			require.NoError(t, err)
			defer func() { require.NoError(t, store.Close()) }()
			_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "a", Payload: []byte("old-a")}, {PrimaryKey: "b"}})
			require.NoError(t, err)
			require.NoError(t, store.Flush(ctx))
			_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "c"}, {PrimaryKey: "d", Payload: []byte("old-d")}})
			require.NoError(t, err)
			require.NoError(t, store.Flush(ctx))
			_, err = store.Delete(ctx, []string{"b"})
			require.NoError(t, err)
			// A delete-only WAL must be checkpointed before purging its records.
			require.NoError(t, store.Flush(ctx))
			require.False(t, store.wal.HasRecords())
			compaction, err := store.BeginCompaction(ctx)
			require.NoError(t, err)
			defer func() { require.NoError(t, compaction.Close()) }()
			require.NoError(t, compaction.Prepare(ctx))
			require.True(t, compaction.Changed())
			require.Equal(t, []uint64{1}, compaction.purged)

			updated, err := store.Update(ctx, []WriteInput{{PrimaryKey: "a", Payload: []byte("new-a")}})
			require.NoError(t, err)
			require.Equal(t, uint64(4), updated[0].DocID)
			_, err = store.Delete(ctx, []string{"c"})
			require.NoError(t, err)
			_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "b", Payload: []byte("new-b")}, {PrimaryKey: "e", Payload: []byte("new-e")}})
			require.NoError(t, err)
			// A later flush must neither reuse reserved output IDs nor disappear
			// when the prepared compaction is committed.
			require.NoError(t, store.Flush(ctx))
			_, err = store.Update(ctx, []WriteInput{{PrimaryKey: "d", Payload: []byte("new-d")}})
			require.NoError(t, err)
			_, err = store.Delete(ctx, []string{"b"})
			require.NoError(t, err)
			wal := store.wal
			primary := store.manager.PrimaryKeys()
			committed, err := compaction.Commit(ctx, nil)
			require.NoError(t, err)
			require.True(t, committed)
			require.Same(t, wal, store.wal)
			require.Same(t, primary, store.manager.PrimaryKeys())
			require.False(t, store.manager.Deletes().IsDeleted(1))
			require.True(t, store.manager.Deletes().IsDeleted(0))
			require.True(t, store.manager.Deletes().IsDeleted(2))
			require.NoError(t, validateCollectionState(store.manager))
			require.NoError(t, store.PruneObsoleteArtifacts(ctx))
			require.NoError(t, store.Close())

			store, err = OpenCollection(ctx, path, CollectionOptions{})
			require.NoError(t, err)
			fetched, err := store.Fetch(ctx, []string{"a", "b", "c", "d", "e"})
			require.NoError(t, err)
			require.Equal(t, []byte("new-a"), fetched[0].Document.Payload)
			require.Nil(t, fetched[1].Document)
			require.Nil(t, fetched[2].Document)
			require.Equal(t, []byte("new-d"), fetched[3].Document.Payload)
			require.Equal(t, []byte("new-e"), fetched[4].Document.Payload)
			inserted, err := store.Insert(ctx, []WriteInput{{PrimaryKey: "next"}})
			require.NoError(t, err)
			require.Equal(t, uint64(8), inserted[0].DocID)
		})
	}
}

func TestCompactionFailedCommitLeavesCurrentWritesRecoverable(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir()
	store, err := CreateCollection(ctx, path, testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 8})
	require.NoError(t, err)
	defer func() { require.NoError(t, store.Close()) }()
	for _, key := range []string{"a", "b"} {
		_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: key}})
		require.NoError(t, err)
		require.NoError(t, store.Flush(ctx))
	}
	compaction, err := store.BeginCompaction(ctx)
	require.NoError(t, err)
	defer func() { require.NoError(t, compaction.Close()) }()
	require.NoError(t, compaction.Prepare(ctx))
	paths := append([]string(nil), compaction.created...)
	require.NotEmpty(t, paths)
	_, err = store.Update(ctx, []WriteInput{{PrimaryKey: "a", Payload: []byte("updated")}})
	require.NoError(t, err)
	generation := store.Manifest().Generation
	output := compaction.outputs[0].Metadata()
	committed, err := compaction.Commit(ctx, []common.SegmentIndexSnapshotMetadata{{
		SegmentID: output.ID, SchemaSHA256: strings.Repeat("a", 64),
		DocumentCount: output.DocCount, MinDocumentID: output.MinDocID, MaxDocumentID: output.MaxDocID,
		Artifacts: []common.IndexArtifactMetadata{{Field: "embedding", Kind: "vector-2", File: "indexes/missing.zvi"}},
	}})
	require.ErrorIs(t, err, os.ErrNotExist)
	require.False(t, committed)
	require.Equal(t, generation, store.Manifest().Generation)
	lock := flock.New(filepath.Join(path, ".version.lock"))
	locked, err := lock.TryLock()
	require.NoError(t, err)
	require.True(t, locked)
	defer func() { require.NoError(t, lock.Close()) }()
	deadline, cancel := context.WithTimeout(ctx, 75*time.Millisecond)
	defer cancel()
	committed, err = compaction.Commit(deadline, nil)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.False(t, committed)
	require.Equal(t, generation, store.Manifest().Generation)
	require.NoError(t, compaction.Close())
	for _, path := range paths {
		_, err := os.Stat(path)
		require.ErrorIs(t, err, os.ErrNotExist)
	}
	require.NoError(t, store.Close())
	require.NoError(t, lock.Close())
	store, err = OpenCollection(ctx, path, CollectionOptions{})
	require.NoError(t, err)
	fetched, err := store.Fetch(ctx, []string{"a", "b"})
	require.NoError(t, err)
	require.Equal(t, []byte("updated"), fetched[0].Document.Payload)
	require.NotNil(t, fetched[1].Document)
}
