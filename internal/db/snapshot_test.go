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
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

func TestPrimaryKeyMapLifecycleAndSnapshot(t *testing.T) {
	ctx := context.Background()
	primary := NewPrimaryKeyMap()
	{
		previous, replaced, err := primary.Put(ctx, "beta", DocumentLocation{SegmentID: 2, DocID: 20})
		require.NoError(t, err)
		require.False(t, replaced)
		require.Equal(t, DocumentLocation{}, previous)
	}
	{
		_, _, err := primary.Put(ctx, "alpha", DocumentLocation{SegmentID: 1, DocID: 10})
		require.NoError(t, err)
	}

	previous, replaced, err := primary.Put(ctx, "beta", DocumentLocation{SegmentID: 3, DocID: 30})
	require.NoError(t, err)
	require.True(t, replaced)
	require.Equal(t, DocumentLocation{SegmentID: 2, DocID: 20}, previous)

	locations, found := primary.MultiGet([]string{"beta", "missing", "alpha"})
	require.Equal(t, []bool{true, false, true}, found)
	require.True(t, locations[0].DocID == 30)
	require.True(t, locations[2].DocID == 10)

	clone := primary.Clone()
	{
		_, _, err := clone.Put(ctx, "clone", DocumentLocation{DocID: 99})
		require.NoError(t, err)
	}
	{
		_, found := primary.Get("clone")
		require.False(t, found,
			"clone shares entries")
	}

	dir := t.TempDir()
	firstName := filepath.Join(dir, "idmap.1")
	{
		err := primary.WriteSnapshot(ctx, firstName)
		require.NoError(t, err)
	}

	loaded, err := LoadPrimaryKeyMap(ctx, firstName)
	require.NoError(t, err)
	require.True(t, loaded.Count() == 2)
	{
		location, found := loaded.Get("beta")
		require.True(t, found)
		require.True(t, location.DocID == 30)
	}

	deterministic := NewPrimaryKeyMap()
	_, _, _ = deterministic.Put(ctx, "alpha", DocumentLocation{SegmentID: 1, DocID: 10})
	_, _, _ = deterministic.Put(ctx, "beta", DocumentLocation{SegmentID: 3, DocID: 30})
	secondName := filepath.Join(dir, "idmap.2")
	{
		err := deterministic.WriteSnapshot(ctx, secondName)
		require.NoError(t, err)
	}

	firstBytes, _ := os.ReadFile(firstName)
	secondBytes, _ := os.ReadFile(secondName)
	require.Equal(t, secondBytes, firstBytes,
		"primary-key snapshot depends on insertion order")
	{
		err := primary.WriteSnapshot(ctx, firstName)
		require.ErrorIs(t, err, os.ErrExist)
	}

	deleted, deletedFound, err := primary.Delete(ctx, "beta")
	require.NoError(t, err)
	require.True(t, deletedFound)
	require.True(t, deleted.DocID == 30)
	require.True(t, primary.Count() == 1)
}

func TestPrimaryKeyValidation(t *testing.T) {
	primary := NewPrimaryKeyMap()
	{
		_, _, err := primary.Put(context.Background(), "", DocumentLocation{})
		require.Error(t, err,
			"empty key succeeded")
	}
	{
		_, _, err := primary.Put(context.Background(), string([]byte{0xff}), DocumentLocation{})
		require.Error(t, err,
			"invalid UTF-8 key succeeded")
	}
	{
		_, _, err := primary.Put(context.Background(), string(make([]byte, maxPrimaryKeyBytes+1)), DocumentLocation{})
		require.Error(t, err,
			"large key succeeded")
	}
	{
		_, _, err := primary.Put(context.Background(), "one", DocumentLocation{DocID: 1})
		require.NoError(t, err)
	}
	{
		_, _, err := primary.Put(context.Background(), "two", DocumentLocation{DocID: 1})
		require.Error(t, err,
			"duplicate document location succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, _, err := primary.Put(canceled, "key", DocumentLocation{})
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, _, err := primary.Delete(nil, "key")
		require.Error(t, err,
			"nil delete context succeeded")
	}
}

func TestPrimaryKeySnapshotDetectsCorruption(t *testing.T) {
	primary := NewPrimaryKeyMap()
	_, _, _ = primary.Put(context.Background(), "alpha", DocumentLocation{SegmentID: 1, DocID: 10})
	_, _, _ = primary.Put(context.Background(), "beta", DocumentLocation{SegmentID: 2, DocID: 20})
	name := filepath.Join(t.TempDir(), "idmap")
	{
		err := primary.WriteSnapshot(context.Background(), name)
		require.NoError(t, err)
	}

	valid, err := os.ReadFile(name)
	require.NoError(t, err)

	for cut := 0; cut < len(valid); cut++ {
		path := filepath.Join(t.TempDir(), "truncated")
		{
			err := os.WriteFile(path, valid[:cut], 0o600)
			require.NoError(t, err)
		}
		{
			_, err := LoadPrimaryKeyMap(context.Background(), path)
			require.ErrorIs(t, err, ErrSnapshotCorrupt)
		}
	}

	corrupted := append([]byte(nil), valid...)
	corrupted[len(corrupted)-1] ^= 1
	corruptName := filepath.Join(t.TempDir(), "corrupt")
	{
		err := os.WriteFile(corruptName, corrupted, 0o600)
		require.NoError(t, err)
	}
	{
		_, err := LoadPrimaryKeyMap(context.Background(), corruptName)
		require.ErrorIs(t, err, ErrSnapshotCorrupt)
	}

	badCount := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint64(badCount[16:24], 99)
	binary.LittleEndian.PutUint32(badCount[36:40], ailego.CRC32C(badCount[:36]))
	badCountName := filepath.Join(t.TempDir(), "bad-count")
	{
		err := os.WriteFile(badCountName, badCount, 0o600)
		require.NoError(t, err)
	}
	{
		_, err := LoadPrimaryKeyMap(context.Background(), badCountName)
		require.ErrorIs(t, err, ErrSnapshotCorrupt)
	}
}

func TestDeleteStoreLifecycleAndSnapshot(t *testing.T) {
	ctx := context.Background()
	store := NewDeleteStore()
	for _, docID := range []uint64{100, 2, 64} {
		changed, err := store.MarkDeleted(ctx, docID)
		require.NoError(t, err)
		require.True(t, changed)
	}
	{
		changed, _ := store.MarkDeleted(ctx, 64)
		require.False(t, changed,
			"duplicate delete changed store")
	}
	require.True(t, store.Count() == 3)
	require.True(t, store.RangeCount(3, 100) == 2)
	require.True(t, store.IsDeleted(2))

	clone := store.Clone()
	_, _ = clone.MarkDeleted(ctx, 9)
	require.False(t, store.IsDeleted(9),
		"delete clone shares state")
	{
		changed, err := store.Restore(ctx, 64)
		require.NoError(t, err)
		require.True(t, changed)
		require.False(t, store.IsDeleted(64))
	}

	name := filepath.Join(t.TempDir(), "delete.1")
	{
		err := store.WriteSnapshot(ctx, name)
		require.NoError(t, err)
	}

	loaded, err := LoadDeleteStore(ctx, name)
	require.NoError(t, err)
	require.True(t, loaded.Count() == 2,
		"loaded delete store is wrong")
	require.True(t, loaded.IsDeleted(2),
		"loaded delete store is wrong")
	require.True(t, loaded.IsDeleted(100),
		"loaded delete store is wrong")

	encoded, _ := os.ReadFile(name)
	payload := encoded[snapshotHeaderSize:]
	first := binary.LittleEndian.Uint64(payload[:8])
	second := binary.LittleEndian.Uint64(payload[8:16])
	binary.LittleEndian.PutUint64(payload[:8], second)
	binary.LittleEndian.PutUint64(payload[8:16], first)
	binary.LittleEndian.PutUint32(encoded[32:36], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(encoded[36:40], ailego.CRC32C(encoded[:36]))
	badName := filepath.Join(t.TempDir(), "delete-bad")
	{
		err := os.WriteFile(badName, encoded, 0o600)
		require.NoError(t, err)
	}
	{
		_, err := LoadDeleteStore(ctx, badName)
		require.ErrorIs(t, err, ErrSnapshotCorrupt)
	}
}

func FuzzDecodeSnapshot(f *testing.F) {
	encoded, err := encodeSnapshot(primaryMapMagic, 1, []byte("payload"))
	require.NoError(f, err)

	f.Add(encoded)
	f.Add(encoded[:snapshotHeaderSize])
	f.Add([]byte("not a snapshot"))
	f.Fuzz(func(t *testing.T, data []byte) {
		count, payload, err := decodeSnapshot(data, primaryMapMagic)
		if err == nil {
			require.Equal(t, binary.LittleEndian.Uint64(data[16:24]), count)
			require.True(t, bytes.Equal(payload, data[snapshotHeaderSize:]),
				"decoded payload differs")
		}
	})
}
