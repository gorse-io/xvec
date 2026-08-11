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
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/require"
)

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
	encoded, err := encodeSnapshot(deleteSnapshotMagic, 1, []byte("payload"))
	require.NoError(f, err)

	f.Add(encoded)
	f.Add(encoded[:snapshotHeaderSize])
	f.Add([]byte("not a snapshot"))
	f.Fuzz(func(t *testing.T, data []byte) {
		count, payload, err := decodeSnapshot(data, deleteSnapshotMagic)
		if err == nil {
			require.Equal(t, binary.LittleEndian.Uint64(data[16:24]), count)
			require.True(t, bytes.Equal(payload, data[snapshotHeaderSize:]),
				"decoded payload differs")
		}
	})
}
