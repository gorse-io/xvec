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
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestPrimaryKeyMapLifecycleAndSnapshot(t *testing.T) {
	ctx := context.Background()
	primary := NewPrimaryKeyMap()
	if previous, replaced, err := primary.Put(ctx, "beta", DocumentLocation{SegmentID: 2, DocID: 20}); err != nil || replaced || previous != (DocumentLocation{}) {
		t.Fatalf("first put = %#v, %v, %v", previous, replaced, err)
	}
	if _, _, err := primary.Put(ctx, "alpha", DocumentLocation{SegmentID: 1, DocID: 10}); err != nil {
		t.Fatal(err)
	}
	previous, replaced, err := primary.Put(ctx, "beta", DocumentLocation{SegmentID: 3, DocID: 30})
	if err != nil || !replaced || previous != (DocumentLocation{SegmentID: 2, DocID: 20}) {
		t.Fatalf("replacement = %#v, %v, %v", previous, replaced, err)
	}
	locations, found := primary.MultiGet([]string{"beta", "missing", "alpha"})
	if !reflect.DeepEqual(found, []bool{true, false, true}) || locations[0].DocID != 30 || locations[2].DocID != 10 {
		t.Fatalf("multi-get = %#v, %#v", locations, found)
	}
	clone := primary.Clone()
	if _, _, err := clone.Put(ctx, "clone", DocumentLocation{DocID: 99}); err != nil {
		t.Fatal(err)
	}
	if _, found := primary.Get("clone"); found {
		t.Fatal("clone shares entries")
	}

	dir := t.TempDir()
	firstName := filepath.Join(dir, "idmap.1")
	if err := primary.WriteSnapshot(ctx, firstName); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadPrimaryKeyMap(ctx, firstName)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Count() != 2 {
		t.Fatalf("loaded count = %d", loaded.Count())
	}
	if location, found := loaded.Get("beta"); !found || location.DocID != 30 {
		t.Fatalf("loaded beta = %#v, %v", location, found)
	}

	deterministic := NewPrimaryKeyMap()
	_, _, _ = deterministic.Put(ctx, "alpha", DocumentLocation{SegmentID: 1, DocID: 10})
	_, _, _ = deterministic.Put(ctx, "beta", DocumentLocation{SegmentID: 3, DocID: 30})
	secondName := filepath.Join(dir, "idmap.2")
	if err := deterministic.WriteSnapshot(ctx, secondName); err != nil {
		t.Fatal(err)
	}
	firstBytes, _ := os.ReadFile(firstName)
	secondBytes, _ := os.ReadFile(secondName)
	if !reflect.DeepEqual(firstBytes, secondBytes) {
		t.Fatal("primary-key snapshot depends on insertion order")
	}
	if err := primary.WriteSnapshot(ctx, firstName); !errors.Is(err, os.ErrExist) {
		t.Fatalf("overwrite error = %v", err)
	}

	deleted, deletedFound, err := primary.Delete(ctx, "beta")
	if err != nil || !deletedFound || deleted.DocID != 30 || primary.Count() != 1 {
		t.Fatalf("delete = %#v, %v, %v", deleted, deletedFound, err)
	}
}

func TestPrimaryKeyValidation(t *testing.T) {
	primary := NewPrimaryKeyMap()
	if _, _, err := primary.Put(context.Background(), "", DocumentLocation{}); err == nil {
		t.Fatal("empty key succeeded")
	}
	if _, _, err := primary.Put(context.Background(), string([]byte{0xff}), DocumentLocation{}); err == nil {
		t.Fatal("invalid UTF-8 key succeeded")
	}
	if _, _, err := primary.Put(context.Background(), string(make([]byte, maxPrimaryKeyBytes+1)), DocumentLocation{}); err == nil {
		t.Fatal("large key succeeded")
	}
	if _, _, err := primary.Put(context.Background(), "one", DocumentLocation{DocID: 1}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := primary.Put(context.Background(), "two", DocumentLocation{DocID: 1}); err == nil {
		t.Fatal("duplicate document location succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := primary.Put(canceled, "key", DocumentLocation{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled put = %v", err)
	}
	if _, _, err := primary.Delete(nil, "key"); err == nil {
		t.Fatal("nil delete context succeeded")
	}
}

func TestPrimaryKeySnapshotDetectsCorruption(t *testing.T) {
	primary := NewPrimaryKeyMap()
	_, _, _ = primary.Put(context.Background(), "alpha", DocumentLocation{SegmentID: 1, DocID: 10})
	_, _, _ = primary.Put(context.Background(), "beta", DocumentLocation{SegmentID: 2, DocID: 20})
	name := filepath.Join(t.TempDir(), "idmap")
	if err := primary.WriteSnapshot(context.Background(), name); err != nil {
		t.Fatal(err)
	}
	valid, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	for cut := 0; cut < len(valid); cut++ {
		path := filepath.Join(t.TempDir(), "truncated")
		if err := os.WriteFile(path, valid[:cut], 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadPrimaryKeyMap(context.Background(), path); !errors.Is(err, ErrSnapshotCorrupt) {
			t.Fatalf("cut %d error = %v", cut, err)
		}
	}

	corrupted := append([]byte(nil), valid...)
	corrupted[len(corrupted)-1] ^= 1
	corruptName := filepath.Join(t.TempDir(), "corrupt")
	if err := os.WriteFile(corruptName, corrupted, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPrimaryKeyMap(context.Background(), corruptName); !errors.Is(err, ErrSnapshotCorrupt) {
		t.Fatalf("payload corruption = %v", err)
	}

	badCount := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint64(badCount[16:24], 99)
	binary.LittleEndian.PutUint32(badCount[36:40], ailego.CRC32C(badCount[:36]))
	badCountName := filepath.Join(t.TempDir(), "bad-count")
	if err := os.WriteFile(badCountName, badCount, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPrimaryKeyMap(context.Background(), badCountName); !errors.Is(err, ErrSnapshotCorrupt) {
		t.Fatalf("count corruption = %v", err)
	}
}

func TestDeleteStoreLifecycleAndSnapshot(t *testing.T) {
	ctx := context.Background()
	store := NewDeleteStore()
	for _, docID := range []uint64{100, 2, 64} {
		changed, err := store.MarkDeleted(ctx, docID)
		if err != nil || !changed {
			t.Fatalf("mark %d = %v, %v", docID, changed, err)
		}
	}
	if changed, _ := store.MarkDeleted(ctx, 64); changed {
		t.Fatal("duplicate delete changed store")
	}
	if store.Count() != 3 || store.RangeCount(3, 100) != 2 || !store.IsDeleted(2) {
		t.Fatalf("delete state count=%d range=%d", store.Count(), store.RangeCount(3, 100))
	}
	clone := store.Clone()
	_, _ = clone.MarkDeleted(ctx, 9)
	if store.IsDeleted(9) {
		t.Fatal("delete clone shares state")
	}
	if changed, err := store.Restore(ctx, 64); err != nil || !changed || store.IsDeleted(64) {
		t.Fatalf("restore = %v, %v", changed, err)
	}

	name := filepath.Join(t.TempDir(), "delete.1")
	if err := store.WriteSnapshot(ctx, name); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadDeleteStore(ctx, name)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Count() != 2 || !loaded.IsDeleted(2) || !loaded.IsDeleted(100) {
		t.Fatalf("loaded delete store is wrong")
	}

	encoded, _ := os.ReadFile(name)
	payload := encoded[snapshotHeaderSize:]
	first := binary.LittleEndian.Uint64(payload[:8])
	second := binary.LittleEndian.Uint64(payload[8:16])
	binary.LittleEndian.PutUint64(payload[:8], second)
	binary.LittleEndian.PutUint64(payload[8:16], first)
	binary.LittleEndian.PutUint32(encoded[32:36], ailego.CRC32C(payload))
	binary.LittleEndian.PutUint32(encoded[36:40], ailego.CRC32C(encoded[:36]))
	badName := filepath.Join(t.TempDir(), "delete-bad")
	if err := os.WriteFile(badName, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDeleteStore(ctx, badName); !errors.Is(err, ErrSnapshotCorrupt) {
		t.Fatalf("unsorted delete snapshot = %v", err)
	}
}

func FuzzDecodeSnapshot(f *testing.F) {
	encoded, err := encodeSnapshot(primaryMapMagic, 1, []byte("payload"))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(encoded)
	f.Add(encoded[:snapshotHeaderSize])
	f.Add([]byte("not a snapshot"))
	f.Fuzz(func(t *testing.T, data []byte) {
		count, payload, err := decodeSnapshot(data, primaryMapMagic)
		if err == nil {
			if count != binary.LittleEndian.Uint64(data[16:24]) {
				t.Fatalf("decoded count = %d", count)
			}
			if !bytes.Equal(payload, data[snapshotHeaderSize:]) {
				t.Fatal("decoded payload differs")
			}
		}
	})
}
