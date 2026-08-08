// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
package indexstore

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestStoreOperationsIteratorsAndReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.pebble")
	store, err := Open(path, Options{})
	require.NoError(t, err)

	require.NoError(t, store.Set([]byte("term/apple/0001"), []byte("one")))
	require.NoError(t, store.Set([]byte("term/apple/0002"), []byte("two")))
	require.NoError(t, store.Set([]byte("term/banana/0001"), []byte("three")))
	value, err := store.Get([]byte("term/apple/0001"))
	require.NoError(t, err)
	require.Equal(t, []byte("one"), value)
	value[0] = 'X'
	value, err = store.Get([]byte("term/apple/0001"))
	require.NoError(t, err)
	require.Equal(t, []byte("one"), value, "Get returned aliased Pebble memory")

	prefix, err := store.NewPrefixIterator([]byte("term/apple/"))
	require.NoError(t, err)
	require.Equal(t, [][2]string{{"term/apple/0001", "one"}, {"term/apple/0002", "two"}}, collectIterator(t, prefix))

	ranged, err := store.NewRangeIterator([]byte("term/apple/0002"), []byte("term/banana/0002"))
	require.NoError(t, err)
	require.Equal(t, [][2]string{{"term/apple/0002", "two"}, {"term/banana/0001", "three"}}, collectIterator(t, ranged))

	batch := store.NewBatch()
	require.NoError(t, batch.Set([]byte("term/cherry/0001"), []byte("four")))
	require.NoError(t, batch.Delete([]byte("term/apple/0001")))
	require.NoError(t, batch.Commit())
	require.NoError(t, batch.Close())
	_, err = store.Get([]byte("term/apple/0001"))
	require.ErrorIs(t, err, ErrNotFound)
	require.NoError(t, store.Delete([]byte("term/apple/0002")))
	require.NoError(t, store.Flush())
	require.NoError(t, store.Compact([]byte("term/"), []byte("term0")))
	checkpoint := filepath.Join(t.TempDir(), "checkpoint.pebble")
	require.NoError(t, store.Checkpoint(checkpoint))
	require.NoError(t, store.Close())
	require.NoError(t, store.Close())

	readOnly, err := Open(path, Options{ReadOnly: true})
	require.NoError(t, err)
	value, err = readOnly.Get([]byte("term/cherry/0001"))
	require.NoError(t, err)
	require.Equal(t, []byte("four"), value)
	require.Error(t, readOnly.Set([]byte("x"), []byte("y")))
	require.NoError(t, readOnly.Close())

	copyStore, err := Open(checkpoint, Options{ReadOnly: true})
	require.NoError(t, err)
	value, err = copyStore.Get([]byte("term/cherry/0001"))
	require.NoError(t, err)
	require.Equal(t, []byte("four"), value)
	require.NoError(t, copyStore.Close())
}

func TestStoreDestroyAndClosedErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "destroy.pebble")
	store, err := Open(path, Options{})
	require.NoError(t, err)
	require.NoError(t, store.Set([]byte("key"), []byte("value")))
	require.NoError(t, store.Destroy())
	_, err = os.Stat(path)
	require.ErrorIs(t, err, os.ErrNotExist)
	_, err = store.Get([]byte("key"))
	require.ErrorIs(t, err, ErrClosed)
	require.ErrorIs(t, store.Destroy(), ErrClosed)
}

func TestStoreRejectsSymlink(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target")
	require.NoError(t, os.Mkdir(target, 0o700))
	link := filepath.Join(root, "link")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	_, err := Open(link, Options{})
	require.Error(t, err)
}

func collectIterator(t *testing.T, iterator *Iterator) [][2]string {
	t.Helper()
	defer func() { require.NoError(t, iterator.Close()) }()
	var values [][2]string
	for valid := iterator.First(); valid; valid = iterator.Next() {
		values = append(values, [2]string{string(iterator.Key()), string(iterator.Value())})
	}
	require.NoError(t, iterator.Error())
	return values
}

func TestPrefixSuccessor(t *testing.T) {
	t.Parallel()
	require.Equal(t, []byte{'b'}, prefixSuccessor([]byte{'a', 0xff}))
	require.Nil(t, prefixSuccessor([]byte{0xff, 0xff}))
	require.True(t, errors.Is(ErrNotFound, ErrNotFound))
}
