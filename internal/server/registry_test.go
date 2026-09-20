// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package server

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/xvec"
	"github.com/stretchr/testify/require"
)

func testSchema(name string) xvec.CollectionSchema {
	return xvec.NewCollectionSchema(name, xvec.NewField("title", xvec.DataTypeString))
}

func TestRegistryCreateAndAcquireShareHandle(t *testing.T) {
	registry, err := NewRegistry(filepath.Join(t.TempDir(), "data"), xvec.CollectionOptions{MaxBufferSize: 12345})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, registry.Close()) })

	created, err := registry.Create(context.Background(), testSchema("books"), CreateOptions{})
	require.NoError(t, err)
	acquired, release, err := registry.Acquire(context.Background(), "books")
	require.NoError(t, err)
	require.Same(t, created, acquired)
	require.Equal(t, uint32(12345), acquired.Options().MaxBufferSize)

	release()
	release()
}

func TestRegistryCreateMmapOverrideIsExplicit(t *testing.T) {
	registry, err := NewRegistry(t.TempDir(), xvec.CollectionOptions{EnableMmap: true})
	require.NoError(t, err)
	mmap := false
	collection, err := registry.Create(context.Background(), testSchema("books"), CreateOptions{EnableMmap: &mmap})
	require.NoError(t, err)
	require.False(t, collection.Options().EnableMmap)
	require.NoError(t, registry.Close())

	reopened, err := NewRegistry(registry.DataDir(), xvec.CollectionOptions{EnableMmap: true})
	require.NoError(t, err)
	collection, release, err := reopened.Acquire(context.Background(), "books")
	require.NoError(t, err)
	require.False(t, collection.Options().EnableMmap)
	release()
	require.NoError(t, reopened.Close())
}

func TestRegistryLazilyReopensPersistedCollection(t *testing.T) {
	dataDir := t.TempDir()
	first, err := NewRegistry(dataDir, xvec.NewCollectionOptions())
	require.NoError(t, err)
	created, err := first.Create(context.Background(), testSchema("books"), CreateOptions{})
	require.NoError(t, err)
	path := created.Path()
	require.NoError(t, first.Close())

	second, err := NewRegistry(dataDir, xvec.NewCollectionOptions())
	require.NoError(t, err)
	reopened, release, err := second.Acquire(context.Background(), "books")
	require.NoError(t, err)
	require.Equal(t, path, reopened.Path())
	release()
	require.NoError(t, second.Close())
}

func TestRegistryConcurrentAcquireUsesOneHandle(t *testing.T) {
	dataDir := t.TempDir()
	first, err := NewRegistry(dataDir, xvec.NewCollectionOptions())
	require.NoError(t, err)
	_, err = first.Create(context.Background(), testSchema("books"), CreateOptions{})
	require.NoError(t, err)
	require.NoError(t, first.Close())

	registry, err := NewRegistry(dataDir, xvec.NewCollectionOptions())
	require.NoError(t, err)
	const count = 8
	collections := make([]*xvec.Collection, count)
	releases := make([]func(), count)
	errs := make([]error, count)
	var wait sync.WaitGroup
	wait.Add(count)
	for i := range count {
		go func() {
			defer wait.Done()
			collections[i], releases[i], errs[i] = registry.Acquire(context.Background(), "books")
		}()
	}
	wait.Wait()
	for i := range count {
		require.NoError(t, errs[i])
		require.Same(t, collections[0], collections[i])
		releases[i]()
	}
	require.NoError(t, registry.Close())
}

func TestRegistryListIncludesPersistedAndLoadedCollections(t *testing.T) {
	dataDir := t.TempDir()
	registry, err := NewRegistry(dataDir, xvec.NewCollectionOptions())
	require.NoError(t, err)
	_, err = registry.Create(context.Background(), testSchema("zebra"), CreateOptions{})
	require.NoError(t, err)
	_, err = registry.Create(context.Background(), testSchema("alpha"), CreateOptions{})
	require.NoError(t, err)
	require.NoError(t, registry.Close())

	require.NoError(t, os.Mkdir(filepath.Join(dataDir, "junkdir"), 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(dataDir, "junkfile"), []byte("junk"), 0o600))
	require.NoError(t, os.Mkdir(filepath.Join(dataDir, "x"), 0o700))
	if err := os.Symlink(filepath.Join(dataDir, "alpha"), filepath.Join(dataDir, "linked")); err != nil {
		t.Skipf("symlinks are unavailable: %v", err)
	}

	registry, err = NewRegistry(dataDir, xvec.NewCollectionOptions())
	require.NoError(t, err)
	names, err := registry.List()
	require.NoError(t, err)
	require.Equal(t, []string{"alpha", "zebra"}, names)
	require.NoError(t, registry.Close())
}

func TestRegistryRejectsInvalidAndSymlinkNames(t *testing.T) {
	dataDir := t.TempDir()
	outside := t.TempDir()
	registry, err := NewRegistry(dataDir, xvec.NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, registry.Close()) }()

	for _, name := range []string{"../out", "ab", "with.dot", "nested/name"} {
		_, _, err := registry.Acquire(context.Background(), name)
		require.ErrorIs(t, err, ErrInvalidCollectionName)
	}

	require.NoError(t, os.Symlink(outside, filepath.Join(dataDir, "linked")))
	_, _, err = registry.Acquire(context.Background(), "linked")
	require.ErrorIs(t, err, ErrSymlinkCollection)
	_, err = registry.Create(context.Background(), testSchema("linked"), CreateOptions{})
	require.ErrorIs(t, err, ErrSymlinkCollection)
	_, statErr := os.Stat(filepath.Join(outside, "CURRENT"))
	require.ErrorIs(t, statErr, os.ErrNotExist)
}

func TestRegistryDestroyWaitsForActiveLease(t *testing.T) {
	registry, err := NewRegistry(t.TempDir(), xvec.NewCollectionOptions())
	require.NoError(t, err)
	collection, err := registry.Create(context.Background(), testSchema("books"), CreateOptions{})
	require.NoError(t, err)
	_, release, err := registry.Acquire(context.Background(), "books")
	require.NoError(t, err)

	done := make(chan error, 1)
	go func() { done <- registry.Destroy(context.Background(), "books") }()
	require.Eventually(t, func() bool {
		_, releaseProbe, acquireErr := registry.Acquire(context.Background(), "books")
		if acquireErr == nil {
			releaseProbe()
		}
		return errors.Is(acquireErr, ErrCollectionUnavailable)
	}, time.Second, time.Millisecond)
	select {
	case err := <-done:
		t.Fatalf("Destroy returned before release: %v", err)
	default:
	}

	release()
	require.NoError(t, <-done)
	_, statErr := os.Stat(collection.Path())
	require.ErrorIs(t, statErr, os.ErrNotExist)
	require.NoError(t, registry.Close())
}

func TestRegistryDestroyCancellationRestoresAvailability(t *testing.T) {
	registry, err := NewRegistry(t.TempDir(), xvec.NewCollectionOptions())
	require.NoError(t, err)
	_, err = registry.Create(context.Background(), testSchema("books"), CreateOptions{})
	require.NoError(t, err)
	_, release, err := registry.Acquire(context.Background(), "books")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	require.ErrorIs(t, registry.Destroy(ctx, "books"), context.DeadlineExceeded)

	reopened, releaseReopened, err := registry.Acquire(context.Background(), "books")
	require.NoError(t, err)
	require.NotNil(t, reopened)
	releaseReopened()
	release()
	require.NoError(t, registry.Close())
}

func TestRegistryCloseBlocksNewWorkAndWaitsForLeases(t *testing.T) {
	registry, err := NewRegistry(t.TempDir(), xvec.NewCollectionOptions())
	require.NoError(t, err)
	collection, err := registry.Create(context.Background(), testSchema("books"), CreateOptions{})
	require.NoError(t, err)
	_, release, err := registry.Acquire(context.Background(), "books")
	require.NoError(t, err)

	done := make(chan error, 1)
	go func() { done <- registry.Close() }()
	require.Eventually(t, func() bool {
		_, releaseProbe, acquireErr := registry.Acquire(context.Background(), "books")
		if acquireErr == nil {
			releaseProbe()
		}
		return errors.Is(acquireErr, ErrRegistryClosed)
	}, time.Second, time.Millisecond)
	select {
	case err := <-done:
		t.Fatalf("Close returned before release: %v", err)
	default:
	}

	release()
	require.NoError(t, <-done)
	_, _, err = registry.Acquire(context.Background(), "books")
	require.ErrorIs(t, err, ErrRegistryClosed)
	_, err = registry.Create(context.Background(), testSchema("other"), CreateOptions{})
	require.ErrorIs(t, err, ErrRegistryClosed)
	require.ErrorIs(t, registry.Destroy(context.Background(), "books"), ErrRegistryClosed)
	require.NoError(t, registry.Close())
	require.NoError(t, collection.Close())
}
