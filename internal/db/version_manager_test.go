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
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/gofrs/flock"
	"github.com/stretchr/testify/require"
)

func TestVersionManagerCreatePublishAndOpen(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "collection")
	initial := sampleManifest(99)
	initial.FormatVersion = 0
	manager, err := CreateVersionManager(context.Background(), dir, initial)
	require.NoError(t, err)

	current := manager.Current()
	require.Equal(t, DiskFormatVersion, current.FormatVersion)
	require.True(t, current.Generation == 1)

	current.Schema[0] = '['
	current.PersistedSegments[0].Files[0] = "changed"
	{
		err := manager.Current().Validate()
		require.NoError(t, err)
	}

	published, err := manager.Update(context.Background(), func(next *Manifest) error {
		next.EnableMmap = false
		next.IDMap = idMapCheckpointName(6)
		return nil
	})
	require.NoError(t, err)
	require.True(t, published.Generation == 2)
	require.False(t, published.EnableMmap)
	require.Equal(t, idMapCheckpointName(6), published.IDMap)

	published.Schema[0] = '['
	{
		err := manager.Current().Validate()
		require.NoError(t, err)
	}

	reopened, err := OpenVersionManager(context.Background(), dir)
	require.NoError(t, err)
	require.Equal(t, manager.Current(), reopened.Current())
	{
		_, err := CreateVersionManager(context.Background(), dir, initial)
		require.ErrorIs(t, err, ErrManifestExists)
	}
	{
		count := manifestCount(t, dir)
		require.True(t, count == 2)
	}
}

func TestVersionManagerRejectsStalePublisher(t *testing.T) {
	dir := t.TempDir()
	first, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)

	stale, err := OpenVersionManager(context.Background(), dir)
	require.NoError(t, err)
	{
		_, err := first.Update(context.Background(), func(next *Manifest) error {
			next.DeleteSnapshotGeneration++
			return nil
		})
		require.NoError(t, err)
	}
	{
		_, err := stale.Publish(context.Background(), stale.Current())
		require.ErrorIs(t, err, ErrManifestConflict)
	}
	require.True(t, stale.Current().Generation == 1)
	{
		count := manifestCount(t, dir)
		require.True(t, count == 2)
	}
}

func TestVersionManagerRejectsConcurrentUpdateFromSameSnapshot(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)

	ready := sync.WaitGroup{}
	ready.Add(2)
	release := make(chan struct{})
	errorsByUpdate := make(chan error, 2)
	for range 2 {
		go func() {
			_, err := manager.Update(context.Background(), func(next *Manifest) error {
				next.IDMap = idMapCheckpointName(6)
				ready.Done()
				<-release
				return nil
			})
			errorsByUpdate <- err
		}()
	}
	ready.Wait()
	close(release)

	succeeded, conflicted := 0, 0
	for range 2 {
		err := <-errorsByUpdate
		switch {
		case err == nil:
			succeeded++
		case errors.Is(err, ErrManifestConflict):
			conflicted++
		default:
			require.NoError(t, err)
		}
	}
	require.True(t, succeeded == 1)
	require.True(t, conflicted == 1)
	{
		current := manager.Current()
		require.True(t, current.Generation == 2)
		require.Equal(t, idMapCheckpointName(6), current.IDMap)
	}
}

func TestVersionManagerIgnoresUnpublishedFiles(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)

	orphan := sampleManifest(2)
	encoded, err := MarshalManifest(orphan)
	require.NoError(t, err)
	{
		err := writeExclusiveSynced(filepath.Join(dir, manifestFileName(2)), encoded)
		require.NoError(t, err)
	}

	pointer, err := marshalCurrent(manifestFileName(2))
	require.NoError(t, err)
	{
		err := os.WriteFile(filepath.Join(dir, ".current-crash.tmp"), pointer, 0o600)
		require.NoError(t, err)
	}

	reopened, err := OpenVersionManager(context.Background(), dir)
	require.NoError(t, err)
	require.True(t, reopened.Current().Generation == 1)

	published, err := manager.Publish(context.Background(), manager.Current())
	require.NoError(t, err)
	require.True(t, published.Generation == 3)
	{
		err := os.WriteFile(filepath.Join(dir, manifestFileName(4)), []byte("partial"), 0o600)
		require.NoError(t, err)
	}

	reopened, err = OpenVersionManager(context.Background(), dir)
	require.NoError(t, err)
	require.True(t, reopened.Current().Generation == 3)

	published, err = manager.Publish(context.Background(), manager.Current())
	require.NoError(t, err)
	require.True(t, published.Generation == 5)
}

func TestVersionManagerCreateSkipsFailedInitialManifest(t *testing.T) {
	dir := t.TempDir()
	orphan := sampleManifest(1)
	encoded, err := MarshalManifest(orphan)
	require.NoError(t, err)
	{
		err := writeExclusiveSynced(filepath.Join(dir, manifestFileName(1)), encoded)
		require.NoError(t, err)
	}

	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(0))
	require.NoError(t, err)
	require.True(t, manager.Current().Generation == 2)
}

func TestVersionManagerMutationFailureIsAtomic(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)

	mutationError := errors.New("mutation failed")
	{
		_, err := manager.Update(context.Background(), func(next *Manifest) error {
			next.EnableMmap = !next.EnableMmap
			return mutationError
		})
		require.ErrorIs(t, err, mutationError)
	}
	require.True(t, manager.Current().Generation == 1,
		"failed mutation changed the version")
	require.True(t, manifestCount(t, dir) == 1,
		"failed mutation changed the version")

	invalid := manager.Current()
	invalid.FormatVersion = DiskFormatVersion + 1
	{
		_, err := manager.Publish(context.Background(), invalid)
		require.ErrorIs(t, err, ErrUnsupportedFormatVersion)
	}
	require.True(t, manager.Current().Generation == 1,
		"invalid publish changed the version")
	require.True(t, manifestCount(t, dir) == 1,
		"invalid publish changed the version")
}

func TestVersionManagerContextCancellation(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	dir := filepath.Join(t.TempDir(), "not-created")
	{
		_, err := CreateVersionManager(canceled, dir, sampleManifest(1))
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := os.Stat(dir)
		require.ErrorIs(t, err, os.ErrNotExist)
	}

	dir = t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)
	{
		_, err := manager.Publish(canceled, manager.Current())
		require.ErrorIs(t, err, context.Canceled)
	}
	require.True(t, manager.Current().Generation == 1,
		"canceled publish changed the version")
	require.True(t, manifestCount(t, dir) == 1,
		"canceled publish changed the version")

	lock := flock.New(filepath.Join(dir, versionLockName))
	locked, err := lock.TryLock()
	require.NoError(t, err)
	require.True(t, locked)

	defer lock.Close()
	deadline, cancelDeadline := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancelDeadline()
	{
		_, err := manager.Publish(deadline, manager.Current())
		require.ErrorIs(t, err, context.DeadlineExceeded)
	}
}

func TestVersionManagerDetectsActiveCorruption(t *testing.T) {
	t.Run("current", func(t *testing.T) {
		dir := t.TempDir()
		{
			_, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
			require.NoError(t, err)
		}

		currentPath := filepath.Join(dir, currentFileName)
		current, err := os.ReadFile(currentPath)
		require.NoError(t, err)

		current[len(current)-1] ^= 1
		{
			err := os.WriteFile(currentPath, current, 0o600)
			require.NoError(t, err)
		}
		{
			_, err := OpenVersionManager(context.Background(), dir)
			require.ErrorIs(t, err, ErrManifestCorrupt)
		}
	})

	t.Run("manifest", func(t *testing.T) {
		dir := t.TempDir()
		manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
		require.NoError(t, err)

		manifestPath := filepath.Join(dir, manifestFileName(manager.Current().Generation))
		manifest, err := os.ReadFile(manifestPath)
		require.NoError(t, err)

		manifest[len(manifest)-1] ^= 1
		{
			err := os.WriteFile(manifestPath, manifest, 0o600)
			require.NoError(t, err)
		}
		{
			_, err := OpenVersionManager(context.Background(), dir)
			require.ErrorIs(t, err, ErrManifestCorrupt)
		}
	})

	t.Run("missing manifest", func(t *testing.T) {
		dir := t.TempDir()
		manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
		require.NoError(t, err)
		{
			err := os.Remove(filepath.Join(dir, manifestFileName(manager.Current().Generation)))
			require.NoError(t, err)
		}
		{
			_, err := OpenVersionManager(context.Background(), dir)
			require.ErrorIs(t, err, ErrManifestNotFound)
		}
	})
}

func TestVersionManagerAtomicCurrentForConcurrentReaders(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)

	done := make(chan struct{})
	errCh := make(chan error, 1)
	var wait sync.WaitGroup
	wait.Add(1)
	go func() {
		defer wait.Done()
		for {
			select {
			case <-done:
				return
			default:
			}
			opened, err := OpenVersionManager(context.Background(), dir)
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}
			if err := opened.Current().Validate(); err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}
		}
	}()
	for range 25 {
		if _, err := manager.Update(context.Background(), func(next *Manifest) error {
			next.DeleteSnapshotGeneration++
			return nil
		}); err != nil {
			close(done)
			wait.Wait()
			require.NoError(t, err)
		}
	}
	close(done)
	wait.Wait()
	select {
	case err := <-errCh:
		require.NoError(t, err)
	default:
	}
}

func TestVersionManagerArgumentValidation(t *testing.T) {
	{
		_, err := CreateVersionManager(nil, t.TempDir(), sampleManifest(1))
		require.Error(t, err,
			"nil create context succeeded")
	}
	{
		_, err := OpenVersionManager(nil, t.TempDir())
		require.Error(t, err,
			"nil open context succeeded")
	}
	{
		_, err := OpenVersionManager(context.Background(), filepath.Join(t.TempDir(), "missing"))
		require.ErrorIs(t, err, ErrManifestNotFound)
	}

	file := filepath.Join(t.TempDir(), "file")
	{
		err := os.WriteFile(file, nil, 0o600)
		require.NoError(t, err)
	}
	{
		_, err := OpenVersionManager(context.Background(), file)
		require.Error(t, err,
			"file collection path succeeded")
	}

	var manager *VersionManager
	{
		_, err := manager.Publish(context.Background(), Manifest{})
		require.Error(t, err,
			"nil manager publish succeeded")
	}
	{
		_, err := manager.Update(context.Background(), func(*Manifest) error { return nil })
		require.Error(t, err,
			"nil manager update succeeded")
	}

	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)
	{
		_, err := manager.Publish(nil, manager.Current())
		require.Error(t, err,
			"nil publish context succeeded")
	}
	{
		_, err := manager.Update(context.Background(), nil)
		require.Error(t, err,
			"nil mutation succeeded")
	}
}

func TestCurrentFrameTruncation(t *testing.T) {
	dir := t.TempDir()
	encoded, err := marshalCurrent(manifestFileName(1))
	require.NoError(t, err)

	for length := 0; length < len(encoded); length++ {
		{
			err := os.WriteFile(filepath.Join(dir, currentFileName), encoded[:length], 0o600)
			require.NoError(t, err)
		}
		{
			_, err := readCurrent(dir)
			require.ErrorIs(t, err, ErrManifestCorrupt)
		}
	}
	{
		err := os.WriteFile(filepath.Join(dir, currentFileName), encoded, 0o600)
		require.NoError(t, err)
	}

	name, err := readCurrent(dir)
	require.NoError(t, err)
	require.Equal(t, manifestFileName(1), name)
}

func TestManifestSchemaCanBeUpdated(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	require.NoError(t, err)

	updated, err := manager.Update(context.Background(), func(next *Manifest) error {
		next.Schema = json.RawMessage(`{"name":"articles","version":2}`)
		return nil
	})
	require.NoError(t, err)
	require.True(t, string(updated.Schema) == `{"name":"articles","version":2}`)
}

func manifestCount(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	require.NoError(t, err)

	count := 0
	for _, entry := range entries {
		if _, ok := parseManifestFileName(entry.Name()); ok {
			count++
		}
	}
	return count
}
