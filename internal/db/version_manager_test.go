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
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestVersionManagerCreatePublishAndOpen(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "collection")
	initial := sampleManifest(99)
	initial.FormatVersion = 0
	manager, err := CreateVersionManager(context.Background(), dir, initial)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	current := manager.Current()
	if current.FormatVersion != DiskFormatVersion || current.Generation != 1 {
		t.Fatalf("initial version = %#v", current)
	}

	current.Schema[0] = '['
	current.PersistedSegments[0].Files[0] = "changed"
	if err := manager.Current().Validate(); err != nil {
		t.Fatalf("Current returned shared state: %v", err)
	}

	published, err := manager.Update(context.Background(), func(next *Manifest) error {
		next.EnableMmap = false
		next.IDMapGeneration++
		return nil
	})
	if err != nil {
		t.Fatalf("update: %v", err)
	}
	if published.Generation != 2 || published.EnableMmap || published.IDMapGeneration != 6 {
		t.Fatalf("published = %#v", published)
	}
	published.Schema[0] = '['
	if err := manager.Current().Validate(); err != nil {
		t.Fatalf("Publish returned shared state: %v", err)
	}

	reopened, err := OpenVersionManager(context.Background(), dir)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if !reflect.DeepEqual(reopened.Current(), manager.Current()) {
		t.Fatalf("reopened = %#v, current = %#v", reopened.Current(), manager.Current())
	}
	if _, err := CreateVersionManager(context.Background(), dir, initial); !errors.Is(err, ErrManifestExists) {
		t.Fatalf("second create error = %v", err)
	}
	if count := manifestCount(t, dir); count != 2 {
		t.Fatalf("manifest count = %d, want 2", count)
	}
}

func TestVersionManagerRejectsStalePublisher(t *testing.T) {
	dir := t.TempDir()
	first, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}
	stale, err := OpenVersionManager(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Update(context.Background(), func(next *Manifest) error {
		next.DeleteSnapshotGeneration++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := stale.Publish(context.Background(), stale.Current()); !errors.Is(err, ErrManifestConflict) {
		t.Fatalf("stale publish error = %v", err)
	}
	if stale.Current().Generation != 1 {
		t.Fatalf("stale in-memory generation = %d", stale.Current().Generation)
	}
	if count := manifestCount(t, dir); count != 2 {
		t.Fatalf("stale publish created a manifest; count = %d", count)
	}
}

func TestVersionManagerRejectsConcurrentUpdateFromSameSnapshot(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}

	ready := sync.WaitGroup{}
	ready.Add(2)
	release := make(chan struct{})
	errorsByUpdate := make(chan error, 2)
	for range 2 {
		go func() {
			_, err := manager.Update(context.Background(), func(next *Manifest) error {
				next.IDMapGeneration++
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
			t.Fatalf("update error = %v", err)
		}
	}
	if succeeded != 1 || conflicted != 1 {
		t.Fatalf("succeeded = %d, conflicted = %d", succeeded, conflicted)
	}
	if current := manager.Current(); current.Generation != 2 || current.IDMapGeneration != 6 {
		t.Fatalf("current = %#v", current)
	}
}

func TestVersionManagerIgnoresUnpublishedFiles(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}

	orphan := sampleManifest(2)
	encoded, err := MarshalManifest(orphan)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeExclusiveSynced(filepath.Join(dir, manifestFileName(2)), encoded); err != nil {
		t.Fatal(err)
	}
	pointer, err := marshalCurrent(manifestFileName(2))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".current-crash.tmp"), pointer, 0o600); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenVersionManager(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if reopened.Current().Generation != 1 {
		t.Fatalf("opened unpublished generation %d", reopened.Current().Generation)
	}
	published, err := manager.Publish(context.Background(), manager.Current())
	if err != nil {
		t.Fatal(err)
	}
	if published.Generation != 3 {
		t.Fatalf("generation after orphan = %d, want 3", published.Generation)
	}

	if err := os.WriteFile(filepath.Join(dir, manifestFileName(4)), []byte("partial"), 0o600); err != nil {
		t.Fatal(err)
	}
	reopened, err = OpenVersionManager(context.Background(), dir)
	if err != nil {
		t.Fatalf("partial orphan affected recovery: %v", err)
	}
	if reopened.Current().Generation != 3 {
		t.Fatalf("opened partial orphan generation %d", reopened.Current().Generation)
	}
	published, err = manager.Publish(context.Background(), manager.Current())
	if err != nil {
		t.Fatal(err)
	}
	if published.Generation != 5 {
		t.Fatalf("generation after partial orphan = %d, want 5", published.Generation)
	}
}

func TestVersionManagerCreateSkipsFailedInitialManifest(t *testing.T) {
	dir := t.TempDir()
	orphan := sampleManifest(1)
	encoded, err := MarshalManifest(orphan)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeExclusiveSynced(filepath.Join(dir, manifestFileName(1)), encoded); err != nil {
		t.Fatal(err)
	}
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(0))
	if err != nil {
		t.Fatal(err)
	}
	if manager.Current().Generation != 2 {
		t.Fatalf("created generation = %d, want 2", manager.Current().Generation)
	}
}

func TestVersionManagerMutationFailureIsAtomic(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}
	mutationError := errors.New("mutation failed")
	if _, err := manager.Update(context.Background(), func(next *Manifest) error {
		next.EnableMmap = !next.EnableMmap
		return mutationError
	}); !errors.Is(err, mutationError) {
		t.Fatalf("mutation error = %v", err)
	}
	if manager.Current().Generation != 1 || manifestCount(t, dir) != 1 {
		t.Fatal("failed mutation changed the version")
	}

	invalid := manager.Current()
	invalid.FormatVersion = DiskFormatVersion + 1
	if _, err := manager.Publish(context.Background(), invalid); !errors.Is(err, ErrUnsupportedFormatVersion) {
		t.Fatalf("invalid publish error = %v", err)
	}
	if manager.Current().Generation != 1 || manifestCount(t, dir) != 1 {
		t.Fatal("invalid publish changed the version")
	}
}

func TestVersionManagerContextCancellation(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	dir := filepath.Join(t.TempDir(), "not-created")
	if _, err := CreateVersionManager(canceled, dir, sampleManifest(1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled create error = %v", err)
	}
	if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("canceled create made directory: %v", err)
	}

	dir = t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Publish(canceled, manager.Current()); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled publish error = %v", err)
	}
	if manager.Current().Generation != 1 || manifestCount(t, dir) != 1 {
		t.Fatal("canceled publish changed the version")
	}

	lock, err := ailego.AcquireFileLock(context.Background(), filepath.Join(dir, versionLockName), ailego.LockExclusive)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	deadline, cancelDeadline := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancelDeadline()
	if _, err := manager.Publish(deadline, manager.Current()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("blocked publish error = %v", err)
	}
}

func TestVersionManagerDetectsActiveCorruption(t *testing.T) {
	t.Run("current", func(t *testing.T) {
		dir := t.TempDir()
		if _, err := CreateVersionManager(context.Background(), dir, sampleManifest(1)); err != nil {
			t.Fatal(err)
		}
		currentPath := filepath.Join(dir, currentFileName)
		current, err := os.ReadFile(currentPath)
		if err != nil {
			t.Fatal(err)
		}
		current[len(current)-1] ^= 1
		if err := os.WriteFile(currentPath, current, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenVersionManager(context.Background(), dir); !errors.Is(err, ErrManifestCorrupt) {
			t.Fatalf("corrupt CURRENT error = %v", err)
		}
	})

	t.Run("manifest", func(t *testing.T) {
		dir := t.TempDir()
		manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
		if err != nil {
			t.Fatal(err)
		}
		manifestPath := filepath.Join(dir, manifestFileName(manager.Current().Generation))
		manifest, err := os.ReadFile(manifestPath)
		if err != nil {
			t.Fatal(err)
		}
		manifest[len(manifest)-1] ^= 1
		if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenVersionManager(context.Background(), dir); !errors.Is(err, ErrManifestCorrupt) {
			t.Fatalf("corrupt manifest error = %v", err)
		}
	})

	t.Run("missing manifest", func(t *testing.T) {
		dir := t.TempDir()
		manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(filepath.Join(dir, manifestFileName(manager.Current().Generation))); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenVersionManager(context.Background(), dir); !errors.Is(err, ErrManifestNotFound) {
			t.Fatalf("missing manifest error = %v", err)
		}
	})
}

func TestVersionManagerAtomicCurrentForConcurrentReaders(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}

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
			next.IDMapGeneration++
			return nil
		}); err != nil {
			close(done)
			wait.Wait()
			t.Fatal(err)
		}
	}
	close(done)
	wait.Wait()
	select {
	case err := <-errCh:
		t.Fatalf("concurrent recovery observed partial publication: %v", err)
	default:
	}
}

func TestVersionManagerArgumentValidation(t *testing.T) {
	if _, err := CreateVersionManager(nil, t.TempDir(), sampleManifest(1)); err == nil {
		t.Fatal("nil create context succeeded")
	}
	if _, err := OpenVersionManager(nil, t.TempDir()); err == nil {
		t.Fatal("nil open context succeeded")
	}
	if _, err := OpenVersionManager(context.Background(), filepath.Join(t.TempDir(), "missing")); !errors.Is(err, ErrManifestNotFound) {
		t.Fatalf("missing directory error = %v", err)
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenVersionManager(context.Background(), file); err == nil {
		t.Fatal("file collection path succeeded")
	}
	var manager *VersionManager
	if _, err := manager.Publish(context.Background(), Manifest{}); err == nil {
		t.Fatal("nil manager publish succeeded")
	}
	if _, err := manager.Update(context.Background(), func(*Manifest) error { return nil }); err == nil {
		t.Fatal("nil manager update succeeded")
	}
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Publish(nil, manager.Current()); err == nil {
		t.Fatal("nil publish context succeeded")
	}
	if _, err := manager.Update(context.Background(), nil); err == nil {
		t.Fatal("nil mutation succeeded")
	}
}

func TestCurrentFrameTruncation(t *testing.T) {
	dir := t.TempDir()
	encoded, err := marshalCurrent(manifestFileName(1))
	if err != nil {
		t.Fatal(err)
	}
	for length := 0; length < len(encoded); length++ {
		if err := os.WriteFile(filepath.Join(dir, currentFileName), encoded[:length], 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := readCurrent(dir); !errors.Is(err, ErrManifestCorrupt) {
			t.Fatalf("truncation at %d error = %v", length, err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, currentFileName), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	name, err := readCurrent(dir)
	if err != nil || name != manifestFileName(1) {
		t.Fatalf("complete CURRENT = %q, %v", name, err)
	}
}

func TestManifestSchemaCanBeUpdated(t *testing.T) {
	dir := t.TempDir()
	manager, err := CreateVersionManager(context.Background(), dir, sampleManifest(1))
	if err != nil {
		t.Fatal(err)
	}
	updated, err := manager.Update(context.Background(), func(next *Manifest) error {
		next.Schema = json.RawMessage(`{"name":"articles","version":2}`)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(updated.Schema) != `{"name":"articles","version":2}` {
		t.Fatalf("schema = %s", updated.Schema)
	}
}

func manifestCount(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, entry := range entries {
		if _, ok := parseManifestFileName(entry.Name()); ok {
			count++
		}
	}
	return count
}
