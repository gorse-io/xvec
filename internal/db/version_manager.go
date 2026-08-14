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
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/gofrs/flock"
	"github.com/gorse-io/xvec/internal/ailego/hash"
)

const (
	currentFileName  = "CURRENT"
	versionLockName  = ".version.lock"
	manifestPrefix   = "MANIFEST-"
	manifestDigits   = 20
	currentHeaderLen = 16
	maxCurrentSize   = 1024
)

var currentMagic = [8]byte{'Z', 'V', 'E', 'C', 'C', 'U', 'R', '1'}

// VersionManager owns the current immutable manifest snapshot. Publishing is
// serialized across goroutines and processes and never mutates an existing
// manifest file.
type VersionManager struct {
	dir         string
	mu          sync.RWMutex
	current     Manifest
	currentName string
}

// CreateVersionManager creates and atomically publishes an initial manifest.
// Existing unreferenced manifests from a failed create are skipped safely.
func CreateVersionManager(ctx context.Context, dir string, initial Manifest) (manager *VersionManager, err error) {
	if ctx == nil {
		return nil, errors.New("db: nil create version manager context")
	}
	if dir == "" {
		return nil, errors.New("db: empty collection directory")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("db: create collection directory: %w", err)
	}

	lock := flock.New(filepath.Join(dir, versionLockName))
	locked, err := lock.TryLockContext(ctx, fileLockRetryDelay)
	if err != nil {
		return nil, fmt.Errorf("db: lock manifest creation: %w", err)
	}
	if !locked {
		return nil, errors.New("db: manifest creation lock unavailable")
	}
	defer func() { err = errors.Join(err, lock.Close()) }()

	if _, err := readCurrent(dir); err == nil {
		return nil, ErrManifestExists
	} else if !errors.Is(err, ErrManifestNotFound) {
		return nil, err
	}
	generation, err := nextManifestGeneration(dir)
	if err != nil {
		return nil, err
	}
	initial = prepareManifest(initial, generation)
	name, committed, err := publishLocked(ctx, dir, "", initial)
	if committed {
		return &VersionManager{dir: dir, current: initial.Clone(), currentName: name}, err
	}
	if err != nil {
		return nil, err
	}
	return nil, errors.New("db: initial manifest was not committed")
}

// OpenVersionManager loads the manifest named by CURRENT. It never selects an
// unreferenced manifest, even when that file has a larger generation.
func OpenVersionManager(ctx context.Context, dir string) (*VersionManager, error) {
	if ctx == nil {
		return nil, errors.New("db: nil open version manager context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := os.Stat(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrManifestNotFound
		}
		return nil, fmt.Errorf("db: stat collection directory: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("db: collection path %q is not a directory", dir)
	}
	name, err := readCurrent(dir)
	if err != nil {
		return nil, err
	}
	manifest, err := readManifestFile(filepath.Join(dir, name))
	if err != nil {
		return nil, err
	}
	if manifestFileName(manifest.Generation) != name {
		return nil, fmt.Errorf("%w: CURRENT names %s but payload generation is %d", ErrManifestCorrupt, name, manifest.Generation)
	}
	return &VersionManager{dir: dir, current: manifest.Clone(), currentName: name}, nil
}

// Current returns an independent copy of the published version.
func (m *VersionManager) Current() Manifest {
	if m == nil {
		return Manifest{}
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.current.Clone()
}

// Publish atomically replaces CURRENT with an immutable copy of next. The
// manager assigns its generation. If another manager has published since this
// manager was opened, Publish returns ErrManifestConflict.
func (m *VersionManager) Publish(ctx context.Context, next Manifest) (Manifest, error) {
	if m == nil {
		return Manifest{}, errors.New("db: nil version manager")
	}
	if ctx == nil {
		return Manifest{}, errors.New("db: nil publish context")
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	return m.publishManagerLocked(ctx, next)
}

func (m *VersionManager) publishManagerLocked(ctx context.Context, next Manifest) (manifest Manifest, err error) {
	lock := flock.New(filepath.Join(m.dir, versionLockName))
	locked, err := lock.TryLockContext(ctx, fileLockRetryDelay)
	if err != nil {
		return Manifest{}, fmt.Errorf("db: lock manifest publication: %w", err)
	}
	if !locked {
		return Manifest{}, errors.New("db: manifest publication lock unavailable")
	}
	defer func() { err = errors.Join(err, lock.Close()) }()

	currentName, err := readCurrent(m.dir)
	if err != nil {
		return Manifest{}, err
	}
	if currentName != m.currentName {
		return Manifest{}, fmt.Errorf("%w: expected %s, found %s", ErrManifestConflict, m.currentName, currentName)
	}
	generation, err := nextManifestGeneration(m.dir)
	if err != nil {
		return Manifest{}, err
	}
	next = prepareManifest(next, generation)
	name, committed, err := publishLocked(ctx, m.dir, m.currentName, next)
	if committed {
		m.current = next.Clone()
		m.currentName = name
	}
	if err != nil {
		if committed {
			return next.Clone(), err
		}
		return Manifest{}, err
	}
	return next.Clone(), nil
}

// Update clones the current manifest, invokes mutate, and publishes the clone.
// A mutate error leaves memory and disk unchanged.
func (m *VersionManager) Update(ctx context.Context, mutate func(*Manifest) error) (Manifest, error) {
	if m == nil {
		return Manifest{}, errors.New("db: nil version manager")
	}
	if mutate == nil {
		return Manifest{}, errors.New("db: nil manifest mutation")
	}
	if ctx == nil {
		return Manifest{}, errors.New("db: nil update context")
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}
	base := m.Current()
	next := base.Clone()
	if err := mutate(&next); err != nil {
		return Manifest{}, err
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.current.Generation != base.Generation {
		return Manifest{}, fmt.Errorf(
			"%w: update started at generation %d, current is %d",
			ErrManifestConflict,
			base.Generation,
			m.current.Generation,
		)
	}
	return m.publishManagerLocked(ctx, next)
}

func prepareManifest(manifest Manifest, generation uint64) Manifest {
	manifest = manifest.Clone()
	if manifest.FormatVersion == 0 {
		manifest.FormatVersion = DiskFormatVersion
	}
	manifest.Generation = generation
	return manifest
}

func publishLocked(ctx context.Context, dir, expectedCurrent string, manifest Manifest) (name string, committed bool, err error) {
	if err := manifest.Validate(); err != nil {
		return "", false, err
	}
	if err := ctx.Err(); err != nil {
		return "", false, err
	}
	name = manifestFileName(manifest.Generation)
	encoded, err := MarshalManifest(manifest)
	if err != nil {
		return "", false, err
	}
	if err := writeExclusiveSynced(filepath.Join(dir, name), encoded); err != nil {
		if errors.Is(err, os.ErrExist) {
			return "", false, fmt.Errorf("%w: generation %d already exists", ErrManifestConflict, manifest.Generation)
		}
		return "", false, fmt.Errorf("db: write manifest %s: %w", name, err)
	}
	if err := syncDirectory(dir); err != nil {
		return "", false, fmt.Errorf("db: sync manifest directory: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return "", false, err
	}

	pointer, err := marshalCurrent(name)
	if err != nil {
		return "", false, err
	}
	temp, err := writeTempSynced(ctx, dir, ".current-*.tmp", pointer)
	if err != nil {
		return "", false, err
	}
	defer func() { _ = os.Remove(temp) }()
	if expectedCurrent == "" {
		err = installFileNoReplace(temp, filepath.Join(dir, currentFileName))
		if errors.Is(err, os.ErrExist) {
			return "", false, ErrManifestExists
		}
	} else {
		err = atomicReplaceFile(temp, filepath.Join(dir, currentFileName))
	}
	if err != nil {
		return "", false, fmt.Errorf("db: publish CURRENT: %w", err)
	}
	committed = true
	if err := syncDirectory(dir); err != nil {
		return name, true, fmt.Errorf("db: sync published CURRENT: %w", err)
	}
	return name, true, nil
}

func readManifestFile(name string) (Manifest, error) {
	file, err := os.Open(name)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Manifest{}, fmt.Errorf("%w: %s", ErrManifestNotFound, filepath.Base(name))
		}
		return Manifest{}, fmt.Errorf("db: open manifest: %w", err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return Manifest{}, fmt.Errorf("db: stat manifest: %w", err)
	}
	if info.Size() < manifestHeaderSize || info.Size() > manifestHeaderSize+maxManifestSize {
		return Manifest{}, fmt.Errorf("%w: invalid file size %d", ErrManifestCorrupt, info.Size())
	}
	encoded := make([]byte, int(info.Size()))
	if _, err := io.ReadFull(file, encoded); err != nil {
		return Manifest{}, fmt.Errorf("%w: read file: %v", ErrManifestCorrupt, err)
	}
	return UnmarshalManifest(encoded)
}

func manifestFileName(generation uint64) string {
	return fmt.Sprintf("%s%0*d", manifestPrefix, manifestDigits, generation)
}

func parseManifestFileName(name string) (uint64, bool) {
	if len(name) != len(manifestPrefix)+manifestDigits || !strings.HasPrefix(name, manifestPrefix) {
		return 0, false
	}
	generation, err := strconv.ParseUint(name[len(manifestPrefix):], 10, 64)
	return generation, err == nil && generation > 0
}

func nextManifestGeneration(dir string) (uint64, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, fmt.Errorf("db: scan manifest directory: %w", err)
	}
	var maximum uint64
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if generation, ok := parseManifestFileName(entry.Name()); ok {
			maximum = max(maximum, generation)
		}
	}
	if maximum == ^uint64(0) {
		return 0, errors.New("db: manifest generation exhausted")
	}
	return maximum + 1, nil
}

func marshalCurrent(name string) ([]byte, error) {
	if _, ok := parseManifestFileName(name); !ok {
		return nil, fmt.Errorf("%w: invalid manifest name %q", ErrManifestCorrupt, name)
	}
	encoded := make([]byte, currentHeaderLen+len(name))
	copy(encoded[:8], currentMagic[:])
	binary.LittleEndian.PutUint16(encoded[8:10], uint16(len(name)))
	binary.LittleEndian.PutUint16(encoded[10:12], currentHeaderLen)
	binary.LittleEndian.PutUint32(encoded[12:16], hashutil.CRC32C([]byte(name)))
	copy(encoded[currentHeaderLen:], name)
	return encoded, nil
}

func readCurrent(dir string) (string, error) {
	file, err := openCurrentFile(filepath.Join(dir, currentFileName))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", ErrManifestNotFound
		}
		return "", fmt.Errorf("db: open CURRENT: %w", err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("db: stat CURRENT: %w", err)
	}
	if info.Size() < currentHeaderLen || info.Size() > maxCurrentSize {
		return "", fmt.Errorf("%w: invalid CURRENT size %d", ErrManifestCorrupt, info.Size())
	}
	encoded := make([]byte, int(info.Size()))
	if _, err := io.ReadFull(file, encoded); err != nil {
		return "", fmt.Errorf("%w: read CURRENT: %v", ErrManifestCorrupt, err)
	}
	if !bytes.Equal(encoded[:8], currentMagic[:]) {
		return "", fmt.Errorf("%w: invalid CURRENT magic", ErrManifestCorrupt)
	}
	if headerLength := binary.LittleEndian.Uint16(encoded[10:12]); headerLength != currentHeaderLen {
		return "", fmt.Errorf("%w: invalid CURRENT header size %d", ErrManifestCorrupt, headerLength)
	}
	nameLength := int(binary.LittleEndian.Uint16(encoded[8:10]))
	if nameLength != len(encoded)-currentHeaderLen {
		return "", fmt.Errorf("%w: invalid CURRENT payload length", ErrManifestCorrupt)
	}
	name := string(encoded[currentHeaderLen:])
	if actual, expected := hashutil.CRC32C([]byte(name)), binary.LittleEndian.Uint32(encoded[12:16]); actual != expected {
		return "", fmt.Errorf("%w: CURRENT checksum got %08x, want %08x", ErrManifestCorrupt, actual, expected)
	}
	if _, ok := parseManifestFileName(name); !ok {
		return "", fmt.Errorf("%w: invalid CURRENT manifest name %q", ErrManifestCorrupt, name)
	}
	return name, nil
}

func writeExclusiveSynced(name string, contents []byte) (err error) {
	file, err := os.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	remove := true
	defer func() {
		if file != nil {
			closeErr := file.Close()
			if err == nil && closeErr != nil {
				err = closeErr
			}
		}
		if remove {
			_ = os.Remove(name)
		}
	}()
	if err = writeFull(file, contents); err != nil {
		return err
	}
	if err = file.Sync(); err != nil {
		return err
	}
	if err = file.Close(); err != nil {
		return err
	}
	file = nil
	remove = false
	return nil
}

func writeTempSynced(ctx context.Context, dir, pattern string, contents []byte) (name string, err error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	file, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return "", fmt.Errorf("db: create publication temp file: %w", err)
	}
	name = file.Name()
	remove := true
	defer func() {
		if file != nil {
			closeErr := file.Close()
			if err == nil && closeErr != nil {
				err = closeErr
			}
		}
		if remove {
			_ = os.Remove(name)
		}
	}()
	if err = writeFull(file, contents); err != nil {
		return "", fmt.Errorf("db: write publication temp file: %w", err)
	}
	if err = file.Sync(); err != nil {
		return "", fmt.Errorf("db: sync publication temp file: %w", err)
	}
	if err = file.Close(); err != nil {
		return "", fmt.Errorf("db: close publication temp file: %w", err)
	}
	file = nil
	if err = ctx.Err(); err != nil {
		return "", err
	}
	remove = false
	return name, nil
}

func writeFull(writer io.Writer, contents []byte) error {
	for len(contents) > 0 {
		written, err := writer.Write(contents)
		if written > 0 {
			contents = contents[written:]
		}
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}
