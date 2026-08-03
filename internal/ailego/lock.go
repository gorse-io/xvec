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

package ailego

import (
	"context"
	"errors"
	"os"
	"sync"
	"time"
)

// LockMode selects a shared reader lock or an exclusive writer lock.
type LockMode uint8

const (
	LockShared LockMode = iota + 1
	LockExclusive
)

// ErrLockUnavailable is returned by TryFileLock when another process or file
// handle holds an incompatible lock.
var ErrLockUnavailable = errors.New("ailego: file lock unavailable")

// FileLock holds an advisory whole-file lock. Lock files are deliberately not
// removed on Close because unlinking a live lock path can split lock domains.
type FileLock struct {
	mu     sync.Mutex
	file   *os.File
	closed bool
}

// TryFileLock attempts to acquire a lock without waiting.
func TryFileLock(path string, mode LockMode) (*FileLock, error) {
	lock, err := openFileLock(path, mode)
	if err != nil {
		return nil, err
	}
	acquired, err := tryPlatformLock(lock.file, mode)
	if err != nil {
		_ = lock.file.Close()
		return nil, err
	}
	if !acquired {
		_ = lock.file.Close()
		return nil, ErrLockUnavailable
	}
	return lock, nil
}

// AcquireFileLock waits for a compatible lock until ctx is canceled. It uses
// non-blocking OS lock attempts so context cancellation remains portable.
func AcquireFileLock(ctx context.Context, path string, mode LockMode) (*FileLock, error) {
	if ctx == nil {
		return nil, errors.New("ailego: nil lock context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	lock, err := openFileLock(path, mode)
	if err != nil {
		return nil, err
	}
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		acquired, err := tryPlatformLock(lock.file, mode)
		if err != nil {
			_ = lock.file.Close()
			return nil, err
		}
		if acquired {
			return lock, nil
		}

		select {
		case <-ctx.Done():
			_ = lock.file.Close()
			return nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

// Close releases the lock and closes its file handle. It is idempotent.
func (l *FileLock) Close() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	return errors.Join(unlockPlatformFile(l.file), l.file.Close())
}

func openFileLock(path string, mode LockMode) (*FileLock, error) {
	if mode != LockShared && mode != LockExclusive {
		return nil, errors.New("ailego: invalid file lock mode")
	}
	flags := os.O_CREATE | os.O_RDWR
	if mode == LockShared {
		// Existing lock files can be opened on a read-only collection. Fall
		// back to creation for the first shared lock on a new lock domain.
		if file, err := os.Open(path); err == nil {
			return &FileLock{file: file}, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
	}
	file, err := os.OpenFile(path, flags, 0o600)
	if err != nil {
		return nil, err
	}
	return &FileLock{file: file}, nil
}
