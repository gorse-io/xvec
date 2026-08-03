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
	"path/filepath"
	"testing"
	"time"
)

func TestFileLockCompatibility(t *testing.T) {
	path := filepath.Join(t.TempDir(), "collection.lock")
	sharedOne, err := TryFileLock(path, LockShared)
	if err != nil {
		t.Fatal(err)
	}
	defer sharedOne.Close()
	sharedTwo, err := TryFileLock(path, LockShared)
	if err != nil {
		t.Fatalf("second shared lock: %v", err)
	}
	if err := sharedTwo.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := TryFileLock(path, LockExclusive); !errors.Is(err, ErrLockUnavailable) {
		t.Fatalf("exclusive lock error = %v", err)
	}

	if err := sharedOne.Close(); err != nil {
		t.Fatal(err)
	}
	exclusive, err := TryFileLock(path, LockExclusive)
	if err != nil {
		t.Fatal(err)
	}
	defer exclusive.Close()
	if _, err := TryFileLock(path, LockShared); !errors.Is(err, ErrLockUnavailable) {
		t.Fatalf("shared lock error = %v", err)
	}
}

func TestAcquireFileLockHonorsContext(t *testing.T) {
	path := filepath.Join(t.TempDir(), "collection.lock")
	first, err := TryFileLock(path, LockExclusive)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if _, err := AcquireFileLock(ctx, path, LockExclusive); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("AcquireFileLock error = %v", err)
	}
}

func TestAcquireFileLockWithCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	path := filepath.Join(t.TempDir(), "collection.lock")
	if _, err := AcquireFileLock(ctx, path, LockExclusive); !errors.Is(err, context.Canceled) {
		t.Fatalf("AcquireFileLock error = %v", err)
	}
}
