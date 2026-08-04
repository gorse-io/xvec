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
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestFileLockCompatibility(t *testing.T) {
	path := filepath.Join(t.TempDir(), "collection.lock")
	sharedOne, err := TryFileLock(path, LockShared)
	require.NoError(t, err)

	defer sharedOne.Close()
	sharedTwo, err := TryFileLock(path, LockShared)
	require.NoError(t, err)
	{
		err := sharedTwo.Close()
		require.NoError(t, err)
	}
	{
		_, err := TryFileLock(path, LockExclusive)
		require.ErrorIs(t, err, ErrLockUnavailable)
	}
	{
		err := sharedOne.Close()
		require.NoError(t, err)
	}

	exclusive, err := TryFileLock(path, LockExclusive)
	require.NoError(t, err)

	defer exclusive.Close()
	{
		_, err := TryFileLock(path, LockShared)
		require.ErrorIs(t, err, ErrLockUnavailable)
	}
}

func TestAcquireFileLockHonorsContext(t *testing.T) {
	path := filepath.Join(t.TempDir(), "collection.lock")
	first, err := TryFileLock(path, LockExclusive)
	require.NoError(t, err)

	defer first.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	{
		_, err := AcquireFileLock(ctx, path, LockExclusive)
		require.ErrorIs(t, err, context.DeadlineExceeded)
	}
}

func TestAcquireFileLockWithCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	path := filepath.Join(t.TempDir(), "collection.lock")
	{
		_, err := AcquireFileLock(ctx, path, LockExclusive)
		require.ErrorIs(t, err, context.Canceled)
	}
}
