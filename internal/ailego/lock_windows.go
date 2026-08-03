//go:build windows

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
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

func tryPlatformLock(file *os.File, mode LockMode) (bool, error) {
	flags := uint32(windows.LOCKFILE_FAIL_IMMEDIATELY)
	if mode == LockExclusive {
		flags |= windows.LOCKFILE_EXCLUSIVE_LOCK
	}
	err := windows.LockFileEx(
		windows.Handle(file.Fd()), flags, 0, ^uint32(0), ^uint32(0),
		&windows.Overlapped{},
	)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return false, nil
	}
	return false, err
}

func unlockPlatformFile(file *os.File) error {
	return windows.UnlockFileEx(
		windows.Handle(file.Fd()), 0, ^uint32(0), ^uint32(0),
		&windows.Overlapped{},
	)
}
