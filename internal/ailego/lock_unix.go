//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

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

	"golang.org/x/sys/unix"
)

func tryPlatformLock(file *os.File, mode LockMode) (bool, error) {
	how := unix.LOCK_NB
	if mode == LockShared {
		how |= unix.LOCK_SH
	} else {
		how |= unix.LOCK_EX
	}
	if err := unix.Flock(int(file.Fd()), how); err != nil {
		if errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func unlockPlatformFile(file *os.File) error {
	return unix.Flock(int(file.Fd()), unix.LOCK_UN)
}
