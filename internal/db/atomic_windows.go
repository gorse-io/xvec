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

package db

import (
	"errors"
	"os"
	"time"

	"golang.org/x/sys/windows"
)

func atomicReplaceFile(source, destination string) error {
	var err error
	for range 100 {
		err = moveFile(source, destination, windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH)
		if err == nil {
			return nil
		}
		if !errors.Is(err, windows.ERROR_ACCESS_DENIED) && !errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
			return err
		}
		time.Sleep(time.Millisecond)
	}
	return err
}

func installFileNoReplace(source, destination string) error {
	err := moveFile(source, destination, windows.MOVEFILE_WRITE_THROUGH)
	if errors.Is(err, windows.ERROR_ALREADY_EXISTS) || errors.Is(err, windows.ERROR_FILE_EXISTS) {
		return os.ErrExist
	}
	return err
}

func moveFile(source, destination string, flags uint32) error {
	sourceUTF16, err := windows.UTF16PtrFromString(source)
	if err != nil {
		return err
	}
	destinationUTF16, err := windows.UTF16PtrFromString(destination)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(sourceUTF16, destinationUTF16, flags)
}

// MoveFileEx with MOVEFILE_WRITE_THROUGH flushes the rename. Opening a
// directory for fsync is not portable on Windows.
func syncDirectory(string) error { return nil }
