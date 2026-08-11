//go:build windows

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

package ailego

import "golang.org/x/sys/windows"

func atomicReplaceFile(source, destination string) error {
	sourceUTF16, err := windows.UTF16PtrFromString(source)
	if err != nil {
		return err
	}
	destinationUTF16, err := windows.UTF16PtrFromString(destination)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(
		sourceUTF16,
		destinationUTF16,
		windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH,
	)
}

// MoveFileEx with MOVEFILE_WRITE_THROUGH flushes the replacement. Directory
// handles cannot be portably opened for fsync on Windows.
func syncDirectory(string) error { return nil }
