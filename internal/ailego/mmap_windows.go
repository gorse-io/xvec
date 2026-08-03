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
	"unsafe"

	"golang.org/x/sys/windows"
)

func mapReadOnly(file *os.File, size int) ([]byte, func() error, error) {
	mapping, err := windows.CreateFileMapping(
		windows.Handle(file.Fd()), nil, windows.PAGE_READONLY, 0, 0, nil,
	)
	if err != nil {
		return nil, nil, err
	}
	address, err := windows.MapViewOfFile(mapping, windows.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		_ = windows.CloseHandle(mapping)
		return nil, nil, err
	}
	data := unsafe.Slice((*byte)(unsafe.Pointer(address)), size)
	return data, func() error {
		return errors.Join(
			windows.UnmapViewOfFile(address),
			windows.CloseHandle(mapping),
		)
	}, nil
}
