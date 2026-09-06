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

//go:build windows

package core

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	diskANNWindowsShareMode       = windows.FILE_SHARE_READ | windows.FILE_SHARE_DELETE
	diskANNWindowsCompletionKey   = uintptr(1)
	diskANNWindowsCompletionWait  = 10
	diskANNWindowsPortConcurrency = uint32(math.MaxInt32)
)

var reOpenFileProc = windows.NewLazySystemDLL("kernel32.dll").NewProc("ReOpenFile")

type windowsDiskANNReader struct {
	mu     sync.Mutex
	stable windows.Handle
	file   windows.Handle
	port   windows.Handle
	size   int64
	closed bool
}

func openDiskANNDirectReader(path string) (diskANNReaderAt, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	name, err := windows.UTF16PtrFromString(absolute)
	if err != nil {
		return nil, err
	}
	stable, err := windows.CreateFile(
		name,
		windows.GENERIC_READ,
		diskANNWindowsShareMode,
		nil,
		windows.OPEN_EXISTING,
		windows.FILE_ATTRIBUTE_READONLY,
		0,
	)
	if err != nil {
		return nil, err
	}
	closeStable := true
	defer func() {
		if closeStable {
			_ = windows.CloseHandle(stable)
		}
	}()

	var info windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(stable, &info); err != nil {
		return nil, err
	}
	if info.FileAttributes&windows.FILE_ATTRIBUTE_DIRECTORY != 0 {
		return nil, errors.New("core: DiskANN reader requires a regular file")
	}
	size := int64(uint64(info.FileSizeHigh)<<32 | uint64(info.FileSizeLow))
	file, err := reOpenDiskANNFile(stable, windows.FILE_FLAG_NO_BUFFERING|windows.FILE_FLAG_OVERLAPPED)
	if err != nil {
		return nil, err
	}
	port, err := windows.CreateIoCompletionPort(file, 0, diskANNWindowsCompletionKey, diskANNWindowsPortConcurrency)
	if err != nil {
		_ = windows.CloseHandle(file)
		return nil, err
	}
	closeStable = false
	return &windowsDiskANNReader{stable: stable, file: file, port: port, size: size}, nil
}

func reOpenDiskANNFile(source windows.Handle, flags uint32) (windows.Handle, error) {
	handle, _, callErr := reOpenFileProc.Call(
		uintptr(source),
		uintptr(windows.GENERIC_READ),
		uintptr(diskANNWindowsShareMode),
		uintptr(flags),
	)
	if windows.Handle(handle) == windows.InvalidHandle {
		if callErr == nil || callErr == windows.ERROR_SUCCESS {
			callErr = windows.ERROR_INVALID_HANDLE
		}
		return windows.InvalidHandle, callErr
	}
	return windows.Handle(handle), nil
}

func (r *windowsDiskANNReader) Size() int64 { return r.size }

func (r *windowsDiskANNReader) ReadAt(buffer []byte, offset int64) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return 0, os.ErrClosed
	}
	if offset < 0 {
		return 0, errors.New("core: negative DiskANN read offset")
	}
	if len(buffer) == 0 {
		return 0, nil
	}
	buffered, err := reOpenDiskANNFile(r.stable, windows.FILE_ATTRIBUTE_NORMAL)
	if err != nil {
		return 0, err
	}
	file := os.NewFile(uintptr(buffered), "diskann")
	if file == nil {
		_ = windows.CloseHandle(buffered)
		return 0, errors.New("core: wrap DiskANN file handle")
	}
	n, readErr := file.ReadAt(buffer, offset)
	closeErr := file.Close()
	if readErr != nil {
		return n, readErr
	}
	if closeErr != nil {
		return n, closeErr
	}
	if n != len(buffer) {
		return n, io.EOF
	}
	return n, nil
}

func (r *windowsDiskANNReader) ReadBatchAt(
	ctx context.Context, requests []DiskANNReadRequest, buffers [][]byte,
) error {
	if ctx == nil {
		return errors.New("core: nil DiskANN IOCP context")
	}
	if len(requests) != len(buffers) {
		return errors.New("core: inconsistent DiskANN IOCP batch")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(requests) == 0 {
		return nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return os.ErrClosed
	}

	aligned := make([][]byte, len(requests))
	backing := make([][]byte, len(requests))
	overlapped := make([]windows.Overlapped, len(requests))
	byOverlapped := make(map[*windows.Overlapped]int, len(requests))
	for index, request := range requests {
		if request.Offset < 0 || request.Offset%DiskANNSectorSize != 0 || request.Length <= 0 ||
			request.Length%DiskANNSectorSize != 0 || uint64(request.Length) > math.MaxUint32 ||
			request.Offset > r.size-int64(request.Length) || len(buffers[index]) != request.Length {
			return fmt.Errorf("%w: invalid IOCP request %d", ErrDiskANNShortRead, index)
		}
		backing[index], aligned[index] = makeDiskANNAlignedBuffer(request.Length)
		overlapped[index].Offset = uint32(uint64(request.Offset))
		overlapped[index].OffsetHigh = uint32(uint64(request.Offset) >> 32)
		byOverlapped[&overlapped[index]] = index
	}

	var pinner runtime.Pinner
	for index := range aligned {
		pinner.Pin(&aligned[index][0])
	}
	pinner.Pin(&overlapped[0])
	defer pinner.Unpin()

	submitted := 0
	var resultErr error
	for index := range requests {
		if err := ctx.Err(); err != nil {
			resultErr = err
			break
		}
		var immediate uint32
		err := windows.ReadFile(r.file, aligned[index], &immediate, &overlapped[index])
		if err != nil && !errors.Is(err, windows.ERROR_IO_PENDING) {
			resultErr = fmt.Errorf("core: submit DiskANN IOCP request %d: %w", index, err)
			break
		}
		submitted++
	}
	if resultErr != nil && submitted != 0 {
		cancelDiskANNIO(r.file)
	}

	completed := 0
	canceled := false
	seen := make([]bool, submitted)
	for completed < submitted {
		if !canceled {
			if ctxErr := ctx.Err(); ctxErr != nil {
				resultErr = ctxErr
				cancelDiskANNIO(r.file)
				canceled = true
			}
		}
		var transferred uint32
		var key uintptr
		var completion *windows.Overlapped
		err := windows.GetQueuedCompletionStatus(r.port, &transferred, &key, &completion, diskANNWindowsCompletionWait)
		if completion == nil {
			if errors.Is(err, windows.WAIT_TIMEOUT) {
				continue
			}
			if resultErr == nil {
				resultErr = fmt.Errorf("core: wait for DiskANN IOCP completion: %w", err)
			}
			cancelDiskANNIO(r.file)
			canceled = true
			continue
		}

		index, found := byOverlapped[completion]
		if !found || index >= submitted || seen[index] || key != diskANNWindowsCompletionKey {
			if resultErr == nil {
				resultErr = errors.New("core: invalid DiskANN IOCP completion")
			}
			continue
		}
		seen[index] = true
		completed++
		if err != nil {
			if resultErr == nil {
				if errors.Is(err, windows.ERROR_HANDLE_EOF) || errors.Is(err, windows.ERROR_OPERATION_ABORTED) && ctx.Err() == nil {
					resultErr = fmt.Errorf("%w: IOCP request %d", ErrDiskANNShortRead, index)
				} else if ctxErr := ctx.Err(); ctxErr != nil {
					resultErr = ctxErr
				} else {
					resultErr = fmt.Errorf("core: DiskANN IOCP request %d: %w", index, err)
				}
			}
			continue
		}
		if transferred != uint32(requests[index].Length) && resultErr == nil {
			resultErr = fmt.Errorf("%w: IOCP request %d read %d of %d bytes", ErrDiskANNShortRead, index, transferred, requests[index].Length)
		}
	}

	if resultErr == nil {
		for index := range buffers {
			copy(buffers[index], aligned[index])
		}
	}
	runtime.KeepAlive(backing)
	runtime.KeepAlive(overlapped)
	return resultErr
}

func cancelDiskANNIO(handle windows.Handle) {
	if err := windows.CancelIoEx(handle, nil); err != nil && !errors.Is(err, windows.ERROR_NOT_FOUND) {
		return
	}
}

func makeDiskANNAlignedBuffer(size int) ([]byte, []byte) {
	backing := make([]byte, size+DiskANNSectorSize-1)
	address := uintptr(unsafe.Pointer(&backing[0]))
	offset := int((DiskANNSectorSize - address%DiskANNSectorSize) % DiskANNSectorSize)
	return backing, backing[offset : offset+size : offset+size]
}

func (r *windowsDiskANNReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	var result error
	if err := windows.CloseHandle(r.port); err != nil {
		result = err
	}
	if err := windows.CloseHandle(r.file); err != nil && result == nil {
		result = err
	}
	if err := windows.CloseHandle(r.stable); err != nil && result == nil {
		result = err
	}
	r.port, r.file, r.stable = 0, windows.InvalidHandle, windows.InvalidHandle
	return result
}
