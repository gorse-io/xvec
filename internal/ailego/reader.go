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
	"io"
	"os"
	"sync"

	mmap "github.com/blevesearch/mmap-go"
)

// ReaderAt is a sized, closeable random-access reader for immutable files.
// Implementations permit concurrent ReadAt calls.
type ReaderAt interface {
	io.ReaderAt
	io.Closer
	Size() int64
}

// OpenReaderAt opens path as an immutable random-access reader. When useMMap
// is true, a read-only shared mapping is required; mapping failures are
// returned rather than silently falling back to buffered I/O. Empty files use
// os.File.ReadAt because operating systems cannot map a zero-length region.
func OpenReaderAt(path string, useMMap bool) (ReaderAt, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	if !info.Mode().IsRegular() {
		_ = file.Close()
		return nil, errors.New("ailego: random-access reader requires a regular file")
	}
	if !useMMap || info.Size() == 0 {
		return &fileReaderAt{file: file, size: info.Size()}, nil
	}
	if uint64(info.Size()) > uint64(maxInt()) {
		_ = file.Close()
		return nil, errors.New("ailego: file is too large to memory map")
	}

	data, err := mmap.MapRegion(file, int(info.Size()), mmap.RDONLY, 0, 0)
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	if err := file.Close(); err != nil {
		_ = data.Unmap()
		return nil, err
	}
	return &mmapReaderAt{data: data, size: info.Size()}, nil
}

type fileReaderAt struct {
	file *os.File
	size int64
}

func (r *fileReaderAt) ReadAt(dst []byte, off int64) (int, error) {
	return r.file.ReadAt(dst, off)
}

func (r *fileReaderAt) Size() int64 { return r.size }

func (r *fileReaderAt) Close() error { return r.file.Close() }

type mmapReaderAt struct {
	mu     sync.RWMutex
	data   mmap.MMap
	size   int64
	closed bool
}

func (r *mmapReaderAt) ReadAt(dst []byte, off int64) (int, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return 0, os.ErrClosed
	}
	if off < 0 {
		return 0, errors.New("ailego: negative read offset")
	}
	if len(dst) == 0 {
		return 0, nil
	}
	if off >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(dst, r.data[int(off):])
	if n != len(dst) {
		return n, io.EOF
	}
	return n, nil
}

func (r *mmapReaderAt) Size() int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.size
}

func (r *mmapReaderAt) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	return r.data.Unmap()
}

func maxInt() int { return int(^uint(0) >> 1) }
