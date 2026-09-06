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

package core

import (
	"context"
	"errors"
	"io"

	"github.com/gorse-io/xvec/internal/ailego/io"
)

type diskANNReaderAt interface {
	io.ReaderAt
	io.Closer
	Size() int64
}

type diskANNBatchReader interface {
	ReadBatchAt(context.Context, []DiskANNReadRequest, [][]byte) error
}

func openDiskANNReaderAt(path string, useMmap bool) (diskANNReaderAt, error) {
	if useMmap {
		return ioutil.OpenReaderAt(path, true)
	}
	return openDiskANNDirectReader(path)
}

type diskANNSectionReader struct {
	reader io.ReaderAt
	base   int64
	size   int64
}

type diskANNBatchSectionReader struct {
	*diskANNSectionReader
}

func newDiskANNSectionReader(reader io.ReaderAt, base, size int64) io.ReaderAt {
	section := &diskANNSectionReader{reader: reader, base: base, size: size}
	if _, ok := reader.(diskANNBatchReader); ok {
		return &diskANNBatchSectionReader{diskANNSectionReader: section}
	}
	return section
}

func (r *diskANNSectionReader) ReadAt(buffer []byte, offset int64) (int, error) {
	if offset < 0 {
		return 0, errors.New("core: negative DiskANN section read offset")
	}
	if offset >= r.size {
		if len(buffer) == 0 && offset == r.size {
			return 0, nil
		}
		return 0, io.EOF
	}
	limit := int64(len(buffer))
	if limit > r.size-offset {
		limit = r.size - offset
	}
	n, err := r.reader.ReadAt(buffer[:limit], r.base+offset)
	if int64(len(buffer)) > limit && err == nil {
		err = io.EOF
	}
	return n, err
}

func (r *diskANNBatchSectionReader) ReadBatchAt(
	ctx context.Context, requests []DiskANNReadRequest, buffers [][]byte,
) error {
	batch, ok := r.reader.(diskANNBatchReader)
	if !ok {
		return errors.New("core: DiskANN reader does not support batch reads")
	}
	if len(requests) != len(buffers) {
		return errors.New("core: inconsistent DiskANN batch")
	}
	translated := make([]DiskANNReadRequest, len(requests))
	for index, request := range requests {
		if request.Offset < 0 || request.Length <= 0 || int64(request.Length) > r.size-request.Offset {
			return ErrDiskANNShortRead
		}
		translated[index] = DiskANNReadRequest{Offset: r.base + request.Offset, Length: request.Length}
	}
	return batch.ReadBatchAt(ctx, translated, buffers)
}
