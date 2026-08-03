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
)

// ReadFullAt fills dst from r starting at off. It accepts short reads and
// returns io.ErrNoProgress when a broken ReaderAt returns neither data nor an
// error.
func ReadFullAt(r io.ReaderAt, dst []byte, off int64) error {
	if off < 0 {
		return errors.New("ailego: negative read offset")
	}
	for len(dst) > 0 {
		n, err := r.ReadAt(dst, off)
		if n > 0 {
			dst = dst[n:]
			off += int64(n)
		}
		if len(dst) == 0 {
			return nil
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrNoProgress
		}
	}
	return nil
}

// WriteFullAt writes all of src to w starting at off. It accepts short writes
// and returns io.ErrShortWrite when a broken WriterAt makes no progress.
func WriteFullAt(w io.WriterAt, src []byte, off int64) error {
	if off < 0 {
		return errors.New("ailego: negative write offset")
	}
	for len(src) > 0 {
		n, err := w.WriteAt(src, off)
		if n > 0 {
			src = src[n:]
			off += int64(n)
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}
