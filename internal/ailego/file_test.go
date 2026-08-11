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

import (
	"bytes"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

type shortReaderAt struct{ data []byte }

func (r shortReaderAt) ReadAt(dst []byte, off int64) (int, error) {
	if off >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := min(2, len(dst), len(r.data)-int(off))
	copy(dst, r.data[int(off):int(off)+n])
	return n, nil
}

type shortWriterAt struct{ data []byte }

func (w shortWriterAt) WriteAt(src []byte, off int64) (int, error) {
	n := min(2, len(src), len(w.data)-int(off))
	copy(w.data[int(off):], src[:n])
	return n, nil
}

func TestFullAt(t *testing.T) {
	dst := make([]byte, 6)
	{
		err := ReadFullAt(shortReaderAt{data: []byte("0123456789")}, dst, 2)
		require.NoError(t, err)
	}
	require.True(t, bytes.Equal(dst, []byte("234567")))

	written := make([]byte, 10)
	{
		err := WriteFullAt(shortWriterAt{data: written}, []byte("abcdef"), 2)
		require.NoError(t, err)
	}
	require.True(t, bytes.Equal(written[2:8], []byte("abcdef")))
	{
		err := ReadFullAt(shortReaderAt{}, make([]byte, 1), 0)
		require.Same(t, io.EOF, err)
	}
}
