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

package ioutil

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenReaderAt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segment.dat")
	content := bytes.Repeat([]byte("zvec"), 1024)
	{
		err := os.WriteFile(path, content, 0o600)
		require.NoError(t, err)
	}

	for _, useMMap := range []bool{false, true} {
		t.Run(map[bool]string{false: "read_at", true: "mmap"}[useMMap], func(t *testing.T) {
			reader, err := OpenReaderAt(path, useMMap)
			require.NoError(t, err)
			require.Equal(t, int64(len(content)), reader.Size())

			var wg sync.WaitGroup
			for worker := range 8 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					dst := make([]byte, 16)
					off := int64(worker * 32)
					if err := ReadFullAt(reader, dst, off); !assert.NoError(t, err) {
						return
					}
					assert.True(t, bytes.Equal(dst, content[off:off+16]))
				}()
			}
			wg.Wait()

			tail := make([]byte, 8)
			{
				n, err := reader.ReadAt(tail, int64(len(content)-4))
				require.True(t, n == 4)
				require.Same(t, io.EOF, err)
			}
			{
				err := reader.Close()
				require.NoError(t, err)
			}

			if useMMap {
				require.Equal(t, int64(len(content)), reader.Size())
				{
					_, err := reader.ReadAt(make([]byte, 1), 0)
					require.Same(t, os.ErrClosed, err)
				}
				{
					err := reader.Close()
					require.NoError(t, err)
				}
			}
		})
	}
}

func TestOpenReaderAtEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty")
	{
		err := os.WriteFile(path, nil, 0o600)
		require.NoError(t, err)
	}

	reader, err := OpenReaderAt(path, true)
	require.NoError(t, err)

	defer func() { require.NoError(t, reader.Close()) }()
	require.True(t, reader.Size() == 0)
}
