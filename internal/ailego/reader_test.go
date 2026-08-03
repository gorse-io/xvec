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
	"bytes"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestOpenReaderAt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "segment.dat")
	content := bytes.Repeat([]byte("zvec"), 1024)
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}

	for _, useMMap := range []bool{false, true} {
		t.Run(map[bool]string{false: "read_at", true: "mmap"}[useMMap], func(t *testing.T) {
			reader, err := OpenReaderAt(path, useMMap)
			if err != nil {
				t.Fatal(err)
			}
			if reader.Size() != int64(len(content)) {
				t.Fatalf("size = %d", reader.Size())
			}

			var wg sync.WaitGroup
			for worker := range 8 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					dst := make([]byte, 16)
					off := int64(worker * 32)
					if err := ReadFullAt(reader, dst, off); err != nil {
						t.Error(err)
						return
					}
					if !bytes.Equal(dst, content[off:off+16]) {
						t.Errorf("data at %d differs", off)
					}
				}()
			}
			wg.Wait()

			tail := make([]byte, 8)
			if n, err := reader.ReadAt(tail, int64(len(content)-4)); n != 4 || err != io.EOF {
				t.Fatalf("tail read = (%d, %v)", n, err)
			}
			if err := reader.Close(); err != nil {
				t.Fatal(err)
			}
			if useMMap {
				if reader.Size() != int64(len(content)) {
					t.Fatalf("size after close = %d", reader.Size())
				}
				if _, err := reader.ReadAt(make([]byte, 1), 0); err != os.ErrClosed {
					t.Fatalf("read after close error = %v", err)
				}
				if err := reader.Close(); err != nil {
					t.Fatalf("second close: %v", err)
				}
			}
		})
	}
}

func TestOpenReaderAtEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	reader, err := OpenReaderAt(path, true)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	if reader.Size() != 0 {
		t.Fatalf("size = %d", reader.Size())
	}
}
