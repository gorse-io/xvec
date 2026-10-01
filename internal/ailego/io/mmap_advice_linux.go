//go:build linux

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
	"golang.org/x/sys/unix"
	"os"
)

// DiscardReadOnlyMappedPages advises away complete pages of a verified,
// file-backed mapping. Borrowed slices stay valid and fault the immutable bytes
// back in on demand. It must never be used for anonymous or writable storage.
// Advice is best effort and does not evict the filesystem cache.
func DiscardReadOnlyMappedPages(data []byte, start, end int) {
	page := os.Getpagesize()
	start += (page - start%page) % page
	end -= end % page
	if start < end {
		_ = unix.Madvise(data[start:end], unix.MADV_DONTNEED)
	}
}
