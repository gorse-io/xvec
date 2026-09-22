//go:build amd64 && !noasm

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

// prefetchDenseHNSWNeighbors issues non-blocking cache hints. Reading each cache
// line here instead would serialize memory misses before distance computation.
func prefetchDenseHNSWNeighbors(vectors []float32, dimension int, neighbors []int, offset, lines uint32) {
	count := prefetchNeighborCount(len(neighbors), offset)
	if count == 0 || dimension <= 0 || len(vectors) == 0 {
		return
	}
	lineCount := min(normalizedPrefetchLines(lines, dimension*4), (dimension-1)/16+1)
	prefetchDenseVectors(&vectors[0], uintptr(dimension)*4, neighbors[:count], uintptr(lineCount))
}

//go:noescape
func prefetchDenseVectors(vectors *float32, stride uintptr, neighbors []int, lines uintptr)
