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
	"math"
	"runtime"
)

// The Go runtime has no portable non-faulting prefetch intrinsic. These
// helpers synchronously warm the bounded cache-line prefix requested by the
// hint and never change candidate ordering or admission.
func prefetchDenseHNSWNeighbors[T ~int | ~uint32](vectors []float32, dimension int, neighbors []T, offset, lines uint32) {
	count := prefetchNeighborCount(len(neighbors), offset)
	if count == 0 || dimension <= 0 {
		return
	}
	lineCount := normalizedPrefetchLines(lines, dimension*4)
	var touched uint32
	for _, position := range neighbors[:count] {
		start := int(position) * dimension
		for line := 0; line < lineCount; line++ {
			element := line * 16
			if element >= dimension {
				break
			}
			touched ^= math.Float32bits(vectors[start+element])
		}
	}
	runtime.KeepAlive(touched)
}

func prefetchDenseHNSWNeighborsFP16[T ~int | ~uint32](vectors []uint16, dimension int, neighbors []T, offset, lines uint32) {
	count := prefetchNeighborCount(len(neighbors), offset)
	if count == 0 || dimension <= 0 {
		return
	}
	lineCount := normalizedPrefetchLines(lines, dimension*2)
	var touched uint16
	for _, position := range neighbors[:count] {
		start := int(position) * dimension
		for line := 0; line < lineCount; line++ {
			element := line * 32
			if element >= dimension {
				break
			}
			touched ^= vectors[start+element]
		}
	}
	runtime.KeepAlive(touched)
}

func prefetchSparseHNSWNeighbors(offsets []int, indices []uint32, values []float32, neighbors []int, offset, lines uint32) {
	count := prefetchNeighborCount(len(neighbors), offset)
	if count == 0 {
		return
	}
	var touched uint32
	for _, position := range neighbors[:count] {
		start, end := offsets[position], offsets[position+1]
		lineCount := normalizedSparsePrefetchLines(lines, end-start)
		for line := 0; line < lineCount; line++ {
			element := start + line*8
			if element >= end {
				break
			}
			touched ^= indices[element] ^ math.Float32bits(values[element])
		}
	}
	runtime.KeepAlive(touched)
}

func normalizedSparsePrefetchLines(lines uint32, elements int) int {
	if elements <= 0 {
		return 0
	}
	if lines == 0 {
		lines = uint32(min((elements-1)/8+1, int(MaxHNSWPrefetchLines)))
	}
	lines = min(lines, MaxHNSWPrefetchLines)
	return int(lines)
}

func prefetchQuantizedHNSWNeighbors(codes []QuantizedVector, neighbors []int, offset, lines uint32) {
	count := prefetchNeighborCount(len(neighbors), offset)
	if count == 0 {
		return
	}
	for _, position := range neighbors[:count] {
		code := codes[position].codes
		prefetchQuantizedCode(code, normalizedPrefetchLines(lines, len(code)))
	}
}

func prefetchNeighborCount(length int, offset uint32) int {
	if offset == 0 || length == 0 {
		return 0
	}
	if uint64(offset) >= uint64(length) {
		return length
	}
	return int(offset)
}

func normalizedPrefetchLines(lines uint32, bytes int) int {
	if bytes <= 0 {
		return 0
	}
	if lines == 0 {
		lines = uint32((bytes + 63) / 64)
	}
	lines = min(lines, MaxHNSWPrefetchLines)
	return int(lines)
}

// Borrowed collection rows are not one contiguous FP32 allocation. Warm the
// same bounded prefix as prefetchDenseHNSWNeighbors after gathering row views.
func prefetchDenseHNSWRows(vectors [][]float32, offset, lines uint32) {
	var touched uint32
	for _, vector := range vectors[:prefetchNeighborCount(len(vectors), offset)] {
		lineCount := normalizedPrefetchLines(lines, len(vector)*4)
		for line := 0; line < lineCount; line++ {
			element := line * 16
			if element >= len(vector) {
				break
			}
			touched ^= math.Float32bits(vector[element])
		}
	}
	runtime.KeepAlive(touched)
}

func prefetchDenseHNSWFP16Rows[T ~int | ~uint32](rows [][]uint16, neighbors []T, offset, lines uint32) {
	count := prefetchNeighborCount(len(neighbors), offset)
	if count == 0 {
		return
	}
	dimension := len(rows[int(neighbors[0])])
	lineCount := normalizedPrefetchLines(lines, dimension*2)
	var touched uint16
	for _, position := range neighbors[:count] {
		row := rows[int(position)]
		for line := 0; line < lineCount; line++ {
			element := line * 32
			if element >= len(row) {
				break
			}
			touched ^= row[element]
		}
	}
	runtime.KeepAlive(touched)
}
