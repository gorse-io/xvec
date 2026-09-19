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
	"slices"
)

// maxBlockHeapSearchCapacity bounds the linear merge cost of BlockHeap.
// Benchmarks show the dual-heap traversal is faster above this capacity.
const maxBlockHeapSearchCapacity = 256

type blockHeapCandidate struct {
	id       uint32
	tie      uint64
	distance float32
	checked  bool
}

// BlockHeap retains the closest candidates ordered by ascending distance, then
// ascending tie key and ID, while accepting candidates in blocks. Popped candidates remain
// retained and are skipped by subsequent pops, including after a later block
// rewinds the cursor.
//
// BlockHeap is not safe for concurrent mutation.
type BlockHeap struct {
	data      []blockHeapCandidate
	temporary []blockHeapCandidate
	evicted   []blockHeapCandidate
	capacity  int
	cursor    int
}

// Reset clears the heap for a new search and reserves storage for blocks up to
// blockSize. BlockSize is only an allocation hint; PushBlock accepts any block
// length.
func (h *BlockHeap) Reset(capacity, blockSize int) {
	if capacity < 0 {
		panic("core: negative block heap capacity")
	}
	if blockSize < 0 {
		panic("core: negative block heap block size")
	}
	if cap(h.data) < capacity {
		h.data = make([]blockHeapCandidate, 0, capacity)
	} else {
		h.data = h.data[:0]
	}
	if cap(h.temporary) < blockSize {
		h.temporary = make([]blockHeapCandidate, 0, blockSize)
	} else {
		h.temporary = h.temporary[:0]
	}
	if cap(h.evicted) < blockSize {
		h.evicted = make([]blockHeapCandidate, 0, blockSize)
	} else {
		h.evicted = h.evicted[:0]
	}
	h.capacity = capacity
	h.cursor = 0
}

func (h *BlockHeap) release(maxCapacity int) {
	if cap(h.data) > maxCapacity {
		h.data = nil
	} else {
		h.data = h.data[:0]
	}
	if cap(h.temporary) > maxCapacity {
		h.temporary = nil
	} else {
		h.temporary = h.temporary[:0]
	}
	if cap(h.evicted) > maxCapacity {
		h.evicted = nil
	} else {
		h.evicted = h.evicted[:0]
	}
	h.capacity = 0
	h.cursor = 0
}

// PushBlock inserts one candidate block. Distances and ids must have equal
// lengths. When the heap is full, candidates that do not rank before the
// current worst candidate are ignored.
func (h *BlockHeap) PushBlock(distances []float32, ids []uint32) {
	if len(distances) != len(ids) {
		panic("core: block heap distance and id lengths differ")
	}
	h.pushBlock(distances, ids, nil)
}

func (h *BlockHeap) pushBlockWithTies(distances []float32, ids []uint32, ties []uint64) {
	if len(distances) != len(ids) || len(distances) != len(ties) {
		panic("core: block heap distance, id, and tie lengths differ")
	}
	h.pushBlock(distances, ids, ties)
}

func (h *BlockHeap) pushBlock(distances []float32, ids []uint32, ties []uint64) {
	h.evicted = h.evicted[:0]
	if h.capacity == 0 || len(distances) == 0 {
		return
	}

	h.temporary = h.temporary[:0]
	if len(h.data) == h.capacity {
		worst := h.data[len(h.data)-1]
		for index, distance := range distances {
			candidate := newBlockHeapCandidate(ids[index], distance, ties, index)
			if compareBlockHeapCandidates(candidate, worst) < 0 {
				h.temporary = append(h.temporary, candidate)
			}
		}
	} else {
		for index, distance := range distances {
			h.temporary = append(h.temporary, newBlockHeapCandidate(ids[index], distance, ties, index))
		}
	}
	if len(h.temporary) == 0 {
		return
	}

	slices.SortFunc(h.temporary, compareBlockHeapCandidates)
	if len(h.temporary) > h.capacity {
		h.temporary = h.temporary[:h.capacity]
	}
	h.mergeTemporary()
	h.rewind()
	h.temporary = h.temporary[:0]
}

func newBlockHeapCandidate(id uint32, distance float32, ties []uint64, index int) blockHeapCandidate {
	tie := uint64(id)
	if ties != nil {
		tie = ties[index]
	}
	return blockHeapCandidate{id: id, tie: tie, distance: distance}
}

// appendBlockHeapBoundaryTies keeps equal-distance candidates available for
// graph expansion when the result tie-break excludes them from a full heap.
func appendBlockHeapBoundaryTies(h *BlockHeap, distances []float32, ids []uint32, ties []uint64, overflow []uint32) []uint32 {
	if h.Len() == 0 || h.Len() != h.Cap() {
		return overflow
	}
	worst := h.data[h.Len()-1]
	for _, candidate := range h.evicted {
		if candidate.distance == worst.distance {
			overflow = append(overflow, candidate.id)
		}
	}
	for index, distance := range distances {
		candidate := newBlockHeapCandidate(ids[index], distance, ties, index)
		if distance == worst.distance && compareBlockHeapCandidates(candidate, worst) > 0 {
			overflow = append(overflow, ids[index])
		}
	}
	return overflow
}

// HasNext reports whether an unpopped retained candidate remains.
func (h *BlockHeap) HasNext() bool {
	return h.cursor < len(h.data)
}

// Pop removes the closest unpopped candidate from the expansion sequence.
// The candidate remains part of the retained sorted top-k set.
func (h *BlockHeap) Pop() (uint32, bool) {
	if !h.HasNext() {
		return 0, false
	}
	id := h.data[h.cursor].id
	h.data[h.cursor].checked = true
	h.cursor++
	for h.cursor < len(h.data) && h.data[h.cursor].checked {
		h.cursor++
	}
	return id, true
}

// PopWithNext pops one candidate and returns the next unpopped candidate id.
// next is math.MaxUint32 when no candidate remains.
func (h *BlockHeap) PopWithNext() (id, next uint32, ok bool) {
	id, ok = h.Pop()
	if !ok {
		return 0, math.MaxUint32, false
	}
	if h.HasNext() {
		return id, h.data[h.cursor].id, true
	}
	return id, math.MaxUint32, true
}

// Len returns the retained candidate count.
func (h *BlockHeap) Len() int {
	return len(h.data)
}

// Cap returns the maximum retained candidate count configured by Reset.
func (h *BlockHeap) Cap() int {
	return h.capacity
}

// Sorted returns copies of at most length retained ids and distances in
// ascending distance order. Popped candidates are included.
func (h *BlockHeap) Sorted(length int) ([]uint32, []float32) {
	if length < 0 {
		panic("core: negative block heap sorted length")
	}
	length = min(length, len(h.data))
	ids := make([]uint32, length)
	distances := make([]float32, length)
	for index := range length {
		ids[index] = h.data[index].id
		distances[index] = h.data[index].distance
	}
	return ids, distances
}

// ID returns the retained candidate id at position index in distance order.
func (h *BlockHeap) ID(index int) uint32 {
	return h.data[index].id
}

// Distance returns the retained candidate distance at position index.
func (h *BlockHeap) Distance(index int) float32 {
	return h.data[index].distance
}

func (h *BlockHeap) mergeTemporary() {
	oldLength := len(h.data)
	temporaryLength := len(h.temporary)
	write := oldLength + temporaryLength - 1
	newLength := min(oldLength+temporaryLength, h.capacity)
	h.data = h.data[:newLength]
	i, j := oldLength-1, temporaryLength-1

	for write >= h.capacity {
		if compareBlockHeapCandidates(h.data[i], h.temporary[j]) > 0 {
			if !h.data[i].checked {
				h.evicted = append(h.evicted, h.data[i])
			}
			i--
		} else {
			j--
		}
		write--
	}
	for i >= 0 && j >= 0 {
		if compareBlockHeapCandidates(h.data[i], h.temporary[j]) > 0 {
			h.data[write] = h.data[i]
			i--
		} else {
			h.data[write] = h.temporary[j]
			j--
		}
		write--
	}
	for j >= 0 {
		h.data[write] = h.temporary[j]
		j--
		write--
	}
}

func (h *BlockHeap) rewind() {
	h.cursor = 0
	for h.cursor < len(h.data) && h.data[h.cursor].checked {
		h.cursor++
	}
}

func compareBlockHeapCandidates(left, right blockHeapCandidate) int {
	switch {
	case left.distance < right.distance:
		return -1
	case left.distance > right.distance:
		return 1
	case left.tie < right.tie:
		return -1
	case left.tie > right.tie:
		return 1
	case left.id < right.id:
		return -1
	case left.id > right.id:
		return 1
	default:
		return 0
	}
}
