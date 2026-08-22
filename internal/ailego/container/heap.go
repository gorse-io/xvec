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

package container

// Heap is a binary heap ordered by less. If less(a, b) is true, a is nearer
// the root than b. Heap is not safe for concurrent mutation.
type Heap[T any] struct {
	values []T
	less   func(a, b T) bool
}

// NewHeap constructs an empty heap. It panics when less is nil.
func NewHeap[T any](less func(a, b T) bool) *Heap[T] {
	return NewHeapWithCapacity(0, less)
}

// NewHeapWithCapacity constructs an empty heap with storage reserved for at
// least capacity values. It panics when capacity is negative or less is nil.
func NewHeapWithCapacity[T any](capacity int, less func(a, b T) bool) *Heap[T] {
	if capacity < 0 {
		panic("ailego: negative heap capacity")
	}
	if less == nil {
		panic("ailego: nil heap comparator")
	}
	return &Heap[T]{values: make([]T, 0, capacity), less: less}
}

// Len returns the number of values in h.
func (h *Heap[T]) Len() int { return len(h.values) }

// Push inserts value into h.
func (h *Heap[T]) Push(value T) {
	h.values = append(h.values, value)
	h.siftUp(len(h.values) - 1)
}

// Peek returns the root without removing it.
func (h *Heap[T]) Peek() (T, bool) {
	if len(h.values) == 0 {
		var zero T
		return zero, false
	}
	return h.values[0], true
}

// Pop removes and returns the root.
func (h *Heap[T]) Pop() (T, bool) {
	if len(h.values) == 0 {
		var zero T
		return zero, false
	}
	root := h.values[0]
	last := len(h.values) - 1
	h.values[0] = h.values[last]
	var zero T
	h.values[last] = zero
	h.values = h.values[:last]
	if len(h.values) > 0 {
		h.siftDown(0)
	}
	return root, true
}

// Replace replaces and returns the root. If h is empty, it inserts value and
// reports false because no prior root existed.
func (h *Heap[T]) Replace(value T) (T, bool) {
	if len(h.values) == 0 {
		h.values = append(h.values, value)
		var zero T
		return zero, false
	}
	root := h.values[0]
	h.values[0] = value
	h.siftDown(0)
	return root, true
}

// Values returns a copy of the heap's internal level order.
func (h *Heap[T]) Values() []T { return append([]T(nil), h.values...) }

func (h *Heap[T]) siftUp(index int) {
	for index > 0 {
		parent := (index - 1) / 2
		if !h.less(h.values[index], h.values[parent]) {
			return
		}
		h.values[index], h.values[parent] = h.values[parent], h.values[index]
		index = parent
	}
}

func (h *Heap[T]) siftDown(index int) {
	for {
		left := index*2 + 1
		if left >= len(h.values) {
			return
		}
		best := left
		right := left + 1
		if right < len(h.values) && h.less(h.values[right], h.values[left]) {
			best = right
		}
		if !h.less(h.values[best], h.values[index]) {
			return
		}
		h.values[index], h.values[best] = h.values[best], h.values[index]
		index = best
	}
}
