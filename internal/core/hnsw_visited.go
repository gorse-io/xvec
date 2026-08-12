// SPDX-License-Identifier: Apache-2.0

package core

import "sync"

var hnswVisitedPool = sync.Pool{
	New: func() any { return new(hnswVisited) },
}

func acquireHNSWVisited(size int) *hnswVisited {
	visited := hnswVisitedPool.Get().(*hnswVisited)
	visited.reset(size)
	return visited
}

func releaseHNSWVisited(visited *hnswVisited) {
	hnswVisitedPool.Put(visited)
}

// hnswVisited tracks graph visits without clearing the full node-sized buffer
// between traversals. A generation value distinguishes marks from consecutive
// traversals; the buffer is cleared only when the byte generation wraps.
type hnswVisited struct {
	marks      []uint8
	generation uint8
}

func (v *hnswVisited) reset(size int) {
	if cap(v.marks) < size {
		v.marks = make([]uint8, size)
		v.generation = 1
		return
	}
	v.marks = v.marks[:size]
	v.generation++
	if v.generation == 0 {
		v.marks = v.marks[:cap(v.marks)]
		clear(v.marks)
		v.marks = v.marks[:size]
		v.generation = 1
	}
}

func (v *hnswVisited) seen(position int) bool {
	return v.marks[position] == v.generation
}

func (v *hnswVisited) mark(position int) {
	v.marks[position] = v.generation
}
