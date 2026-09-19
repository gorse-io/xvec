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
	visited.resetBatch()
	hnswVisitedPool.Put(visited)
}

func (v *hnswVisited) resetBatch() {
	clear(v.batchVectors[:cap(v.batchVectors)])
	v.batchPositions = v.batchPositions[:0]
	v.batchVectors = v.batchVectors[:0]
	v.batchMagnitudes = v.batchMagnitudes[:0]
	v.batchScores = v.batchScores[:0]
	if cap(v.batchPositions) > maxPooledDistanceBatchCapacity || cap(v.batchVectors) > maxPooledDistanceBatchCapacity || cap(v.batchMagnitudes) > maxPooledDistanceBatchCapacity || cap(v.batchScores) > maxPooledDistanceBatchCapacity {
		v.batchPositions = nil
		v.batchVectors = nil
		v.batchMagnitudes = nil
		v.batchScores = nil
	}
}

// hnswVisited tracks graph visits without clearing the full node-sized buffer
// between traversals. A generation value distinguishes marks from consecutive
// traversals; the buffer is cleared only when the byte generation wraps.
type hnswVisited struct {
	marks      []uint8
	generation uint8

	batchPositions  []int
	batchVectors    [][]float32
	batchMagnitudes []float32
	batchScores     []float32
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
