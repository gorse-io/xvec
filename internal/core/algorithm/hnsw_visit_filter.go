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
	v.batchIDs = v.batchIDs[:0]
	v.batchTies = v.batchTies[:0]
	v.overflow = v.overflow[:0]
	v.batchVectors = v.batchVectors[:0]
	v.batchMagnitudes = v.batchMagnitudes[:0]
	v.batchScores = v.batchScores[:0]
	v.blockHeap.release(maxPooledDistanceBatchCapacity)
	if cap(v.batchPositions) > maxPooledDistanceBatchCapacity || cap(v.batchIDs) > maxPooledDistanceBatchCapacity || cap(v.batchTies) > maxPooledDistanceBatchCapacity || cap(v.overflow) > maxPooledDistanceBatchCapacity || cap(v.batchVectors) > maxPooledDistanceBatchCapacity || cap(v.batchMagnitudes) > maxPooledDistanceBatchCapacity || cap(v.batchScores) > maxPooledDistanceBatchCapacity {
		v.batchPositions = nil
		v.batchIDs = nil
		v.batchTies = nil
		v.overflow = nil
		v.batchVectors = nil
		v.batchMagnitudes = nil
		v.batchScores = nil
	}
}

// hnswVisited tracks graph visits without clearing the full node-sized buffer
// between traversals. A generation value distinguishes marks from consecutive
// traversals; the buffer is cleared only when the byte generation wraps.
type hnswVisited struct {
	marks              []uint8
	generation         uint8
	expandedMarks      []uint8
	expandedGeneration uint8

	batchPositions  []int
	batchIDs        []uint32
	batchTies       []uint64
	overflow        []uint32
	batchVectors    [][]float32
	batchMagnitudes []float32
	batchScores     []float32
	blockHeap       BlockHeap
}

func (v *hnswVisited) reset(size int) {
	if cap(v.marks) < size {
		v.marks = make([]uint8, size)
		v.generation = 1
	} else {
		v.marks = v.marks[:size]
		v.generation++
		if v.generation == 0 {
			v.marks = v.marks[:cap(v.marks)]
			clear(v.marks)
			v.marks = v.marks[:size]
			v.generation = 1
		}
	}
	if cap(v.expandedMarks) < size {
		v.expandedMarks = make([]uint8, size)
		v.expandedGeneration = 1
	} else {
		v.expandedMarks = v.expandedMarks[:size]
		v.expandedGeneration++
		if v.expandedGeneration == 0 {
			v.expandedMarks = v.expandedMarks[:cap(v.expandedMarks)]
			clear(v.expandedMarks)
			v.expandedMarks = v.expandedMarks[:size]
			v.expandedGeneration = 1
		}
	}
}

func (v *hnswVisited) seen(position int) bool {
	return v.marks[position] == v.generation
}

func (v *hnswVisited) mark(position int) {
	v.marks[position] = v.generation
}

func (v *hnswVisited) expanded(position int) bool {
	return v.expandedMarks[position] == v.expandedGeneration
}

func (v *hnswVisited) markExpanded(position int) {
	v.expandedMarks[position] = v.expandedGeneration
}
