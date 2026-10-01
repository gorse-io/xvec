// SPDX-License-Identifier: Apache-2.0

package core

import "slices"

// hnswSearchHeap is the binary heap used by scalar-code dual-heap traversal.
// The frontier orders best scores first, breaking ties by graph position;
// accepted results order worst scores first, breaking ties by document key.
// It stores no comparator closure and reuses the query's visited scratch.
type hnswSearchHeap struct {
	nodes      []hnswScoredNode
	metric     Metric
	keys       []uint64
	worstFirst bool
}

func (h *hnswSearchHeap) reset(capacity int, metric Metric, keys []uint64, worstFirst bool) {
	h.nodes = slices.Grow(h.nodes[:0], capacity)
	h.metric, h.keys, h.worstFirst = metric, keys, worstFirst
}

func (h *hnswSearchHeap) release() {
	h.nodes = h.nodes[:0]
	h.keys = nil // Do not retain an index through pooled query scratch.
	if cap(h.nodes) > maxPooledDistanceBatchCapacity {
		h.nodes = nil
	}
}

func (h *hnswSearchHeap) less(left, right hnswScoredNode) bool {
	if h.worstFirst {
		left, right = right, left
	}
	if left.score == right.score {
		if h.keys != nil {
			return h.keys[left.position] < h.keys[right.position]
		}
		return left.position < right.position
	}
	return h.metric.Better(left.score, right.score)
}

func (h *hnswSearchHeap) Len() int { return len(h.nodes) }

func (h *hnswSearchHeap) Peek() (hnswScoredNode, bool) {
	if len(h.nodes) == 0 {
		return hnswScoredNode{}, false
	}
	return h.nodes[0], true
}

func (h *hnswSearchHeap) Push(node hnswScoredNode) {
	index := len(h.nodes)
	h.nodes = append(h.nodes, node)
	for index > 0 {
		parent := (index - 1) / 2
		if !h.less(node, h.nodes[parent]) {
			break
		}
		h.nodes[index] = h.nodes[parent]
		index = parent
	}
	h.nodes[index] = node
}

func (h *hnswSearchHeap) Pop() (hnswScoredNode, bool) {
	if len(h.nodes) == 0 {
		return hnswScoredNode{}, false
	}
	root := h.nodes[0]
	last := h.nodes[len(h.nodes)-1]
	h.nodes = h.nodes[:len(h.nodes)-1]
	if len(h.nodes) > 0 {
		h.siftDown(last)
	}
	return root, true
}

func (h *hnswSearchHeap) Replace(node hnswScoredNode) {
	h.siftDown(node)
}

func (h *hnswSearchHeap) siftDown(node hnswScoredNode) {
	index := 0
	for index < len(h.nodes)/2 {
		child := index*2 + 1
		if child+1 < len(h.nodes) && h.less(h.nodes[child+1], h.nodes[child]) {
			child++
		}
		if !h.less(h.nodes[child], node) {
			break
		}
		h.nodes[index] = h.nodes[child]
		index = child
	}
	h.nodes[index] = node
}

// results drains the worst-first heap into an independently owned, best-first
// result slice. No copied heap followed by a separate sort is needed.
func (h *hnswSearchHeap) results() []hnswScoredNode {
	result := make([]hnswScoredNode, len(h.nodes))
	for j := len(result) - 1; j >= 0; j-- {
		result[j], _ = h.Pop()
	}
	return result
}
