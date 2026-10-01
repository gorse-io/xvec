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
	"context"
	"encoding/binary"
	"fmt"
)

// hnswCompactNeighbors stores the immutable graph in three contiguous arrays.
// Mutable construction/streaming retains its existing wide adjacency lists.
// Nodes delimit level lists; offsets delimit uint32 neighbors, in original order.
type hnswCompactNeighbors struct {
	nodes   []int
	offsets []int
	ids     []uint32
}

// hnswNeighborList is a zero-allocation view of either graph representation.
type hnswNeighborList struct {
	positions []int
	ids       []uint32
}

func (n hnswNeighborList) Len() int { return len(n.positions) + len(n.ids) }
func (n hnswNeighborList) At(j int) int {
	if n.ids != nil {
		return int(n.ids[j])
	}
	return n.positions[j]
}
func (i *HNSWIndex) neighborList(position, level int) hnswNeighborList {
	if i.compactNeighbors != nil {
		graph := i.compactNeighbors
		slot := graph.nodes[position] + level
		start, end := graph.offsets[slot], graph.offsets[slot+1]
		return hnswNeighborList{ids: graph.ids[start:end:end]}
	}
	return hnswNeighborList{positions: i.neighbors[position][level]}
}
func (i *HNSWIndex) neighborLevelCount(position int) int {
	if i.compactNeighbors != nil {
		return i.compactNeighbors.nodes[position+1] - i.compactNeighbors.nodes[position]
	}
	return len(i.neighbors[position])
}

// Size the arena before decoding, avoiding geometric growth and intermediate
// wide lists. This structural pass does not touch vector bytes; full validation
// remains in the decoder and validateHNSWIndex.
func newHNSWCompactNeighbors(ctx context.Context, payload []byte, count, vectorBytes, m int) (*hnswCompactNeighbors, error) {
	levels, edges, offset := 0, 0, 0
	for position := 0; position < count; position++ {
		if position&255 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if !hnswPayloadAvailable(payload, offset, hnswRecordFixedBytes+vectorBytes+hnswLevelFixedBytes) {
			return nil, fmt.Errorf("%w: truncated node %d", ErrInvalidHNSWFile, position)
		}
		level := binary.LittleEndian.Uint32(payload[offset+8 : offset+12])
		if level > MaxHNSWLevel {
			return nil, fmt.Errorf("%w: invalid node level", ErrInvalidHNSWFile)
		}
		offset += hnswRecordFixedBytes + vectorBytes
		for l := 0; l <= int(level); l++ {
			if !hnswPayloadAvailable(payload, offset, 4) {
				return nil, fmt.Errorf("%w: truncated neighbors", ErrInvalidHNSWFile)
			}
			degree := uint64(binary.LittleEndian.Uint32(payload[offset : offset+4]))
			offset += 4
			limit := m
			if l == 0 {
				limit *= 2
			}
			if degree > uint64(limit) || degree > uint64(count) || degree > uint64((len(payload)-offset)/4) {
				return nil, fmt.Errorf("%w: invalid neighbor degree", ErrInvalidHNSWFile)
			}
			offset += int(degree) * 4
			edges += int(degree)
			levels++
		}
	}
	if offset != len(payload) {
		return nil, fmt.Errorf("%w: trailing payload data", ErrInvalidHNSWFile)
	}
	return &hnswCompactNeighbors{nodes: make([]int, count+1), offsets: make([]int, 0, levels+1), ids: make([]uint32, 0, edges)}, nil
}

func (i *HNSWIndex) prefetchNeighborList(neighbors hnswNeighborList, offset, lines uint32) {
	if neighbors.ids == nil {
		i.prefetchNeighbors(neighbors.positions, offset, lines)
		return
	}
	if i.fp16 {
		prefetchDenseHNSWNeighborsFP16(i.vectorsFP16, i.dimension, neighbors.ids, offset, lines)
		return
	}
	prefetchDenseHNSWNeighbors(i.vectors, i.dimension, neighbors.ids, offset, lines)
}
func prefetchQuantizedHNSWNeighborList(codes []QuantizedVector, neighbors hnswNeighborList, offset, lines uint32) {
	for j := 0; j < prefetchNeighborCount(neighbors.Len(), offset); j++ {
		code := codes[neighbors.At(j)].codes
		prefetchQuantizedCode(code, normalizedPrefetchLines(lines, len(code)))
	}
}

func (i *HNSWIndex) validNeighborStorage(count int) bool {
	if i.compactNeighbors == nil {
		return len(i.neighbors) == count
	}
	graph := i.compactNeighbors
	if len(i.neighbors) != 0 || len(graph.nodes) != count+1 || len(graph.offsets) == 0 || graph.nodes[0] != 0 || graph.nodes[count] != len(graph.offsets)-1 || graph.offsets[0] != 0 || graph.offsets[len(graph.offsets)-1] != len(graph.ids) {
		return false
	}
	for position := 0; position < count; position++ {
		if graph.nodes[position] < 0 || graph.nodes[position+1] <= graph.nodes[position] || graph.nodes[position+1] > len(graph.offsets)-1 {
			return false
		}
	}
	for slot := 1; slot < len(graph.offsets); slot++ {
		if graph.offsets[slot] < graph.offsets[slot-1] || graph.offsets[slot] > len(graph.ids) {
			return false
		}
	}
	return true
}
