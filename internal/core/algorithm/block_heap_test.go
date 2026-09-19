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
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBlockHeapPushBlockSortsAndTruncates(t *testing.T) {
	var heap BlockHeap
	heap.Reset(4, 3)
	heap.PushBlock([]float32{30, 10, 40}, []uint32{30, 10, 40})
	heap.PushBlock([]float32{50, 20, 5}, []uint32{50, 20, 5})

	require.Equal(t, 4, heap.Len())
	require.Equal(t, 4, heap.Cap())
	ids, distances := heap.Sorted(10)
	require.Equal(t, []uint32{5, 10, 20, 30}, ids)
	require.Equal(t, []float32{5, 10, 20, 30}, distances)
	for index := range ids {
		require.Equal(t, ids[index], heap.ID(index))
		require.Equal(t, distances[index], heap.Distance(index))
	}
}

func TestBlockHeapPopExposesNextUnexpandedCandidate(t *testing.T) {
	var heap BlockHeap
	heap.Reset(4, 4)
	heap.PushBlock([]float32{30, 10, 40, 20}, []uint32{30, 10, 40, 20})

	for _, expected := range []struct {
		id   uint32
		next uint32
	}{
		{id: 10, next: 20},
		{id: 20, next: 30},
		{id: 30, next: 40},
		{id: 40, next: math.MaxUint32},
	} {
		id, next, ok := heap.PopWithNext()
		require.True(t, ok)
		require.Equal(t, expected.id, id)
		require.Equal(t, expected.next, next)
	}
	require.False(t, heap.HasNext())
	_, _, ok := heap.PopWithNext()
	require.False(t, ok)
}

func TestBlockHeapPushBlockRewindsAndSkipsPoppedCandidates(t *testing.T) {
	var heap BlockHeap
	heap.Reset(5, 4)
	heap.PushBlock([]float32{1, 2, 3, 4}, []uint32{1, 2, 3, 4})

	id, ok := heap.Pop()
	require.True(t, ok)
	require.Equal(t, uint32(1), id)
	id, _, ok = heap.PopWithNext()
	require.True(t, ok)
	require.Equal(t, uint32(2), id)

	heap.PushBlock([]float32{0.5, 2.5}, []uint32{0, 5})
	require.Equal(t, 5, heap.Len())

	id, next, ok := heap.PopWithNext()
	require.True(t, ok)
	require.Equal(t, uint32(0), id)
	require.Equal(t, uint32(5), next)
	id, ok = heap.Pop()
	require.True(t, ok)
	require.Equal(t, uint32(5), id)
	id, next, ok = heap.PopWithNext()
	require.True(t, ok)
	require.Equal(t, uint32(3), id)
	require.Equal(t, uint32(math.MaxUint32), next)
	require.False(t, heap.HasNext())
}

func TestBlockHeapMatchesFullSortAcrossResetAndVariableBlocks(t *testing.T) {
	random := rand.New(rand.NewPCG(7, 1))
	for _, capacity := range []int{0, 1, 2, 7, 8, 16, 33} {
		var heap BlockHeap
		heap.Reset(capacity, 2)
		var reference []blockHeapTestCandidate
		var id uint32
		for round := range 64 {
			count := round % 20
			distances := make([]float32, count)
			ids := make([]uint32, count)
			for index := range count {
				ids[index] = id
				distances[index] = float32(random.Uint32()%1024)*10_000 + float32(id)
				reference = append(reference, blockHeapTestCandidate{id: id, distance: distances[index]})
				id++
			}
			slices.SortFunc(reference, func(left, right blockHeapTestCandidate) int {
				switch {
				case left.distance < right.distance:
					return -1
				case left.distance > right.distance:
					return 1
				case left.id < right.id:
					return -1
				case left.id > right.id:
					return 1
				default:
					return 0
				}
			})
			if len(reference) > capacity {
				reference = reference[:capacity]
			}
			heap.PushBlock(distances, ids)
			gotIDs, gotDistances := heap.Sorted(heap.Len())
			require.Len(t, gotIDs, len(reference))
			for index := range reference {
				require.Equal(t, reference[index].id, gotIDs[index])
				require.Equal(t, reference[index].distance, gotDistances[index])
			}
		}

		heap.Reset(1, 1)
		heap.PushBlock([]float32{1}, []uint32{42})
		got, ok := heap.Pop()
		require.True(t, ok)
		require.Equal(t, uint32(42), got)
		require.False(t, heap.HasNext())
	}
}

func TestBlockHeapBreaksDistanceTiesByID(t *testing.T) {
	blocks := []struct {
		distances []float32
		ids       []uint32
	}{
		{distances: []float32{1, 1}, ids: []uint32{9, 7}},
		{distances: []float32{1, 1}, ids: []uint32{8, 6}},
	}
	for _, order := range [][]int{{0, 1}, {1, 0}} {
		var heap BlockHeap
		heap.Reset(2, 2)
		for _, block := range order {
			heap.PushBlock(blocks[block].distances, blocks[block].ids)
		}
		ids, distances := heap.Sorted(heap.Len())
		require.Equal(t, []uint32{6, 7}, ids)
		require.Equal(t, []float32{1, 1}, distances)
	}
}

func TestBlockHeapValidatesArguments(t *testing.T) {
	var heap BlockHeap
	require.Panics(t, func() { heap.Reset(-1, 1) })
	require.Panics(t, func() { heap.Reset(1, -1) })

	heap.Reset(2, 2)
	require.Panics(t, func() { heap.PushBlock([]float32{1}, nil) })
	require.Panics(t, func() { heap.Sorted(-1) })
	require.Panics(t, func() { heap.ID(-1) })
	require.Panics(t, func() { heap.Distance(heap.Len()) })
}

func TestBlockHeapReusesPushStorage(t *testing.T) {
	var heap BlockHeap
	heap.Reset(8, 8)
	distances := []float32{8, 7, 6, 5, 4, 3, 2, 1}
	ids := []uint32{8, 7, 6, 5, 4, 3, 2, 1}
	heap.PushBlock(distances, ids)

	allocations := testing.AllocsPerRun(100, func() {
		heap.Reset(8, 8)
		heap.PushBlock(distances, ids)
	})
	require.Zero(t, allocations)
}

type blockHeapTestCandidate struct {
	id       uint32
	distance float32
}
