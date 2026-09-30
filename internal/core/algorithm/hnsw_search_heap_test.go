// SPDX-License-Identifier: Apache-2.0

package core

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/stretchr/testify/require"
)

func TestHNSWSearchHeapMatchesGeneric(t *testing.T) {
	keys := make([]uint64, 1000)
	for j := range keys {
		keys[j] = uint64((j*137)%len(keys) + 5000)
	}
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		for _, mode := range []struct{ worstFirst, keyTies bool }{{false, false}, {true, true}, {true, false}} {
			t.Run(fmt.Sprintf("metric=%v/worstFirst=%v/keyTies=%v", metric, mode.worstFirst, mode.keyTies), func(t *testing.T) {
				var heap hnswSearchHeap
				var tieKeys []uint64
				if mode.keyTies {
					tieKeys = keys
				}
				heap.reset(3, metric, tieKeys, mode.worstFirst)
				// Use the old generic heap and its original comparators as an
				// independent oracle, including keys that differ from positions.
				index := &ScalarQuantizedHNSWIndex{vectors: &scalarQuantizedVectors{metric: metric, keys: keys}}
				reference := container.NewHeap(func(left, right hnswScoredNode) bool {
					if mode.keyTies {
						return index.resultNodeBetter(right, left)
					}
					if mode.worstFirst {
						return hnswNodeBetter(metric, right, left)
					}
					return hnswNodeBetter(metric, left, right)
				})
				random := rand.New(rand.NewPCG(17, 31))
				for position := range keys {
					// Many ties exercise both frontier and result tie ordering.
					node := hnswScoredNode{position: position, score: float32(random.IntN(11)-5) / 8}
					switch random.IntN(4) {
					case 0:
						got, ok := heap.Pop()
						want, wantOK := reference.Pop()
						require.Equal(t, wantOK, ok)
						require.Equal(t, want, got)
					case 1:
						if reference.Len() != 0 {
							heap.Replace(node)
							reference.Replace(node)
							break
						}
						fallthrough
					default:
						heap.Push(node)
						reference.Push(node)
					}
					require.Equal(t, reference.Len(), heap.Len())
					got, ok := heap.Peek()
					want, wantOK := reference.Peek()
					require.Equal(t, wantOK, ok)
					require.Equal(t, want, got)
				}
				if mode.worstFirst {
					want := make([]hnswScoredNode, reference.Len())
					for j := len(want) - 1; j >= 0; j-- {
						want[j], _ = reference.Pop()
					}
					got := heap.results()
					require.Equal(t, want, got)
					heap.reset(10, metric, keys, true)
					for j := range 10 {
						heap.Push(hnswScoredNode{position: j, score: 100})
					}
					// A subsequent query must not overwrite the returned result.
					require.Equal(t, want, got)
				} else {
					for reference.Len() != 0 {
						got, _ := heap.Pop()
						want, _ := reference.Pop()
						require.Equal(t, want, got)
					}
				}
			})
		}
	}
}

func TestHNSWSearchHeapScratchRelease(t *testing.T) {
	visited := &hnswVisited{}
	keys := []uint64{100, 50}
	visited.frontierHeap.reset(2, MetricL2, nil, false)
	visited.acceptedHeap.reset(2, MetricL2, keys, true)
	visited.acceptedHeap.Push(hnswScoredNode{position: 1})
	storage := slices.Clone(visited.acceptedHeap.nodes)
	visited.resetBatch()
	require.Nil(t, visited.acceptedHeap.keys)
	require.Empty(t, visited.acceptedHeap.nodes)
	require.GreaterOrEqual(t, cap(visited.acceptedHeap.nodes), 2)
	visited.acceptedHeap.reset(2, MetricIP, keys, true)
	visited.acceptedHeap.Push(storage[0])
	require.Equal(t, storage, visited.acceptedHeap.results())

	visited.frontierHeap.reset(maxPooledDistanceBatchCapacity+1, MetricL2, nil, false)
	visited.acceptedHeap.reset(maxPooledDistanceBatchCapacity+1, MetricL2, keys, true)
	visited.resetBatch()
	require.Nil(t, visited.frontierHeap.nodes)
	require.Nil(t, visited.acceptedHeap.nodes)
	require.Nil(t, visited.acceptedHeap.keys)
}
