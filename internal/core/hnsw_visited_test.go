// SPDX-License-Identifier: Apache-2.0

package core

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHNSWVisitedReset(t *testing.T) {
	var visited hnswVisited
	visited.reset(4)
	visited.mark(1)
	require.True(t, visited.seen(1))
	require.False(t, visited.seen(2))

	visited.reset(4)
	require.False(t, visited.seen(1))
	visited.mark(2)
	require.True(t, visited.seen(2))
}

func TestHNSWVisitedResize(t *testing.T) {
	var visited hnswVisited
	visited.reset(2)
	visited.mark(1)
	visited.reset(5)

	require.Len(t, visited.marks, 5)
	for position := range visited.marks {
		require.False(t, visited.seen(position))
	}
	visited.mark(4)
	require.True(t, visited.seen(4))
}

func TestHNSWVisitedGenerationWrap(t *testing.T) {
	var visited hnswVisited
	visited.reset(3)
	visited.mark(1)

	for range 255 {
		visited.reset(3)
	}

	require.NotZero(t, visited.generation)
	for position := range visited.marks {
		require.False(t, visited.seen(position))
	}
}

func TestHNSWVisitedGenerationWrapAfterShrink(t *testing.T) {
	var visited hnswVisited
	visited.reset(4)
	visited.mark(3)
	visited.marks[3] = 2 // Simulate a stale mark matching the first generation after wrap.

	visited.marks = visited.marks[:2]
	for range 255 {
		visited.reset(2)
	}
	visited.reset(4)

	require.False(t, visited.seen(3))
}

func TestHNSWVisitedPoolReusesAllocation(t *testing.T) {
	visited := acquireHNSWVisited(1024)
	releaseHNSWVisited(visited)
	allocations := testing.AllocsPerRun(100, func() {
		visited := acquireHNSWVisited(1024)
		visited.mark(100)
		if !visited.seen(100) {
			panic("marked position is not visited")
		}
		releaseHNSWVisited(visited)
	})
	require.Zero(t, allocations)
}
