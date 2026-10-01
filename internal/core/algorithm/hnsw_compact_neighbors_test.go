// SPDX-License-Identifier: Apache-2.0

package core

import (
	"context"
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompactHNSWGroupExpansionAndStreamingClone(t *testing.T) {
	ctx := context.Background()
	graph := denseGroupHNSWFixture()
	original, err := NewScalarQuantizedHNSWIndex(ctx, graph, QuantizationFP16, nil)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "index.hnsw")
	require.NoError(t, original.Save(ctx, path))
	originals := make(map[uint64][]byte)
	for position, key := range graph.keys {
		for _, value := range graph.vectorAt(position) {
			originals[key] = binary.LittleEndian.AppendUint32(originals[key], math.Float32bits(value))
		}
	}
	compact, err := OpenScalarQuantizedHNSWIndexWithEncodedVectors(ctx, path, QuantizationFP16, nil, originals, true)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, compact.Close()) })
	options := HNSWGroupSearchOptions{GroupByOptions: GroupByOptions{
		GroupCount: 2, TopKPerGroup: 1,
		Resolve: func(key uint64) (string, bool) {
			if key < 20 {
				return "near", true
			}
			return "far", true
		},
	}, EF: 1, PrefetchOffset: 1, PrefetchLines: 2}
	want, err := original.SearchHNSWGroups(ctx, []float32{0}, options)
	require.NoError(t, err)
	got, err := compact.SearchHNSWGroups(ctx, []float32{0}, options)
	require.NoError(t, err)
	require.Len(t, got, 2)
	require.Equal(t, want, got)
	// The compact graph may be cloned for mutable dense streaming. Mutation
	// expands the topology and must not retain or overwrite the original arena.
	before, err := encodeHNSWIndex(ctx, compact.base)
	require.NoError(t, err)
	clone, err := cloneHNSWIndex(ctx, compact.base)
	require.NoError(t, err)
	require.NoError(t, clone.Add(ctx, 99, []float32{5}))
	require.Nil(t, clone.compactNeighbors)
	require.Equal(t, len(graph.keys)+1, clone.Len())
	after, err := encodeHNSWIndex(ctx, compact.base)
	require.NoError(t, err)
	require.Equal(t, before, after)
	assertHNSWGraphInvariants(t, clone)
	// Validate offset tables before any neighbor view indexes their storage.
	compact.base.compactNeighbors.offsets[1] = -1
	require.ErrorIs(t, validateHNSWIndex(ctx, compact.base), ErrInvalidHNSWFile)
}
