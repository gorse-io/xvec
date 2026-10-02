// SPDX-License-Identifier: Apache-2.0
package core

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestChunkedFP16FlatStorage(t *testing.T) {
	ctx := context.Background()
	index, err := NewChunkedDenseFlatIndexFP16(3, MetricL2)
	require.NoError(t, err)
	var first *uint16
	for n := 0; n < denseFP16ChunkRows*3+1; n++ {
		row := []uint16{0x3c00, 0, uint16(n)}
		require.NoError(t, index.AddFP16(ctx, uint64(n), row))
		if n == 0 {
			first = &index.fp16Row(0)[0]
		}
		row[0] = 0
	}
	require.NoError(t, index.Reserve(10000))
	require.Same(t, first, &index.fp16Row(0)[0])
	require.Empty(t, index.vectorsFP16)
	require.Len(t, index.fp16Chunks, 4)
	require.Equal(t, uint16(0x3c00), index.fp16Row(0)[0])
	got, err := index.Search(ctx, []float32{1, 0, 0}, 1)
	require.NoError(t, err)
	require.Equal(t, uint64(0), got[0].Key)
	require.NoError(t, index.Add(ctx, 10000, []float32{1, 0, 0}))
	require.Equal(t, uint16(0x3c00), index.fp16Row(index.Len() - 1)[0])
}

func TestBorrowedFP16FlatRows(t *testing.T) {
	ctx := context.Background()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		rows := [][]uint16{{0x3c00, 0}, {0, 0x3c00}}
		index, err := NewDenseFlatIndexFP16FromBorrowedRows(ctx, 2, metric, []uint64{1, 2}, rows)
		require.NoError(t, err)
		require.Same(t, &rows[0][0], &index.fp16Row(0)[0])
		require.NoError(t, index.Reserve(1000))
		extra := []uint16{0, 0}
		require.NoError(t, index.AddFP16(ctx, 3, extra))
		extra[0] = 0x3c00
		require.Zero(t, index.fp16Row(2)[0])
		vector, found := index.Vector(1)
		require.True(t, found)
		vector[0] = 10
		require.Equal(t, uint16(0x3c00), rows[0][0])
		reference, err := NewDenseFlatIndexFP16(2, metric)
		require.NoError(t, err)
		for key, row := range rows {
			require.NoError(t, reference.AddFP16(ctx, uint64(key+1), row))
		}
		require.NoError(t, reference.AddFP16(ctx, 3, []uint16{0, 0}))
		got, err := index.Search(ctx, []float32{1, 0}, 3)
		require.NoError(t, err)
		want, err := reference.Search(ctx, []float32{1, 0}, 3)
		require.NoError(t, err)
		require.Equal(t, want, got)
		require.Empty(t, index.vectorsFP16)
	}
}
