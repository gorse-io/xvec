// SPDX-License-Identifier: Apache-2.0
package core

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/utility"
	"github.com/stretchr/testify/require"
)

func TestHNSWBorrowedFP16Storage(t *testing.T) {
	ctx := context.Background()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		t.Run(fmt.Sprint(metric), func(t *testing.T) {
			options := DefaultHNSWBuildOptions(metric)
			options.M, options.EFConstruction = 8, 40
			builder, err := NewHNSWBuilderWithBorrowedFP16(17, options)
			require.NoError(t, err)
			require.NoError(t, builder.Reserve(DefaultHNSWBruteForceThreshold+1))
			rows := make(map[uint64][]uint16)
			for n := 0; n <= DefaultHNSWBruteForceThreshold; n++ {
				row := make([]uint16, 17)
				for d := range row {
					row[d] = utility.Float32ToFloat16Bits(float32(n*17+d-1000) / 137)
				}
				rows[uint64(n)] = row
				require.NoError(t, builder.AddBorrowedFP16(ctx, uint64(n), row))
			}
			index, err := builder.Build(ctx)
			require.NoError(t, err)
			require.Empty(t, index.vectorsFP16)
			require.Same(t, &rows[0][0], &index.vectorFP16At(0)[0])
			path := filepath.Join(t.TempDir(), "index")
			require.NoError(t, index.Save(ctx, path))
			query, _ := index.Vector(31)
			want, err := index.Search(ctx, query, 10)
			require.NoError(t, err)
			for _, mapped := range []bool{false, true} {
				reopened, err := OpenHNSWIndexWithBorrowedFP16(ctx, path, rows, mapped)
				require.NoError(t, err)
				require.Empty(t, reopened.vectorsFP16)
				require.NotNil(t, reopened.compactNeighbors)
				require.Same(t, &rows[0][0], &reopened.vectorFP16At(0)[0])
				got, err := reopened.Search(ctx, query, 10)
				require.NoError(t, err)
				require.Equal(t, want, got)
				require.NoError(t, reopened.Add(ctx, 2000, query))
				require.Nil(t, reopened.vectorRowsFP16, "streaming must own its new generation")
				require.NotSame(t, &rows[0][0], &reopened.vectorFP16At(0)[0])
				modified := filepath.Join(t.TempDir(), "modified")
				require.NoError(t, reopened.Save(ctx, modified))
				owned, err := OpenHNSWIndex(ctx, modified)
				require.NoError(t, err)
				vector, found := owned.Vector(2000)
				require.True(t, found)
				require.Equal(t, query, vector)
			}
			saved := rows[0][0]
			rows[0][0] ^= 1
			_, err = OpenHNSWIndexWithBorrowedFP16(ctx, path, rows, true)
			require.ErrorIs(t, err, ErrInvalidHNSWFile)
			rows[0][0] = saved
			delete(rows, 0)
			_, err = OpenHNSWIndexWithBorrowedFP16(ctx, path, rows, true)
			require.ErrorIs(t, err, ErrInvalidHNSWFile)
		})
	}
}

func TestBorrowedFP16FlatAppendOwnership(t *testing.T) {
	ctx := context.Background()
	flat, err := NewDenseFlatIndexFP16FromBorrowedRows(ctx, 2, MetricL2, nil, nil)
	require.NoError(t, err)
	row := []uint16{0x3c00, 0}
	require.NoError(t, flat.AddBorrowedFP16(ctx, 1, row))
	require.Same(t, &row[0], &flat.fp16Row(0)[0])
	require.NoError(t, flat.AddFP16(ctx, 2, row))
	require.NotSame(t, &row[0], &flat.fp16Row(1)[0])
	require.ErrorIs(t, flat.AddBorrowedFP16(ctx, 1, row), ErrDuplicateKey)
	require.Error(t, flat.AddBorrowedFP16(ctx, 3, row[:1]))
}

func TestHNSWBorrowedFP16InputFailuresAndCopying(t *testing.T) {
	ctx := context.Background()
	builder, err := NewHNSWBuilderWithBorrowedFP16(2, DefaultHNSWBuildOptions(MetricL2))
	require.NoError(t, err)
	row := []uint16{0x3c00, 0}
	require.Error(t, builder.AddBorrowedFP16(nil, 1, row))
	require.ErrorIs(t, builder.AddBorrowedFP16(ctx, 1, row[:1]), ErrInvalidDimension)
	require.Error(t, builder.AddBorrowedFP16(ctx, 1, []uint16{0x7c00, 0}))
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	require.ErrorIs(t, builder.AddBorrowedFP16(canceled, 1, row), context.Canceled)
	require.NoError(t, builder.AddFP16(ctx, 1, row))
	row[0] = 0
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	require.Equal(t, uint16(0x3c00), index.vectorFP16At(0)[0])
	got, err := index.Search(ctx, []float32{1, 0}, 1)
	require.NoError(t, err)
	require.Equal(t, float32(0), got[0].Score)
	require.ErrorIs(t, builder.AddBorrowedFP16(ctx, 2, row), ErrBuilderClosed)
	plain, err := NewHNSWBuilderFP16(2, DefaultHNSWBuildOptions(MetricL2))
	require.NoError(t, err)
	require.Error(t, plain.AddBorrowedFP16(ctx, 1, row))
	require.NoError(t, plain.AddFP16(ctx, 1, row), "failed borrowing must not consume the key")
}
