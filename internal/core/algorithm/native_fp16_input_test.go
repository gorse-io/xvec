// SPDX-License-Identifier: Apache-2.0

package core

import (
	"context"
	"testing"

	mathutil "github.com/gorse-io/xvec/internal/ailego/math"
	"github.com/gorse-io/xvec/internal/ailego/utility"
	"github.com/stretchr/testify/require"
)

func TestNativeFP16InputOwnershipAndFailures(t *testing.T) {
	ctx := context.Background()
	bits := []uint16{0x8000, 1, 0x7bff, 0x3555}
	flat, err := NewDenseFlatIndexFP16(4, MetricCosine)
	require.NoError(t, err)
	builder, err := NewHNSWBuilderFP16(4, DefaultHNSWBuildOptions(MetricCosine))
	require.NoError(t, err)
	for _, add := range []func(context.Context, uint64, []uint16) error{flat.AddFP16, builder.AddFP16} {
		require.Error(t, add(nil, 0, bits))
		require.ErrorIs(t, add(ctx, 0, bits[:3]), ErrInvalidDimension)
		for _, bad := range []uint16{0x7c00, 0xfc00, 0x7c01, 0xffff} {
			require.ErrorIs(t, add(ctx, 0, []uint16{bad, 0, 0, 0}), mathutil.ErrNonFiniteVector)
		}
		canceled, cancel := context.WithCancel(ctx)
		cancel()
		require.ErrorIs(t, add(canceled, 0, bits), context.Canceled)
		require.NoError(t, add(ctx, 0, bits), "failed additions must not consume the key")
		require.ErrorIs(t, add(ctx, 0, bits), ErrDuplicateKey)
	}
	bits[0] = 123
	require.Equal(t, uint16(0x8000), flat.vectorsFP16[0])
	index, err := builder.Build(ctx)
	require.NoError(t, err)
	require.Equal(t, flat.vectorsFP16, index.vectorsFP16)
	require.ErrorIs(t, builder.AddFP16(ctx, 1, bits), ErrBuilderClosed)
	fp32, err := NewHNSWBuilder(4, DefaultHNSWBuildOptions(MetricL2))
	require.NoError(t, err)
	require.Error(t, fp32.AddFP16(ctx, 0, bits))
}

func TestNativeFP16InputMatchesFloat32Input(t *testing.T) {
	ctx := context.Background()
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		options := DefaultHNSWBuildOptions(metric)
		options.M, options.EFConstruction = 8, 40
		direct, err := NewHNSWBuilderFP16(17, options)
		require.NoError(t, err)
		converted, err := NewHNSWBuilderFP16(17, options)
		require.NoError(t, err)
		for n := 0; n < 128; n++ {
			vector, bits := make([]float32, 17), make([]uint16, 17)
			for d := range bits {
				bits[d] = utility.Float32ToFloat16Bits(float32(n*17+d-1000) / 137)
				vector[d] = utility.Float16BitsToFloat32(bits[d])
			}
			require.NoError(t, direct.AddFP16(ctx, uint64(n), bits))
			require.NoError(t, converted.Add(ctx, uint64(n), vector))
		}
		first, err := direct.Build(ctx)
		require.NoError(t, err)
		second, err := converted.Build(ctx)
		require.NoError(t, err)
		require.Equal(t, second.vectorsFP16, first.vectorsFP16)
		require.Equal(t, second.neighbors, first.neighbors)
	}
}
