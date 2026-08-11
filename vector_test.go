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

package xvec

import (
	"math"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFloat16KnownValues(t *testing.T) {
	tests := []struct {
		value float32
		bits  Float16
	}{
		{0, 0x0000},
		{float32(math.Copysign(0, -1)), 0x8000},
		{1, 0x3c00},
		{-2, 0xc000},
		{65504, 0x7bff},
		{float32(math.Ldexp(1, -14)), 0x0400},
		{float32(math.Ldexp(1, -24)), 0x0001},
	}
	for _, test := range tests {
		{
			got := Float16FromFloat32(test.value)
			assert.Equal(t, test.bits, got)
		}
		{
			got := test.bits.Float32()
			assert.Equal(t, math.Float32bits(test.value), math.Float32bits(got))
		}
	}
	{
		got := Float16FromFloat32(float32(math.Inf(1)))
		require.True(t, got == 0x7c00)
	}
	{
		got := Float16FromFloat32(float32(math.NaN()))
		require.True(t, uint16(got)&0x7c00 == 0x7c00)
		require.False(t, uint16(got)&0x03ff == 0)
	}
}

func TestFloat16RoundTripAllBitPatterns(t *testing.T) {
	for bits := range 1 << 16 {
		value := Float16(bits)
		{
			got := Float16FromFloat32(value.Float32())
			require.Equal(t, value, got)
		}
	}
}

func TestDenseVectorTypes(t *testing.T) {
	vectors := []DenseVector{
		VectorBinary32{1}, VectorBinary64{1}, VectorFP16{1}, VectorFP32{1},
		VectorFP64{1}, VectorInt4{1}, VectorInt8{1}, VectorInt16{1},
	}
	wantTypes := []DataType{
		DataTypeVectorBinary32, DataTypeVectorBinary64, DataTypeVectorFP16,
		DataTypeVectorFP32, DataTypeVectorFP64, DataTypeVectorInt4,
		DataTypeVectorInt8, DataTypeVectorInt16,
	}
	for index, vector := range vectors {
		require.Equal(t, wantTypes[index], vector.DataType())
		require.True(t, vector.Dimension() == 1)
	}
	{
		err := (VectorInt4{-8, 0, 7}).Validate()
		require.NoError(t, err)
	}
	{
		err := (VectorInt4{8}).Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
}

func TestSparseVectorCanonical(t *testing.T) {
	vector := SparseVectorFP32{
		Indices: []uint32{9, 2, 5},
		Values:  []float32{0.9, 0.2, 0.5},
	}
	canonical, err := vector.Canonical()
	require.NoError(t, err)
	require.True(t, slices.Equal(canonical.Indices, []uint32{2, 5, 9}))
	require.True(t, slices.Equal(canonical.Values, []float32{0.2, 0.5, 0.9}))
	require.True(t, slices.Equal(vector.Indices, []uint32{9, 2, 5}),
		"Canonical mutated its input")
	{
		err := (SparseVectorFP16{Indices: []uint32{1}, Values: nil}).Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
	{
		err := (SparseVectorFP32{Indices: []uint32{2, 2}, Values: []float32{1, 2}}).Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
	}

	tooLarge := SparseVectorFP32{
		Indices: make([]uint32, MaxSparseDimensions+1),
		Values:  make([]float32, MaxSparseDimensions+1),
	}
	{
		err := tooLarge.Validate()
		require.ErrorIs(t, err, ErrInvalidArgument)
	}
}
