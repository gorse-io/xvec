// Copyright 2026-present the zvec-go project
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

package ailego

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFloat16KnownValues(t *testing.T) {
	t.Parallel()
	tests := []struct {
		value float32
		bits  uint16
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
			got := Float32ToFloat16Bits(test.value)
			assert.Equal(t, test.bits, got)
		}
		{
			got := Float16BitsToFloat32(test.bits)
			assert.Equal(t, math.Float32bits(test.value), math.Float32bits(got))
		}
	}
	{
		got := Float32ToFloat16Bits(float32(math.Inf(1)))
		require.True(t, got == 0x7c00)
	}
	{
		got := Float32ToFloat16Bits(float32(math.NaN()))
		require.True(t, got&0x7c00 == 0x7c00)
		require.False(t, got&0x03ff == 0)
	}
}

func TestFloat16RoundToNearestEven(t *testing.T) {
	t.Parallel()
	{
		got := Float32ToFloat16Bits(1 + float32(math.Ldexp(1, -11)))
		require.True(t, got == 0x3c00)
	}
	{
		got := Float32ToFloat16Bits(1 + 3*float32(math.Ldexp(1, -11)))
		require.True(t, got == 0x3c02)
	}
}

func TestFloat16RoundTripAllBitPatterns(t *testing.T) {
	for bits := range 1 << 16 {
		input := uint16(bits)
		{
			got := Float32ToFloat16Bits(Float16BitsToFloat32(input))
			require.Equal(t, input, got)
		}
	}
}
