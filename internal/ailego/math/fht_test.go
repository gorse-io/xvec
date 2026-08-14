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

package mathutil

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFHTInPlace(t *testing.T) {
	t.Parallel()
	data := []float32{1, 2, 3, 4}
	{
		err := FHTInPlace(data)
		require.NoError(t, err)
	}
	require.True(t, slices.Equal(data, []float32{10, -2, -4, 0}))
	{
		err := FHTInPlace(data)
		require.NoError(t, err)
	}

	ScaleFloat32(data, .25)
	require.True(t, slices.Equal(data, []float32{1, 2, 3, 4}))
}

func TestFHTFlipSigns(t *testing.T) {
	t.Parallel()
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	{
		err := FHTFlipSigns([]byte{0b10001001, 1}, data)
		require.NoError(t, err)
	}
	require.True(t, slices.Equal(data, []float32{-1, 2, 3, -4, 5, 6, 7, -8, -9}))
	{
		err := FHTFlipSigns([]byte{0}, data)
		require.ErrorIs(t, err, ErrShortSignBits)
	}
}

func TestFHTKacWalkRoundTrip(t *testing.T) {
	t.Parallel()
	for _, input := range [][]float32{{1}, {1, 2}, {1, 2, 3, 4, 5}, {-2, 7, 1, 0, 4, -3}} {
		data := slices.Clone(input)
		{
			err := FHTKacWalk(data)
			require.NoError(t, err)
		}
		{
			err := FHTInverseKacWalk(data)
			require.NoError(t, err)
		}

		for index := range input {
			require.InDelta(t, input[index], data[index], 1e-6)
		}
	}
}

func TestFHTValidation(t *testing.T) {
	t.Parallel()
	for _, data := range [][]float32{nil, {1, 2, 3}} {
		{
			err := FHTInPlace(data)
			require.ErrorIs(t, err, ErrInvalidFHTLength)
		}
	}
	{
		err := FHTKacWalk(nil)
		require.ErrorIs(t, err, ErrInvalidFHTLength)
	}
	{
		err := FHTInverseKacWalk(nil)
		require.ErrorIs(t, err, ErrInvalidFHTLength)
	}
}
