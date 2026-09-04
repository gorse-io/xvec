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

package simd

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestScalarRotatorKernels(t *testing.T) {
	testRotatorKernels(t,
		func(flip []byte, data []float32) {
			flipSignScalarKernel(unsafe.Pointer(&flip[0]), unsafe.Pointer(&data[0]), int64(len(data)))
		},
		func(data []float32) {
			kacsWalkScalarKernel(unsafe.Pointer(&data[0]), int64(len(data)))
		},
	)
}

func TestRotatorDispatch(t *testing.T) {
	testRotatorKernels(t, FlipSign, KacsWalk)
}

func testRotatorKernels(
	t *testing.T,
	flipSign func([]byte, []float32),
	kacsWalk func([]float32),
) {
	t.Helper()

	for _, dimension := range []int{64, 128} {
		signs := make([]byte, dimension/8)
		data := make([]float32, dimension)
		want := make([]float32, dimension)
		for i := range dimension {
			data[i] = float32(i + 1)
			want[i] = data[i]
			if i%3 == 0 || i%11 == 0 {
				signs[i/8] |= 1 << (i % 8)
				want[i] = -want[i]
			}
		}

		flipSign(signs, data)
		require.Equal(t, want, data, "dimension %d", dimension)
	}

	for _, length := range []int{32, 64, 128} {
		data := make([]float32, length)
		want := make([]float32, length)
		for i := range length / 2 {
			x := float32(i + 1)
			y := float32(100 + i)
			data[i], data[i+length/2] = x, y
			want[i], want[i+length/2] = x+y, x-y
		}

		kacsWalk(data)
		require.Equal(t, want, data, "length %d", length)
	}
}
