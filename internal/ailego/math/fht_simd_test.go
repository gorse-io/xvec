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
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func testFHTKernels(t *testing.T, kernels fhtKernels) {
	t.Helper()

	for _, size := range []int{1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			input := make([]float32, size)
			for index := range input {
				input[index] = rand.Float32()*20 - 10
			}

			signs := make([]byte, (size+7)/8)
			for index := range signs {
				signs[index] = byte(rand.Uint32())
			}
			wantSigns := slices.Clone(input)
			fhtFlipSignsScalar(signs, wantSigns)
			gotSigns := slices.Clone(input)
			kernels.flipSigns(signs, gotSigns)
			require.Equal(t, wantSigns, gotSigns)

			wantWalk := slices.Clone(input)
			fhtKacWalkScalar(wantWalk)
			gotWalk := slices.Clone(input)
			kernels.kacWalk(gotWalk)
			require.Equal(t, wantWalk, gotWalk)

			wantInverse := slices.Clone(wantWalk)
			fhtInverseKacWalkScalar(wantInverse)
			gotInverse := slices.Clone(gotWalk)
			kernels.inverseKacWalk(gotInverse)
			require.Equal(t, wantInverse, gotInverse)
		})
	}

	for _, size := range []int{1, 2, 4, 8, 16, 32, 64, 128, 256} {
		input := make([]float32, size)
		for index := range input {
			input[index] = rand.Float32()*20 - 10
		}
		want := slices.Clone(input)
		fhtInPlaceScalar(want)
		got := slices.Clone(input)
		kernels.inPlace(got)
		require.Equal(t, want, got)
	}
}
