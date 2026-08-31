//go:build !noasm && amd64

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
	"golang.org/x/sys/cpu"
)

func TestFHTAVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported")
	}
	testFHTKernels(t, fhtKernels{
		flipSigns:      fhtFlipSignsAVX2,
		kacWalk:        fhtKacWalkAVX2,
		inverseKacWalk: fhtInverseKacWalkAVX2,
		inPlace:        fhtInPlaceAVX2,
	})
}

func TestFHTAVX512ScalarFallback(t *testing.T) {
	signs := []byte{0b10100101, 0b01011010}
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	wantSigns := slices.Clone(input)
	fhtFlipSignsScalar(signs, wantSigns)
	gotSigns := slices.Clone(input)
	fhtFlipSignsAVX512(signs, gotSigns)
	require.Equal(t, wantSigns, gotSigns)

	walkInput := make([]float32, 31)
	for index := range walkInput {
		walkInput[index] = float32(index + 1)
	}
	wantWalk := slices.Clone(walkInput)
	fhtKacWalkScalar(wantWalk)
	gotWalk := slices.Clone(walkInput)
	fhtKacWalkAVX512(gotWalk)
	require.Equal(t, wantWalk, gotWalk)

	wantInverse := slices.Clone(wantWalk)
	fhtInverseKacWalkScalar(wantInverse)
	gotInverse := slices.Clone(gotWalk)
	fhtInverseKacWalkAVX512(gotInverse)
	require.Equal(t, wantInverse, gotInverse)

	inPlaceInput := walkInput[:16]
	wantInPlace := slices.Clone(inPlaceInput)
	fhtInPlaceScalar(wantInPlace)
	gotInPlace := slices.Clone(inPlaceInput)
	fhtInPlaceAVX512(gotInPlace)
	require.Equal(t, wantInPlace, gotInPlace)
}

func TestFHTAVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512DQ {
		t.Skip("AVX-512F and AVX-512DQ are not supported")
	}
	testFHTKernels(t, fhtKernels{
		flipSigns:      fhtFlipSignsAVX512,
		kacWalk:        fhtKacWalkAVX512,
		inverseKacWalk: fhtInverseKacWalkAVX512,
		inPlace:        fhtInPlaceAVX512,
	})
}
