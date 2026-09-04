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

package simd

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/cpu"
)

func TestRotatorAVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported")
	}
	testRotator(t,
		func(flip []byte, data []float32) {
			flip_sign_avx2(unsafe.Pointer(&flip[0]), unsafe.Pointer(&data[0]), int64(len(data)))
		},
		func(data []float32) {
			kacs_walk_avx2(unsafe.Pointer(&data[0]), int64(len(data)))
		},
	)
}

func TestSelectAMD64RotatorKernels(t *testing.T) {
	tests := []struct {
		name     string
		avx512   bool
		avx2     bool
		expected rotatorImplementation
	}{
		{name: "AVX-512", avx512: true, avx2: true, expected: rotatorAVX512},
		{name: "AVX2", avx2: true, expected: rotatorAVX2},
		{name: "scalar", expected: rotatorScalar},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			implementation, _, _ := selectAMD64RotatorKernels(test.avx512, test.avx2)
			require.Equal(t, test.expected, implementation)
		})
	}
}

func TestFlipSignAVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512DQ {
		t.Skip("AVX-512F and AVX-512DQ are not supported")
	}

	const dimension = 128
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

	flip_sign_avx512(unsafe.Pointer(&signs[0]), unsafe.Pointer(&data[0]), int64(len(data)))
	require.Equal(t, want, data)
}

func TestKacsWalkAVX512(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX-512F is not supported")
	}

	data := make([]float32, 64)
	want := make([]float32, 64)
	for i := range 32 {
		x := float32(i + 1)
		y := float32(100 + i)
		data[i], data[i+32] = x, y
		want[i], want[i+32] = x+y, x-y
	}

	kacs_walk_avx512(unsafe.Pointer(&data[0]), int64(len(data)))
	require.Equal(t, want, data)
}
