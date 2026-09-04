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

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/cpu"
)

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

	FlipSignAVX512(signs, data)
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

	KacsWalkAVX512(data)
	require.Equal(t, want, data)
}
