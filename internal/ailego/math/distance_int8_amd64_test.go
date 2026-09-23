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
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/cpu"
)

func TestInnerProductInt8AVX2(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 is not supported by this CPU")
	}
	testInnerProductInt8(t, innerProductInt8AVX2)
}

func TestInnerProductInt8AVX512(t *testing.T) {
	if !cpu.X86.HasAVX2 || !cpu.X86.HasAVX512F || !cpu.X86.HasAVX512BW {
		t.Skip("AVX2, AVX-512F and AVX-512BW are not supported by this CPU")
	}
	testInnerProductInt8(t, innerProductInt8AVX512)
}

func TestInnerProductInt8AVX512ScalarFallback(t *testing.T) {
	left := make([]byte, 63)
	right := make([]byte, 63)
	for i := range left {
		left[i] = byte(i - 31)
		right[i] = byte(31 - i)
	}
	require.Equal(t, innerProductInt8Scalar(left, right), innerProductInt8AVX512(left, right))
}
