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
		if got := Float32ToFloat16Bits(test.value); got != test.bits {
			t.Errorf("Float32ToFloat16Bits(%v) = %#04x, want %#04x", test.value, got, test.bits)
		}
		if got := Float16BitsToFloat32(test.bits); math.Float32bits(got) != math.Float32bits(test.value) {
			t.Errorf("Float16BitsToFloat32(%#04x) = %v, want %v", test.bits, got, test.value)
		}
	}
	if got := Float32ToFloat16Bits(float32(math.Inf(1))); got != 0x7c00 {
		t.Fatalf("positive infinity = %#04x", got)
	}
	if got := Float32ToFloat16Bits(float32(math.NaN())); got&0x7c00 != 0x7c00 || got&0x03ff == 0 {
		t.Fatalf("NaN = %#04x", got)
	}
}

func TestFloat16RoundToNearestEven(t *testing.T) {
	t.Parallel()
	if got := Float32ToFloat16Bits(1 + float32(math.Ldexp(1, -11))); got != 0x3c00 {
		t.Fatalf("even low tie = %#04x, want 0x3c00", got)
	}
	if got := Float32ToFloat16Bits(1 + 3*float32(math.Ldexp(1, -11))); got != 0x3c02 {
		t.Fatalf("odd low tie = %#04x, want 0x3c02", got)
	}
}

func TestFloat16RoundTripAllBitPatterns(t *testing.T) {
	for bits := range 1 << 16 {
		input := uint16(bits)
		if got := Float32ToFloat16Bits(Float16BitsToFloat32(input)); got != input {
			t.Fatalf("round trip %#04x = %#04x", input, got)
		}
	}
}
