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

package zvec

import (
	"errors"
	"math"
	"slices"
	"testing"
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
		if got := Float16FromFloat32(test.value); got != test.bits {
			t.Errorf("Float16FromFloat32(%v) = %#04x, want %#04x", test.value, got, test.bits)
		}
		if got := test.bits.Float32(); math.Float32bits(got) != math.Float32bits(test.value) {
			t.Errorf("Float16(%#04x).Float32() = %v, want %v", test.bits, got, test.value)
		}
	}
	if got := Float16FromFloat32(float32(math.Inf(1))); got != 0x7c00 {
		t.Fatalf("positive infinity = %#04x", got)
	}
	if got := Float16FromFloat32(float32(math.NaN())); uint16(got)&0x7c00 != 0x7c00 || uint16(got)&0x03ff == 0 {
		t.Fatalf("NaN = %#04x", got)
	}
}

func TestFloat16RoundTripAllBitPatterns(t *testing.T) {
	for bits := range 1 << 16 {
		value := Float16(bits)
		if got := Float16FromFloat32(value.Float32()); got != value {
			t.Fatalf("round trip %#04x = %#04x", value, got)
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
		if vector.DataType() != wantTypes[index] || vector.Dimension() != 1 {
			t.Fatalf("vector %d metadata = (%s, %d)", index, vector.DataType(), vector.Dimension())
		}
	}
	if err := (VectorInt4{-8, 0, 7}).Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (VectorInt4{8}).Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid INT4 error = %v", err)
	}
}

func TestSparseVectorCanonical(t *testing.T) {
	vector := SparseVectorFP32{
		Indices: []uint32{9, 2, 5},
		Values:  []float32{0.9, 0.2, 0.5},
	}
	canonical, err := vector.Canonical()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(canonical.Indices, []uint32{2, 5, 9}) ||
		!slices.Equal(canonical.Values, []float32{0.2, 0.5, 0.9}) {
		t.Fatalf("canonical vector = %#v", canonical)
	}
	if !slices.Equal(vector.Indices, []uint32{9, 2, 5}) {
		t.Fatal("Canonical mutated its input")
	}

	if err := (SparseVectorFP16{Indices: []uint32{1}, Values: nil}).Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("length mismatch error = %v", err)
	}
	if err := (SparseVectorFP32{Indices: []uint32{2, 2}, Values: []float32{1, 2}}).Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("duplicate error = %v", err)
	}
	tooLarge := SparseVectorFP32{
		Indices: make([]uint32, MaxSparseDimensions+1),
		Values:  make([]float32, MaxSparseDimensions+1),
	}
	if err := tooLarge.Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("coordinate count error = %v", err)
	}
}
