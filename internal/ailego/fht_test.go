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
	"errors"
	"math"
	"slices"
	"testing"
)

func TestFHTInPlace(t *testing.T) {
	t.Parallel()
	data := []float32{1, 2, 3, 4}
	if err := FHTInPlace(data); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(data, []float32{10, -2, -4, 0}) {
		t.Fatalf("transform = %v", data)
	}
	if err := FHTInPlace(data); err != nil {
		t.Fatal(err)
	}
	ScaleFloat32(data, .25)
	if !slices.Equal(data, []float32{1, 2, 3, 4}) {
		t.Fatalf("inverse = %v", data)
	}
}

func TestFHTFlipSigns(t *testing.T) {
	t.Parallel()
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	if err := FHTFlipSigns([]byte{0b10001001, 1}, data); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(data, []float32{-1, 2, 3, -4, 5, 6, 7, -8, -9}) {
		t.Fatalf("flipped = %v", data)
	}
	if err := FHTFlipSigns([]byte{0}, data); !errors.Is(err, ErrShortSignBits) {
		t.Fatalf("short signs error = %v", err)
	}
}

func TestFHTKacWalkRoundTrip(t *testing.T) {
	t.Parallel()
	for _, input := range [][]float32{{1}, {1, 2}, {1, 2, 3, 4, 5}, {-2, 7, 1, 0, 4, -3}} {
		data := slices.Clone(input)
		if err := FHTKacWalk(data); err != nil {
			t.Fatal(err)
		}
		if err := FHTInverseKacWalk(data); err != nil {
			t.Fatal(err)
		}
		for index := range input {
			if difference := math.Abs(float64(data[index] - input[index])); difference > 1e-6 {
				t.Fatalf("length %d element %d = %g, want %g", len(input), index, data[index], input[index])
			}
		}
	}
}

func TestFHTValidation(t *testing.T) {
	t.Parallel()
	for _, data := range [][]float32{nil, {1, 2, 3}} {
		if err := FHTInPlace(data); !errors.Is(err, ErrInvalidFHTLength) {
			t.Fatalf("length %d error = %v", len(data), err)
		}
	}
	if err := FHTKacWalk(nil); !errors.Is(err, ErrInvalidFHTLength) {
		t.Fatalf("empty Kac error = %v", err)
	}
	if err := FHTInverseKacWalk(nil); !errors.Is(err, ErrInvalidFHTLength) {
		t.Fatalf("empty inverse Kac error = %v", err)
	}
}
