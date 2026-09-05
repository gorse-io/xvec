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

package rabitq

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestMatrixRotatorIsRandomOrthogonal(t *testing.T) {
	rotator, err := ChooseRotator(4, MatrixRotator, 4)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([][]float32, 4)
	for i := range rows {
		src := make([]float32, 4)
		src[i] = 1
		rows[i] = make([]float32, 4)
		if err := rotator.Rotate(src, rows[i]); err != nil {
			t.Fatal(err)
		}
	}
	identity := true
	for i := range rows {
		for j := range rows {
			want := float32(0)
			if i == j {
				want = 1
			}
			if math.Abs(float64(DotProduct(rows[i], rows[j])-want)) > 1e-5 {
				t.Fatalf("row %d dot row %d = %v, want %v", i, j, DotProduct(rows[i], rows[j]), want)
			}
		}
		if math.Abs(float64(rows[i][i]-1)) > 1e-5 {
			identity = false
		}
	}
	if identity {
		t.Fatal("fresh matrix rotator is the identity matrix")
	}
}

func TestMatrixRotatorSupportsPadding(t *testing.T) {
	rotator, err := ChooseRotator(3, MatrixRotator, 4)
	if err != nil {
		t.Fatal(err)
	}
	if rotator.Size() != 4 || rotator.DumpBytes() != 3*4*4 {
		t.Fatalf("size=%d dump=%d", rotator.Size(), rotator.DumpBytes())
	}
	rows := make([][]float32, 3)
	for i := range rows {
		src := make([]float32, 3)
		src[i] = 1
		rows[i] = make([]float32, 4)
		if err := rotator.Rotate(src, rows[i]); err != nil {
			t.Fatal(err)
		}
	}
	for i := range rows {
		for j := range rows {
			want := float32(0)
			if i == j {
				want = 1
			}
			if math.Abs(float64(DotProduct(rows[i], rows[j])-want)) > 1e-5 {
				t.Fatalf("row %d dot row %d = %v, want %v", i, j, DotProduct(rows[i], rows[j]), want)
			}
		}
	}
}

func TestFHTKacRotatorMatchesUpstream(t *testing.T) {
	const dim, paddedDim = 96, 128
	rotator, err := ChooseRotator(dim, FhtKacRotator, paddedDim)
	if err != nil {
		t.Fatal(err)
	}
	state := make([]byte, rotator.DumpBytes())
	for i := range state {
		state[i] = byte(0xa5 ^ i)
	}
	if err := rotator.Load(state); err != nil {
		t.Fatal(err)
	}
	src, dst := make([]float32, dim), make([]float32, paddedDim)
	for i := range src {
		src[i] = float32(i*17%23-11) / 7
	}
	if err := rotator.Rotate(src, dst); err != nil {
		t.Fatal(err)
	}
	want := map[int]float32{
		0: -1.91071439, 1: 1.27008927, 2: 0.267857075, 31: -0.762276888,
		63: -0.0747767687, 64: -0.229910612, 95: -0.7265625, 127: -0.556919694,
	}
	for index, expected := range want {
		if math.Abs(float64(dst[index]-expected)) > 1e-6 {
			t.Fatalf("output %d = %v, want %v", index, dst[index], expected)
		}
	}
}

func TestRotatorPersistenceAndRotation(t *testing.T) {
	fht, err := ChooseRotator(64, FhtKacRotator, 64)
	if err != nil {
		t.Fatal(err)
	}
	if fht.Size() != 64 || fht.DumpBytes() != 32 {
		t.Fatalf("FHT size=%d dump=%d", fht.Size(), fht.DumpBytes())
	}
	state := make([]byte, fht.DumpBytes())
	if err := fht.Load(state); err != nil {
		t.Fatal(err)
	}
	src, dst := make([]float32, 64), make([]float32, 64)
	src[0] = 1
	if err := fht.Rotate(src, dst); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		want := float32(0)
		if i == 0 {
			want = 1
		}
		if got != want {
			t.Fatalf("FHT output %d = %v, want %v", i, got, want)
		}
	}
	saved := make([]byte, fht.DumpBytes())
	if err := fht.Save(saved); err != nil {
		t.Fatal(err)
	}
	for i := range saved {
		if saved[i] != 0 {
			t.Fatalf("saved byte %d = %d", i, saved[i])
		}
	}

	matrix, err := ChooseRotator(2, MatrixRotator, 2)
	if err != nil {
		t.Fatal(err)
	}
	identity := make([]byte, matrix.DumpBytes())
	binary.LittleEndian.PutUint32(identity[0:], math.Float32bits(1))
	binary.LittleEndian.PutUint32(identity[4:], math.Float32bits(0))
	binary.LittleEndian.PutUint32(identity[8:], math.Float32bits(0))
	binary.LittleEndian.PutUint32(identity[12:], math.Float32bits(1))
	if err := matrix.Load(identity); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, 2)
	if err := matrix.Rotate([]float32{3, -4}, out); err != nil {
		t.Fatal(err)
	}
	if out[0] != 3 || out[1] != -4 {
		t.Fatalf("matrix rotation = %v", out)
	}

	swap := make([]byte, matrix.DumpBytes())
	binary.LittleEndian.PutUint32(swap[0:], math.Float32bits(0))
	binary.LittleEndian.PutUint32(swap[4:], math.Float32bits(1))
	binary.LittleEndian.PutUint32(swap[8:], math.Float32bits(1))
	binary.LittleEndian.PutUint32(swap[12:], math.Float32bits(0))
	if err := matrix.Load(swap); err != nil {
		t.Fatal(err)
	}
	inPlace := []float32{3, -4}
	if err := matrix.Rotate(inPlace, inPlace); err != nil {
		t.Fatal(err)
	}
	if inPlace[0] != -4 || inPlace[1] != 3 {
		t.Fatalf("in-place matrix rotation = %v", inPlace)
	}
	if _, err := ChooseRotator(63, FhtKacRotator, 63); err == nil {
		t.Fatal("invalid FHT padding accepted")
	}
}
