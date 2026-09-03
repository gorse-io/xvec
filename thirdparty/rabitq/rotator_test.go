// Copyright 2026-present the xvec project
// Licensed under the Apache License, Version 2.0.

package rabitq

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"testing"
)

type shortWriter struct{}

func (shortWriter) Write(data []byte) (int, error) { return len(data) / 2, nil }

func TestFHTKacRotatorStateAndNorm(t *testing.T) {
	const dim, padded = 70, 128
	signs := make([]byte, 4*padded/8)
	for i := range signs {
		signs[i] = byte(i*37 + 11)
	}
	r, err := NewFHTKacRotatorFromState(dim, padded, signs)
	if err != nil {
		t.Fatal(err)
	}
	data, _, _ := fixture(dim)
	got, err := r.Rotate(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != padded {
		t.Fatalf("rotated length = %d", len(got))
	}
	var before, after float64
	for _, x := range data {
		before += float64(x * x)
	}
	for _, x := range got {
		after += float64(x * x)
	}
	if math.Abs(before-after) > before*2e-5 {
		t.Fatalf("norm changed: %g -> %g", before, after)
	}
	state, err := r.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := NewFHTKacRotatorFromState(dim, padded, state)
	if err != nil {
		t.Fatal(err)
	}
	got2, _ := restored.Rotate(data)
	if !bytes.Equal(float32Bytes(got), float32Bytes(got2)) {
		t.Fatal("restored rotation differs")
	}
	state[0] ^= 0xff
	if bytes.Equal(state, restored.State()) {
		t.Fatal("state aliases caller")
	}
}

func TestMatrixRotatorState(t *testing.T) {
	// Two orthonormal rows in R^3.
	matrix := []float32{1, 0, 0, 0, 0.6, 0.8}
	r, err := NewMatrixRotatorFromMatrix(2, 3, matrix)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.Rotate([]float32{2, 5})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 3, 4}
	for i := range want {
		if !close32(got[i], want[i]) {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	state, err := r.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := NewMatrixRotatorFromState(2, 3, state)
	if err != nil {
		t.Fatal(err)
	}
	got2, _ := restored.Rotate([]float32{2, 5})
	if !bytes.Equal(float32Bytes(got), float32Bytes(got2)) {
		t.Fatal("restored matrix differs")
	}
	if _, err := NewMatrixRotatorFromMatrix(2, 3, []float32{1, 0, 0, 1, 0, 0}); err == nil {
		t.Fatal("accepted non-orthonormal matrix")
	}
	nonOrthonormal := float32Bytes([]float32{1, 0, 0, 1, 0, 0})
	if _, err := NewMatrixRotatorFromState(2, 3, nonOrthonormal); err != nil {
		t.Fatalf("raw state loader rejected a C++-accepted payload: %v", err)
	}
	var saved bytes.Buffer
	if err := r.Save(&saved); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(saved.Bytes(), state) || r.BinarySize() != len(state) || r.DumpBytes() != len(state) {
		t.Fatal("matrix serialization helpers disagree")
	}
	if err := r.Save(shortWriter{}); err != io.ErrShortWrite {
		t.Fatalf("short writer error = %v, want %v", err, io.ErrShortWrite)
	}
}

func TestPaddedDimensionAndLoadRotator(t *testing.T) {
	if got, err := PaddedDimension(65, RotatorFHTKac); err != nil || got != 128 {
		t.Fatalf("FHT padded dimension = %d, %v", got, err)
	}
	if got, err := PaddedDimension(65, RotatorMatrix); err != nil || got != 65 {
		t.Fatalf("matrix padded dimension = %d, %v", got, err)
	}
	state := make([]byte, 4*128/8)
	rotator, err := LoadRotator(65, 128, RotatorFHTKac, state)
	if err != nil {
		t.Fatal(err)
	}
	if rotator.Type() != RotatorFHTKac || rotator.BinarySize() != len(state) {
		t.Fatalf("loaded rotator = type %d size %d", rotator.Type(), rotator.BinarySize())
	}
}

func TestChooseRotatorBothPaths(t *testing.T) {
	fht, err := NewRotator(64, 64, RotatorFHTKac, bytes.NewReader(make([]byte, 32)))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := fht.(*FHTKacRotator); !ok {
		t.Fatalf("FHT path returned %T", fht)
	}
	matrixRandom := make([]byte, 2*3*8)
	for i := range matrixRandom {
		matrixRandom[i] = byte(i*29 + 7)
	}
	matrix, err := NewRotator(2, 3, RotatorMatrix, bytes.NewReader(matrixRandom))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := matrix.(*MatrixRotator); !ok {
		t.Fatalf("matrix path returned %T", matrix)
	}
	chosen, err := ChooseRotator(64, 0, RotatorFHTKac)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := chosen.(*FHTKacRotator); !ok || chosen.PaddedDimension() != 64 {
		t.Fatalf("ChooseRotator returned %T with padded dimension %d", chosen, chosen.PaddedDimension())
	}
	if _, err := NewRotator(64, 64, RotatorType(99), bytes.NewReader(make([]byte, 32))); err == nil {
		t.Fatal("accepted unknown rotator type")
	}
}

func TestRotatorValidation(t *testing.T) {
	if _, err := NewFHTKacRotatorFromState(65, 65, make([]byte, 36)); err == nil {
		t.Fatal("accepted non-64-aligned padded dimension")
	}
	if _, err := NewMatrixRotatorFromState(2, 3, make([]byte, 23)); err == nil {
		t.Fatal("accepted short matrix state")
	}
}

func float32Bytes(values []float32) []byte {
	out := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(value))
	}
	return out
}
