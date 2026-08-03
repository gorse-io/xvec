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

package core

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestFHTRotatorPowerOfTwoFixture(t *testing.T) {
	t.Parallel()
	rotator, err := NewFHTRotatorFromSigns(4, []byte{0b0001, 0b0010, 0b0100, 0b1000})
	if err != nil {
		t.Fatal(err)
	}
	input := []float32{1, 2, 3, 4}
	rotated, err := rotator.Rotate(input)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{-3, 4, -1, -2}
	assertFloatSlicesClose(t, rotated, want, 1e-6)
	if !slices.Equal(input, []float32{1, 2, 3, 4}) {
		t.Fatal("Rotate mutated input")
	}
	reverted, err := rotator.Unrotate(rotated)
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesClose(t, reverted, input, 1e-6)
}

func TestFHTRotatorArbitraryDimensions(t *testing.T) {
	t.Parallel()
	for _, dimension := range []int{1, 3, 5, 7, 12, 97} {
		signs := make([]byte, 4*((dimension+7)/8))
		for index := range signs {
			signs[index] = byte(index*73 + dimension)
		}
		rotator, err := NewFHTRotatorFromSigns(dimension, signs)
		if err != nil {
			t.Fatal(err)
		}
		input := make([]float32, dimension)
		for index := range input {
			input[index] = float32(index%11-5) / 3
		}
		rotated, err := rotator.Rotate(input)
		if err != nil {
			t.Fatalf("dimension %d rotate: %v", dimension, err)
		}
		reverted, err := rotator.Unrotate(rotated)
		if err != nil {
			t.Fatalf("dimension %d unrotate: %v", dimension, err)
		}
		assertFloatSlicesClose(t, reverted, input, 2e-4)
		if difference := math.Abs(vectorNormSquared(rotated) - vectorNormSquared(input)); difference > 2e-4*max(1, vectorNormSquared(input)) {
			t.Fatalf("dimension %d norm difference = %g", dimension, difference)
		}
	}
}

func TestFHTRotatorPreservesMetrics(t *testing.T) {
	t.Parallel()
	const dimension = 11
	rotator, err := NewFHTRotatorFromSigns(dimension, []byte{1, 7, 13, 29, 61, 127, 193, 251})
	if err != nil {
		t.Fatal(err)
	}
	left := []float32{1, -2, 3, 4, -.5, 7, 0, 2, -9, 1.5, 8}
	right := []float32{-3, 1, 2, 0, 6, -.25, 4, 9, 2, -1, 5}
	leftRotated, _ := rotator.Rotate(left)
	rightRotated, _ := rotator.Rotate(right)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		want, err := metric.Compute(left, right)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metric.Compute(leftRotated, rightRotated)
		if err != nil {
			t.Fatal(err)
		}
		if difference := math.Abs(float64(got - want)); difference > 2e-5*max(1, math.Abs(float64(want))) {
			t.Fatalf("metric %d rotated score = %g, original = %g", metric, got, want)
		}
	}
}

func TestFHTRotatorStateAndRandomness(t *testing.T) {
	t.Parallel()
	random := bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	rotator, err := NewFHTRotatorWithReader(9, random)
	if err != nil {
		t.Fatal(err)
	}
	if rotator.Dimension() != 9 || !slices.Equal(rotator.Signs(), []byte{1, 2, 3, 4, 5, 6, 7, 8}) {
		t.Fatalf("rotator state = (%d, %v)", rotator.Dimension(), rotator.Signs())
	}
	signs := rotator.Signs()
	signs[0] = 99
	if rotator.Signs()[0] != 1 {
		t.Fatal("Signs exposed mutable state")
	}
	restored, err := NewFHTRotatorFromSigns(9, rotator.Signs())
	if err != nil {
		t.Fatal(err)
	}
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	left, _ := rotator.Rotate(input)
	right, _ := restored.Rotate(input)
	if !slices.Equal(left, right) {
		t.Fatalf("restored rotation differs: %v vs %v", left, right)
	}

	if _, err := NewFHTRotatorWithReader(9, io.LimitReader(bytes.NewReader([]byte{1}), 1)); err == nil {
		t.Fatal("short randomness accepted")
	}
}

func TestFHTRotatorValidation(t *testing.T) {
	t.Parallel()
	if _, err := NewFHTRotator(0); !errors.Is(err, ErrInvalidRotator) {
		t.Fatalf("zero dimension error = %v", err)
	}
	if _, err := NewFHTRotator(MaxRotationDimension + 1); !errors.Is(err, ErrInvalidRotator) {
		t.Fatalf("oversized dimension error = %v", err)
	}
	if _, err := NewFHTRotatorWithReader(4, nil); !errors.Is(err, ErrInvalidRotator) {
		t.Fatalf("nil random error = %v", err)
	}
	if _, err := NewFHTRotatorFromSigns(9, make([]byte, 7)); !errors.Is(err, ErrInvalidSigns) {
		t.Fatalf("short state error = %v", err)
	}
	rotator, _ := NewFHTRotatorFromSigns(4, make([]byte, 4))
	if _, err := rotator.Rotate([]float32{1}); !errors.Is(err, ailego.ErrDimensionMismatch) {
		t.Fatalf("dimension error = %v", err)
	}
	if _, err := rotator.Rotate([]float32{1, 2, 3, float32(math.NaN())}); !errors.Is(err, ailego.ErrNonFiniteVector) {
		t.Fatalf("non-finite error = %v", err)
	}
	var nilRotator *FHTRotator
	if _, err := nilRotator.Rotate([]float32{1}); !errors.Is(err, ErrInvalidRotator) {
		t.Fatalf("nil rotator error = %v", err)
	}
}

func TestFHTRotatorBatchAndConcurrency(t *testing.T) {
	t.Parallel()
	rotator, _ := NewFHTRotatorFromSigns(4, []byte{1, 2, 3, 4})
	input := [][]float32{{1, 2, 3, 4}, {-1, 0, 2, 5}, {9, 8, 7, 6}}
	batch, err := rotator.RotateBatch(context.Background(), input, 2)
	if err != nil {
		t.Fatal(err)
	}
	for index := range input {
		want, _ := rotator.Rotate(input[index])
		if !slices.Equal(batch[index], want) {
			t.Fatalf("batch %d = %v, want %v", index, batch[index], want)
		}
	}
	input[0][0] = 100
	if batch[0][0] == 100 {
		t.Fatal("batch aliases input")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := rotator.RotateBatch(ctx, input, 2); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled batch error = %v", err)
	}
	if _, err := rotator.RotateBatch(nil, input, 2); err == nil {
		t.Fatal("nil context accepted")
	}

	const goroutines = 16
	want, _ := rotator.Rotate([]float32{1, 2, 3, 4})
	var wait sync.WaitGroup
	wait.Add(goroutines)
	for range goroutines {
		go func() {
			defer wait.Done()
			got, err := rotator.Rotate([]float32{1, 2, 3, 4})
			if err != nil || !slices.Equal(got, want) {
				t.Errorf("concurrent rotation = %v, %v", got, err)
			}
		}()
	}
	wait.Wait()
}

func TestRotationReformer(t *testing.T) {
	t.Parallel()
	rotator, _ := NewFHTRotatorFromSigns(4, []byte{1, 2, 3, 4})
	reformer, err := NewRotationReformer(rotator)
	if err != nil {
		t.Fatal(err)
	}
	input := []float32{1, 2, 3, 4}
	transformed, err := reformer.Transform(input)
	if err != nil {
		t.Fatal(err)
	}
	reverted, err := reformer.Revert(transformed)
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesClose(t, reverted, input, 1e-5)
	if reformer.Dimension() != 4 {
		t.Fatalf("dimension = %d", reformer.Dimension())
	}
	if _, err := NewRotationReformer(nil); !errors.Is(err, ErrInvalidRotator) {
		t.Fatalf("nil rotator error = %v", err)
	}
}

func FuzzFHTRotatorRoundTrip(f *testing.F) {
	f.Add(uint8(5), uint64(0x0102030405060708), float32(1), float32(-2))
	f.Add(uint8(8), uint64(0xfedcba9876543210), float32(.25), float32(9))
	f.Add(uint8(191), uint64(0x0102030405060743), float32(1.0/7), float32(-763))
	f.Fuzz(func(t *testing.T, rawDimension uint8, seed uint64, first, second float32) {
		dimension := int(rawDimension%64) + 1
		signs := make([]byte, 4*((dimension+7)/8))
		for index := range signs {
			signs[index] = byte(seed >> uint(index%8*8))
		}
		rotator, err := NewFHTRotatorFromSigns(dimension, signs)
		if err != nil {
			t.Fatal(err)
		}
		vector := make([]float32, dimension)
		for index := range vector {
			if index%2 == 0 {
				vector[index] = first
			} else {
				vector[index] = second
			}
		}
		rotated, err := rotator.Rotate(vector)
		if err != nil {
			if errors.Is(err, ailego.ErrNonFiniteVector) {
				return
			}
			t.Fatal(err)
		}
		reverted, err := rotator.Unrotate(rotated)
		if err != nil {
			if errors.Is(err, ailego.ErrNonFiniteVector) {
				return
			}
			t.Fatal(err)
		}
		inputScale := max(1, math.Abs(float64(first)), math.Abs(float64(second)))
		for index := range vector {
			tolerance := 2e-5 * inputScale
			if math.Abs(float64(reverted[index]-vector[index])) > tolerance {
				t.Fatalf("dimension %d element %d = %g, want %g", dimension, index, reverted[index], vector[index])
			}
		}
	})
}

func vectorNormSquared(vector []float32) float64 {
	var sum float64
	for _, value := range vector {
		sum += float64(value) * float64(value)
	}
	return sum
}
