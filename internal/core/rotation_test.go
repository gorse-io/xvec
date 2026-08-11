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

	"github.com/gorse-io/xvec/internal/ailego"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFHTRotatorPowerOfTwoFixture(t *testing.T) {
	t.Parallel()
	rotator, err := NewFHTRotatorFromSigns(4, []byte{0b0001, 0b0010, 0b0100, 0b1000})
	require.NoError(t, err)

	input := []float32{1, 2, 3, 4}
	rotated, err := rotator.Rotate(input)
	require.NoError(t, err)

	want := []float32{-3, 4, -1, -2}
	assertFloatSlicesClose(t, rotated, want, 1e-6)
	require.True(t, slices.Equal(input, []float32{1, 2, 3, 4}),
		"Rotate mutated input")

	reverted, err := rotator.Unrotate(rotated)
	require.NoError(t, err)

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
		require.NoError(t, err)

		input := make([]float32, dimension)
		for index := range input {
			input[index] = float32(index%11-5) / 3
		}
		rotated, err := rotator.Rotate(input)
		require.NoError(t, err)

		reverted, err := rotator.Unrotate(rotated)
		require.NoError(t, err)

		assertFloatSlicesClose(t, reverted, input, 2e-4)
		require.InDelta(t, vectorNormSquared(input), vectorNormSquared(rotated), 2e-4*max(1, vectorNormSquared(input)))
	}
}

func TestFHTRotatorPreservesMetrics(t *testing.T) {
	t.Parallel()
	const dimension = 11
	rotator, err := NewFHTRotatorFromSigns(dimension, []byte{1, 7, 13, 29, 61, 127, 193, 251})
	require.NoError(t, err)

	left := []float32{1, -2, 3, 4, -.5, 7, 0, 2, -9, 1.5, 8}
	right := []float32{-3, 1, 2, 0, 6, -.25, 4, 9, 2, -1, 5}
	leftRotated, _ := rotator.Rotate(left)
	rightRotated, _ := rotator.Rotate(right)
	for _, metric := range []Metric{MetricL2, MetricIP, MetricCosine, MetricMIPSL2} {
		want, err := metric.Compute(left, right)
		require.NoError(t, err)

		got, err := metric.Compute(leftRotated, rightRotated)
		require.NoError(t, err)
		require.InDelta(t, want, got, 2e-5*max(1, math.Abs(float64(want))))
	}
}

func TestFHTRotatorStateAndRandomness(t *testing.T) {
	t.Parallel()
	random := bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	rotator, err := NewFHTRotatorWithReader(9, random)
	require.NoError(t, err)
	require.True(t, rotator.Dimension() == 9)
	require.True(t, slices.Equal(rotator.Signs(), []byte{1, 2, 3, 4, 5, 6, 7, 8}))

	signs := rotator.Signs()
	signs[0] = 99
	require.True(t, rotator.Signs()[0] == 1,
		"Signs exposed mutable state")

	restored, err := NewFHTRotatorFromSigns(9, rotator.Signs())
	require.NoError(t, err)

	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	left, _ := rotator.Rotate(input)
	right, _ := restored.Rotate(input)
	require.True(t, slices.Equal(left, right))
	{
		_, err := NewFHTRotatorWithReader(9, io.LimitReader(bytes.NewReader([]byte{1}), 1))
		require.Error(t, err,
			"short randomness accepted")
	}
}

func TestFHTRotatorValidation(t *testing.T) {
	t.Parallel()
	{
		_, err := NewFHTRotator(0)
		require.ErrorIs(t, err, ErrInvalidRotator)
	}
	{
		_, err := NewFHTRotator(MaxRotationDimension + 1)
		require.ErrorIs(t, err, ErrInvalidRotator)
	}
	{
		_, err := NewFHTRotatorWithReader(4, nil)
		require.ErrorIs(t, err, ErrInvalidRotator)
	}
	{
		_, err := NewFHTRotatorFromSigns(9, make([]byte, 7))
		require.ErrorIs(t, err, ErrInvalidSigns)
	}

	rotator, _ := NewFHTRotatorFromSigns(4, make([]byte, 4))
	{
		_, err := rotator.Rotate([]float32{1})
		require.ErrorIs(t, err, ailego.ErrDimensionMismatch)
	}
	{
		_, err := rotator.Rotate([]float32{1, 2, 3, float32(math.NaN())})
		require.ErrorIs(t, err, ailego.ErrNonFiniteVector)
	}

	var nilRotator *FHTRotator
	{
		_, err := nilRotator.Rotate([]float32{1})
		require.ErrorIs(t, err, ErrInvalidRotator)
	}
}

func TestFHTRotatorBatchAndConcurrency(t *testing.T) {
	t.Parallel()
	rotator, _ := NewFHTRotatorFromSigns(4, []byte{1, 2, 3, 4})
	input := [][]float32{{1, 2, 3, 4}, {-1, 0, 2, 5}, {9, 8, 7, 6}}
	batch, err := rotator.RotateBatch(context.Background(), input, 2)
	require.NoError(t, err)

	for index := range input {
		want, _ := rotator.Rotate(input[index])
		require.True(t, slices.Equal(batch[index], want))
	}
	input[0][0] = 100
	require.False(t, batch[0][0] == 100,
		"batch aliases input")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := rotator.RotateBatch(ctx, input, 2)
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := rotator.RotateBatch(nil, input, 2)
		require.Error(t, err,
			"nil context accepted")
	}

	const goroutines = 16
	want, _ := rotator.Rotate([]float32{1, 2, 3, 4})
	var wait sync.WaitGroup
	wait.Add(goroutines)
	for range goroutines {
		go func() {
			defer wait.Done()
			got, err := rotator.Rotate([]float32{1, 2, 3, 4})
			assert.False(t, err != nil || !slices.Equal(got, want))
		}()
	}
	wait.Wait()
}

func TestRotationReformer(t *testing.T) {
	t.Parallel()
	rotator, _ := NewFHTRotatorFromSigns(4, []byte{1, 2, 3, 4})
	reformer, err := NewRotationReformer(rotator)
	require.NoError(t, err)

	input := []float32{1, 2, 3, 4}
	transformed, err := reformer.Transform(input)
	require.NoError(t, err)

	reverted, err := reformer.Revert(transformed)
	require.NoError(t, err)

	assertFloatSlicesClose(t, reverted, input, 1e-5)
	require.True(t, reformer.Dimension() == 4)
	{
		_, err := NewRotationReformer(nil)
		require.ErrorIs(t, err, ErrInvalidRotator)
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
		require.NoError(t, err)

		vector := make([]float32, dimension)
		for index := range vector {
			if index%2 == 0 {
				vector[index] = first
			} else {
				vector[index] = second
			}
		}
		rotated, err := rotator.Rotate(vector)
		if errors.Is(err, ailego.ErrNonFiniteVector) {
			return
		}
		require.NoError(t, err)
		reverted, err := rotator.Unrotate(rotated)
		if errors.Is(err, ailego.ErrNonFiniteVector) {
			return
		}
		require.NoError(t, err)
		inputScale := max(1, math.Abs(float64(first)), math.Abs(float64(second)))
		for index := range vector {
			tolerance := 2e-5 * inputScale
			require.InDelta(t, vector[index], reverted[index], tolerance)
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
