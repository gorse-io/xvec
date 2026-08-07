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
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBitmap(t *testing.T) {
	bitmap := NewBitmap(1)
	for _, bit := range []uint64{130, 1, 64} {
		require.True(t, bitmap.Set(bit))
	}
	require.False(t, bitmap.Set(64),
		"setting an existing bit reported a change")
	require.True(t, bitmap.Contains(130),
		"bitmap membership or count is incorrect")
	require.False(t, bitmap.Contains(129),
		"bitmap membership or count is incorrect")
	require.True(t, bitmap.Count() == 3,
		"bitmap membership or count is incorrect")

	var got []uint64
	bitmap.Range(func(bit uint64) bool {
		got = append(got, bit)
		return true
	})
	{
		want := []uint64{1, 64, 130}
		require.True(t, slices.Equal(got, want))
	}

	other := NewBitmap(0)
	other.Set(2)
	other.Set(64)
	bitmap.Or(other)
	require.True(t, bitmap.Count() == 4)

	bitmap.AndNot(other)
	require.False(t, bitmap.Contains(2),
		"AndNot result is incorrect")
	require.False(t, bitmap.Contains(64),
		"AndNot result is incorrect")
	require.True(t, bitmap.Count() == 2,
		"AndNot result is incorrect")

	clone := bitmap.Clone()
	clone.Set(9)
	require.False(t, bitmap.Contains(9),
		"Clone shares mutable storage")

	clone.And(bitmap)
	require.False(t, clone.Contains(9),
		"And result is incorrect")
	require.Equal(t, bitmap.Count(), clone.Count(),
		"And result is incorrect")

	clone.And(nil)
	require.True(t, clone.Count() == 0,
		"And(nil) did not clear the bitmap")
}

func TestBitmapSnapshot(t *testing.T) {
	bitmap := NewBitmap(0)
	bitmap.Set(1)
	bitmap.Set(64)
	require.Equal(t, []uint64{2, 1}, bitmap.Snapshot())
}

func TestBitmapSupportsSparseHighBits(t *testing.T) {
	bitmap := NewBitmap(0)
	const highBit = uint64(1) << 40

	require.True(t, bitmap.Set(highBit))
	require.True(t, bitmap.Contains(highBit))
	require.Equal(t, uint64(1), bitmap.Count())

	var got []uint64
	bitmap.Range(func(bit uint64) bool {
		got = append(got, bit)
		return true
	})
	require.Equal(t, []uint64{highBit}, got)
}

func TestBitmapZeroValue(t *testing.T) {
	var bitmap Bitmap
	require.False(t, bitmap.Contains(1))
	require.False(t, bitmap.Clear(1))
	require.Zero(t, bitmap.Count())
	require.True(t, bitmap.Set(1))
	require.True(t, bitmap.Contains(1))
}

func TestBitmapRangeMayMutateReceiver(t *testing.T) {
	bitmap := NewBitmap(0)
	bitmap.Set(1)
	bitmap.Set(2)

	var visited []uint64
	bitmap.Range(func(bit uint64) bool {
		visited = append(visited, bit)
		bitmap.Clear(bit)
		bitmap.Set(bit + 10)
		return true
	})

	require.Equal(t, []uint64{1, 2}, visited)
	require.False(t, bitmap.Contains(1))
	require.False(t, bitmap.Contains(2))
	require.True(t, bitmap.Contains(11))
	require.True(t, bitmap.Contains(12))
}

func TestBitmapConcurrentBinaryOperations(t *testing.T) {
	left := NewBitmap(0)
	right := NewBitmap(0)
	for bit := range uint64(100) {
		left.Set(bit)
		right.Set(bit + 50)
	}

	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for range 100 {
			left.Or(right)
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		for range 100 {
			right.And(left)
		}
	}()
	close(start)
	wg.Wait()

	require.GreaterOrEqual(t, left.Count(), uint64(100))
	require.GreaterOrEqual(t, right.Count(), uint64(50))
	right.Range(func(bit uint64) bool {
		require.True(t, left.Contains(bit))
		return true
	})
}

func TestBitmapConcurrentAccess(t *testing.T) {
	bitmap := NewBitmap(0)
	var wg sync.WaitGroup
	for bit := range 1000 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bitmap.Set(uint64(bit))
			assert.True(t, bitmap.Contains(uint64(bit)))
		}()
	}
	wg.Wait()
	require.True(t, bitmap.Count() == 1000)
}
