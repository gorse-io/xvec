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

package container

import (
	"runtime"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

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

func TestBitmapSnapshotAndZeroValue(t *testing.T) {
	var empty Bitmap
	require.Nil(t, empty.Snapshot())

	var bitmap Bitmap
	require.True(t, bitmap.Set(65))
	require.Equal(t, []uint64{0, 2}, bitmap.Snapshot())
	bitmap.Set(130)
	bitmap.Clear(130)
	require.Equal(t, []uint64{0, 2, 0}, bitmap.Snapshot())
	require.Equal(t, bitmap.Snapshot(), bitmap.Clone().Snapshot())

	preallocated := NewBitmap(130)
	preallocated.Set(0)
	require.Equal(t, []uint64{1, 0, 0}, preallocated.Snapshot())

	grown := NewBitmap(0)
	grown.Set(200)
	grown.Clear(200)
	preallocated.Or(grown)
	require.Equal(t, []uint64{1, 0, 0, 0}, preallocated.Snapshot())
}

func TestBitmapSnapshotWithin(t *testing.T) {
	bitmap := NewBitmap(0)
	bitmap.Set(1 << 26)
	words, ok := bitmap.SnapshotWithin(64)
	require.False(t, ok)
	require.Nil(t, words)

	bitmap.Clear(1 << 26)
	words, ok = bitmap.SnapshotWithin(64)
	require.True(t, ok)
	require.Equal(t, []uint64{0}, words)

	bitmap.Set(1)
	bitmap.Set(65)
	words, ok = bitmap.SnapshotWithin(66)
	require.True(t, ok)
	require.Equal(t, []uint64{2, 2}, words)
}

func TestBitmapAddressableBoundary(t *testing.T) {
	if strconv.IntSize != 32 {
		t.Skip("32-bit addressability boundary")
	}
	bit := uint64(maxInt()) << 6
	bitmap := NewBitmap(0)
	require.Panics(t, func() { bitmap.Set(bit) })
	require.Panics(t, func() { bitmap.Clear(bit) })
	require.Panics(t, func() { bitmap.Contains(bit) })
}

func TestBitmapRangeUsesSnapshot(t *testing.T) {
	bitmap := NewBitmap(0)
	bitmap.Set(1)
	bitmap.Set(2)

	var got []uint64
	bitmap.Range(func(bit uint64) bool {
		got = append(got, bit)
		bitmap.Clear(bit)
		bitmap.Set(bit + 10)
		return true
	})

	require.Equal(t, []uint64{1, 2}, got)
	var remaining []uint64
	bitmap.Range(func(bit uint64) bool {
		remaining = append(remaining, bit)
		return true
	})
	require.Equal(t, []uint64{11, 12}, remaining)
}

func TestBitmapSparseHighBitUsesBoundedMemory(t *testing.T) {
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	var bitmap Bitmap
	require.True(t, bitmap.Set(1<<26))

	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	require.Less(t, after.TotalAlloc-before.TotalAlloc, uint64(1<<20),
		"setting one sparse high bit allocated dense storage")
	require.True(t, bitmap.Contains(1<<26))
}

func TestBitmapReciprocalOperationsDoNotDeadlock(t *testing.T) {
	left, right := NewBitmap(0), NewBitmap(0)
	left.Set(1)
	right.Set(2)

	var wg sync.WaitGroup
	wg.Add(2)
	for _, operation := range []func(){
		func() {
			defer wg.Done()
			for range 1000 {
				left.Or(right)
			}
		},
		func() {
			defer wg.Done()
			for range 1000 {
				right.Or(left)
			}
		},
	} {
		go operation()
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		require.Equal(t, uint64(2), left.Count())
		require.Equal(t, uint64(2), right.Count())
	case <-time.After(5 * time.Second):
		t.Fatal("reciprocal bitmap operations deadlocked")
	}
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
