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
