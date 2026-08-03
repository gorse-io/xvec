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
)

func TestBitmap(t *testing.T) {
	bitmap := NewBitmap(1)
	for _, bit := range []uint64{130, 1, 64} {
		if !bitmap.Set(bit) {
			t.Fatalf("Set(%d) did not change the bit", bit)
		}
	}
	if bitmap.Set(64) {
		t.Fatal("setting an existing bit reported a change")
	}
	if !bitmap.Contains(130) || bitmap.Contains(129) || bitmap.Count() != 3 {
		t.Fatal("bitmap membership or count is incorrect")
	}

	var got []uint64
	bitmap.Range(func(bit uint64) bool {
		got = append(got, bit)
		return true
	})
	if want := []uint64{1, 64, 130}; !slices.Equal(got, want) {
		t.Fatalf("Range = %v, want %v", got, want)
	}

	other := NewBitmap(0)
	other.Set(2)
	other.Set(64)
	bitmap.Or(other)
	if bitmap.Count() != 4 {
		t.Fatalf("count after Or = %d", bitmap.Count())
	}
	bitmap.AndNot(other)
	if bitmap.Contains(2) || bitmap.Contains(64) || bitmap.Count() != 2 {
		t.Fatal("AndNot result is incorrect")
	}
	clone := bitmap.Clone()
	clone.Set(9)
	if bitmap.Contains(9) {
		t.Fatal("Clone shares mutable storage")
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
			if !bitmap.Contains(uint64(bit)) {
				t.Errorf("bit %d was not set", bit)
			}
		}()
	}
	wg.Wait()
	if bitmap.Count() != 1000 {
		t.Fatalf("count = %d", bitmap.Count())
	}
}
