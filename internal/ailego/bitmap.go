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
	"sync"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// Bitmap is a growable, concurrent-safe compressed bitmap.
type Bitmap struct {
	mu   sync.RWMutex
	bits *roaring64.Bitmap
}

// NewBitmap returns an empty bitmap. bitCount is accepted for API
// compatibility; compressed bitmaps allocate containers on demand.
func NewBitmap(bitCount uint64) *Bitmap {
	_ = bitCount
	return &Bitmap{bits: roaring64.NewBitmap()}
}

// Set sets bit and reports whether its value changed.
func (b *Bitmap) Set(bit uint64) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.bits == nil {
		b.bits = roaring64.NewBitmap()
	}
	return b.bits.CheckedAdd(bit)
}

// Clear clears bit and reports whether its value changed.
func (b *Bitmap) Clear(bit uint64) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.bits == nil {
		return false
	}
	return b.bits.CheckedRemove(bit)
}

// Contains reports whether bit is set.
func (b *Bitmap) Contains(bit uint64) bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.bits != nil && b.bits.Contains(bit)
}

// Count returns the number of set bits.
func (b *Bitmap) Count() uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.bits == nil {
		return 0
	}
	return b.bits.GetCardinality()
}

// Snapshot returns a dense copy of the bitmap words in little bit order. Its
// memory use is proportional to the highest set bit, so callers with extremely
// sparse high positions should prefer Clone or Range.
func (b *Bitmap) Snapshot() []uint64 {
	clone := b.snapshot()
	if clone.IsEmpty() {
		return nil
	}
	iterator := clone.Iterator()
	var words []uint64
	for iterator.HasNext() {
		bit := iterator.Next()
		word := bit >> 6
		if word >= uint64(maxInt()) {
			panic("ailego: bitmap index exceeds addressable memory")
		}
		if int(word) >= len(words) {
			words = append(words, make([]uint64, int(word)+1-len(words))...)
		}
		words[word] |= uint64(1) << (bit & 63)
	}
	return words
}

// Clone returns an independent copy of b.
func (b *Bitmap) Clone() *Bitmap { return &Bitmap{bits: b.snapshot()} }

// Or sets every bit present in other.
func (b *Bitmap) Or(other *Bitmap) {
	if other == nil || b == other {
		return
	}
	otherBits := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.bits == nil {
		b.bits = roaring64.NewBitmap()
	}
	b.bits.Or(otherBits)
}

// And retains only bits also present in other.
func (b *Bitmap) And(other *Bitmap) {
	if other == nil {
		b.mu.Lock()
		if b.bits != nil {
			b.bits.Clear()
		}
		b.mu.Unlock()
		return
	}
	if b == other {
		return
	}
	otherBits := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.bits == nil {
		return
	}
	b.bits.And(otherBits)
}

// AndNot clears every bit present in other.
func (b *Bitmap) AndNot(other *Bitmap) {
	if other == nil {
		return
	}
	if b == other {
		b.mu.Lock()
		if b.bits != nil {
			b.bits.Clear()
		}
		b.mu.Unlock()
		return
	}
	otherBits := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.bits == nil {
		return
	}
	b.bits.AndNot(otherBits)
}

// Range calls yield for set bits in ascending order and stops when yield
// returns false. The callback runs against a snapshot and may mutate b.
func (b *Bitmap) Range(yield func(bit uint64) bool) {
	if yield == nil {
		return
	}
	iterator := b.snapshot().Iterator()
	for iterator.HasNext() {
		if !yield(iterator.Next()) {
			return
		}
	}
}

func (b *Bitmap) snapshot() *roaring64.Bitmap {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.bits == nil {
		return roaring64.NewBitmap()
	}
	return b.bits.Clone()
}
