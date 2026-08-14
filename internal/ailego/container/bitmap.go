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
	"sync"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// Bitmap is a growable, concurrent-safe compressed bitmap.
type Bitmap struct {
	mu           sync.RWMutex
	bitmap       roaring64.Bitmap
	logicalWords int
}

// NewBitmap returns a bitmap with a logical capacity for bitCount bits. All
// bits are initially clear. Storage remains sparse until bits are set.
func NewBitmap(bitCount uint64) *Bitmap {
	return &Bitmap{logicalWords: wordsForBits(bitCount)}
}

// Set sets bit and reports whether its value changed.
func (b *Bitmap) Set(bit uint64) bool {
	word := bitmapWordIndex(bit)
	b.mu.Lock()
	defer b.mu.Unlock()
	b.logicalWords = max(b.logicalWords, word+1)
	return b.bitmap.CheckedAdd(bit)
}

// Clear clears bit and reports whether its value changed.
func (b *Bitmap) Clear(bit uint64) bool {
	bitmapWordIndex(bit)
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.bitmap.CheckedRemove(bit)
}

// Contains reports whether bit is set.
func (b *Bitmap) Contains(bit uint64) bool {
	bitmapWordIndex(bit)
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.bitmap.Contains(bit)
}

// Count returns the number of set bits.
func (b *Bitmap) Count() uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.bitmap.GetCardinality()
}

// Snapshot returns a dense copy of the bitmap words in little bit order.
// Its memory use is proportional to the highest bit ever set or NewBitmap's
// logical capacity; sparse callers should prefer Clone or Range.
func (b *Bitmap) Snapshot() []uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.snapshotWords(b.logicalWords)
}

// SnapshotWithin returns a dense snapshot bounded to bitCount bits. It reports
// false without allocating the dense snapshot when a set bit is outside the
// requested domain.
func (b *Bitmap) SnapshotWithin(bitCount uint64) ([]uint64, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if !b.bitmap.IsEmpty() && (bitCount == 0 || b.bitmap.Maximum() >= bitCount) {
		return nil, false
	}
	return b.snapshotWords(min(b.logicalWords, wordsForBits(bitCount))), true
}

func (b *Bitmap) snapshotWords(wordCount int) []uint64 {
	if wordCount == 0 {
		return nil
	}
	words := make([]uint64, wordCount)
	iterator := b.bitmap.Iterator()
	for iterator.HasNext() {
		bit := iterator.Next()
		words[bitmapWordIndex(bit)] |= uint64(1) << (bit & 63)
	}
	return words
}

// Clone returns an independent copy of b.
func (b *Bitmap) Clone() *Bitmap {
	bitmap, logicalWords := b.snapshot()
	return &Bitmap{bitmap: *bitmap, logicalWords: logicalWords}
}

// Or sets every bit present in other.
func (b *Bitmap) Or(other *Bitmap) {
	if other == nil || b == other {
		return
	}
	bitmap, logicalWords := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.bitmap.Or(bitmap)
	b.logicalWords = max(b.logicalWords, logicalWords)
}

// And retains only bits also present in other.
func (b *Bitmap) And(other *Bitmap) {
	if other == nil {
		b.mu.Lock()
		b.bitmap.Clear()
		b.mu.Unlock()
		return
	}
	if b == other {
		return
	}
	bitmap, _ := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.bitmap.And(bitmap)
}

// AndNot clears every bit present in other.
func (b *Bitmap) AndNot(other *Bitmap) {
	if other == nil {
		return
	}
	if b == other {
		b.mu.Lock()
		b.bitmap.Clear()
		b.mu.Unlock()
		return
	}
	bitmap, _ := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.bitmap.AndNot(bitmap)
}

// Range calls yield for set bits in ascending order and stops when yield
// returns false. The callback runs against a snapshot and may mutate b.
func (b *Bitmap) Range(yield func(bit uint64) bool) {
	if yield == nil {
		return
	}
	bitmap, _ := b.snapshot()
	iterator := bitmap.Iterator()
	for iterator.HasNext() {
		if !yield(iterator.Next()) {
			return
		}
	}
}

func (b *Bitmap) snapshot() (*roaring64.Bitmap, int) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.bitmap.Clone(), b.logicalWords
}

func wordsForBits(bitCount uint64) int {
	if bitCount == 0 {
		return 0
	}
	wordCount := (bitCount-1)/64 + 1
	if wordCount > uint64(maxInt()) {
		panic("ailego: bitmap exceeds addressable memory")
	}
	return int(wordCount)
}

func bitmapWordIndex(bit uint64) int {
	word := bit >> 6
	if word >= uint64(maxInt()) {
		panic("ailego: bitmap index exceeds addressable memory")
	}
	return int(word)
}

func maxInt() int { return int(^uint(0) >> 1) }
