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
	"math/bits"
	"sync"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// Bitmap is a growable, concurrent-safe compressed bitmap.
type Bitmap struct {
	mu           sync.RWMutex
	bitmap       roaring64.Bitmap
	logicalWords int
}

// FrozenBitmap is an immutable snapshot. Readers need neither locks nor
// iterator snapshots. It never shares mutable storage with its source.
type FrozenBitmap struct {
	bitmap       roaring64.Bitmap
	logicalWords int
}

// Freeze copies b once for immutable publication.
func (b *Bitmap) Freeze() *FrozenBitmap {
	bitmap, logicalWords := b.snapshot()
	return &FrozenBitmap{bitmap: *bitmap, logicalWords: logicalWords}
}

// Contains reports whether bit is set in the immutable snapshot.
func (b *FrozenBitmap) Contains(bit uint64) bool {
	bitmapWordIndex(bit)
	return b.bitmap.Contains(bit)
}

// Count returns the snapshot's number of set bits.
func (b *FrozenBitmap) Count() uint64 { return b.bitmap.GetCardinality() }

// Within reports whether all set bits are below bitCount, without enumerating.
func (b *FrozenBitmap) Within(bitCount uint64) bool {
	return b.bitmap.IsEmpty() || (bitCount > 0 && b.bitmap.Maximum() < bitCount)
}

// Range visits immutable set bits in ascending order, stopping on false.
func (b *FrozenBitmap) Range(yield func(uint64) bool) {
	if yield == nil {
		return
	}
	iterator := b.bitmap.Iterator()
	for iterator.HasNext() {
		if !yield(iterator.Next()) {
			return
		}
	}
}

// OrFrozen merges a published snapshot without cloning the source. Roaring's
// union copies containers that are inserted into the mutable destination.
func (b *Bitmap) OrFrozen(other *FrozenBitmap) {
	if other == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.bitmap.Or(&other.bitmap)
	b.logicalWords = max(b.logicalWords, other.logicalWords)
}

// NewBitmap returns a bitmap with a logical capacity for bitCount bits. All
// bits are initially clear. Storage remains sparse until bits are set.
func NewBitmap(bitCount uint64) *Bitmap {
	return &Bitmap{logicalWords: wordsForBits(bitCount)}
}

// AppendWords appends dense words in little bit order to the bitmap's logical
// capacity. Clear words retain their capacity without allocating dense storage.
// The input is copied into the compressed representation and is not retained.
func (b *Bitmap) AppendWords(words []uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(words) > maxInt()-b.logicalWords || uint64(b.logicalWords)+uint64(len(words)) > (^uint64(0)>>6)+1 {
		panic("ailego: bitmap exceeds addressable memory")
	}
	for offset, word := range words {
		base := uint64(b.logicalWords+offset) * 64
		for word != 0 {
			b.bitmap.Add(base + uint64(bits.TrailingZeros64(word)))
			word &= word - 1
		}
	}
	b.logicalWords += len(words)
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
