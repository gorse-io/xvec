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

// bitmapContents keeps up to two ordered IDs inline. Higher cardinalities use
// Roaring; empty and singleton postings never allocate its container hierarchy.
// Contents are protected by Bitmap.mu or owned by an immutable snapshot.
type bitmapContents struct {
	large *roaring64.Bitmap
	small [2]uint64
	size  uint8
}

func (c *bitmapContents) contains(bit uint64) bool {
	if c.large != nil {
		return c.large.Contains(bit)
	}
	for _, v := range c.small[:c.size] {
		if v == bit {
			return true
		}
	}
	return false
}
func (c *bitmapContents) count() uint64 {
	if c.large != nil {
		return c.large.GetCardinality()
	}
	return uint64(c.size)
}
func (c *bitmapContents) within(bitCount uint64) bool {
	if c.large != nil {
		return c.large.IsEmpty() || (bitCount > 0 && c.large.Maximum() < bitCount)
	}
	return c.size == 0 || c.small[c.size-1] < bitCount
}
func (c *bitmapContents) promote() {
	if c.large == nil {
		c.large = roaring64.NewBitmap()
		c.large.AddMany(c.small[:c.size])
		c.small = [2]uint64{}
		c.size = 0
	}
}
func (c *bitmapContents) add(bit uint64) bool {
	if c.large != nil {
		return c.large.CheckedAdd(bit)
	}
	if c.contains(bit) {
		return false
	}
	if c.size == uint8(len(c.small)) {
		c.promote()
		return c.large.CheckedAdd(bit)
	}
	pos := int(c.size)
	for pos > 0 && c.small[pos-1] > bit {
		c.small[pos] = c.small[pos-1]
		pos--
	}
	c.small[pos] = bit
	c.size++
	return true
}
func (c *bitmapContents) remove(bit uint64) bool {
	if c.large != nil {
		return c.large.CheckedRemove(bit)
	}
	for pos, v := range c.small[:c.size] {
		if v == bit {
			copy(c.small[pos:], c.small[pos+1:c.size])
			c.size--
			c.small[c.size] = 0
			return true
		}
	}
	return false
}
func (c *bitmapContents) clone() bitmapContents {
	result := *c
	if c.large != nil {
		result.large = c.large.Clone()
	}
	return result
}
func (c *bitmapContents) rangeBits(yield func(uint64) bool) {
	if c.large == nil {
		for _, v := range c.small[:c.size] {
			if !yield(v) {
				return
			}
		}
		return
	}
	iterator := c.large.Iterator()
	for iterator.HasNext() {
		if !yield(iterator.Next()) {
			return
		}
	}
}
func (c *bitmapContents) or(other *bitmapContents) {
	if other.large == nil {
		for _, v := range other.small[:other.size] {
			c.add(v)
		}
		return
	}
	if other.large.IsEmpty() {
		return
	}
	c.promote()
	c.large.Or(other.large)
}
func (c *bitmapContents) and(other *bitmapContents) {
	if c.large != nil && other.large != nil {
		c.large.And(other.large)
		return
	}
	var result bitmapContents
	if c.large == nil {
		for _, v := range c.small[:c.size] {
			if other.contains(v) {
				result.add(v)
			}
		}
	} else {
		for _, v := range other.small[:other.size] {
			if c.contains(v) {
				result.add(v)
			}
		}
	}
	*c = result
}
func (c *bitmapContents) andNot(other *bitmapContents) {
	if c.large != nil && other.large != nil {
		c.large.AndNot(other.large)
		return
	}
	if other.large == nil {
		for _, v := range other.small[:other.size] {
			c.remove(v)
		}
		return
	}
	var result bitmapContents
	for _, v := range c.small[:c.size] {
		if !other.contains(v) {
			result.add(v)
		}
	}
	*c = result
}

// Bitmap is a growable, concurrent-safe compressed bitmap.
type Bitmap struct {
	mu           sync.RWMutex
	contents     bitmapContents
	logicalWords int
}

// FrozenBitmap is an immutable snapshot with independently owned storage.
type FrozenBitmap struct {
	contents     bitmapContents
	logicalWords int
}

// Freeze copies b once for immutable publication.
func (b *Bitmap) Freeze() *FrozenBitmap {
	contents, logicalWords := b.snapshot()
	return &FrozenBitmap{contents: contents, logicalWords: logicalWords}
}
func (b *FrozenBitmap) Contains(bit uint64) bool {
	bitmapWordIndex(bit)
	return b.contents.contains(bit)
}
func (b *FrozenBitmap) Count() uint64               { return b.contents.count() }
func (b *FrozenBitmap) Within(bitCount uint64) bool { return b.contents.within(bitCount) }
func (b *FrozenBitmap) Range(yield func(uint64) bool) {
	if yield != nil {
		b.contents.rangeBits(yield)
	}
}

// OrFrozen merges a snapshot without cloning its immutable source. Roaring's
// union copies containers inserted into the mutable destination.
func (b *Bitmap) OrFrozen(other *FrozenBitmap) {
	if other == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.contents.or(&other.contents)
	b.logicalWords = max(b.logicalWords, other.logicalWords)
}
func NewBitmap(bitCount uint64) *Bitmap { return &Bitmap{logicalWords: wordsForBits(bitCount)} }

// AppendWords appends dense words without retaining the input or allocating
// storage for trailing clear words.
func (b *Bitmap) AppendWords(words []uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(words) > maxInt()-b.logicalWords || uint64(b.logicalWords)+uint64(len(words)) > (^uint64(0)>>6)+1 {
		panic("ailego: bitmap exceeds addressable memory")
	}
	for offset, word := range words {
		base := uint64(b.logicalWords+offset) * 64
		for word != 0 {
			b.contents.add(base + uint64(bits.TrailingZeros64(word)))
			word &= word - 1
		}
	}
	b.logicalWords += len(words)
}
func (b *Bitmap) Set(bit uint64) bool {
	word := bitmapWordIndex(bit)
	b.mu.Lock()
	defer b.mu.Unlock()
	b.logicalWords = max(b.logicalWords, word+1)
	return b.contents.add(bit)
}
func (b *Bitmap) Clear(bit uint64) bool {
	bitmapWordIndex(bit)
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.contents.remove(bit)
}
func (b *Bitmap) Contains(bit uint64) bool {
	bitmapWordIndex(bit)
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.contents.contains(bit)
}
func (b *Bitmap) Count() uint64 { b.mu.RLock(); defer b.mu.RUnlock(); return b.contents.count() }

// Snapshot returns a dense copy including the bitmap's logical clear capacity.
// Sparse callers should prefer Clone or Range.
func (b *Bitmap) Snapshot() []uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.snapshotWords(b.logicalWords)
}
func (b *Bitmap) SnapshotWithin(bitCount uint64) ([]uint64, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if !b.contents.within(bitCount) {
		return nil, false
	}
	return b.snapshotWords(min(b.logicalWords, wordsForBits(bitCount))), true
}
func (b *Bitmap) snapshotWords(wordCount int) []uint64 {
	if wordCount == 0 {
		return nil
	}
	words := make([]uint64, wordCount)
	b.contents.rangeBits(func(bit uint64) bool { words[bitmapWordIndex(bit)] |= uint64(1) << (bit & 63); return true })
	return words
}
func (b *Bitmap) Clone() *Bitmap {
	contents, logicalWords := b.snapshot()
	return &Bitmap{contents: contents, logicalWords: logicalWords}
}
func (b *Bitmap) Or(other *Bitmap) {
	if other == nil || b == other {
		return
	}
	contents, logicalWords := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.contents.or(&contents)
	b.logicalWords = max(b.logicalWords, logicalWords)
}
func (b *Bitmap) And(other *Bitmap) {
	if b == other {
		return
	}
	var contents bitmapContents
	if other != nil {
		contents, _ = other.snapshot()
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.contents.and(&contents)
}
func (b *Bitmap) AndNot(other *Bitmap) {
	if other == nil {
		return
	}
	if b == other {
		b.mu.Lock()
		b.contents = bitmapContents{}
		b.mu.Unlock()
		return
	}
	contents, _ := other.snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.contents.andNot(&contents)
}

// Range visits an independent snapshot; callbacks may mutate b.
func (b *Bitmap) Range(yield func(uint64) bool) {
	if yield == nil {
		return
	}
	contents, _ := b.snapshot()
	contents.rangeBits(yield)
}
func (b *Bitmap) snapshot() (bitmapContents, int) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.contents.clone(), b.logicalWords
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
