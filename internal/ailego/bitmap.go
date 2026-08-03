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
	"math/bits"
	"sync"
)

// Bitmap is a growable, concurrent-safe dense bitmap.
type Bitmap struct {
	mu    sync.RWMutex
	words []uint64
}

// NewBitmap returns a bitmap with capacity for bitCount bits. All bits are
// initially clear.
func NewBitmap(bitCount uint64) *Bitmap {
	wordCount := wordsForBits(bitCount)
	return &Bitmap{words: make([]uint64, wordCount)}
}

// Set sets bit and reports whether its value changed. The bitmap grows when
// necessary.
func (b *Bitmap) Set(bit uint64) bool {
	word := bitmapWordIndex(bit)
	mask := uint64(1) << (bit & 63)
	b.mu.Lock()
	defer b.mu.Unlock()
	if word >= len(b.words) {
		b.words = append(b.words, make([]uint64, word+1-len(b.words))...)
	}
	changed := b.words[word]&mask == 0
	b.words[word] |= mask
	return changed
}

// Clear clears bit and reports whether its value changed.
func (b *Bitmap) Clear(bit uint64) bool {
	word := bitmapWordIndex(bit)
	mask := uint64(1) << (bit & 63)
	b.mu.Lock()
	defer b.mu.Unlock()
	if word >= len(b.words) || b.words[word]&mask == 0 {
		return false
	}
	b.words[word] &^= mask
	return true
}

// Contains reports whether bit is set.
func (b *Bitmap) Contains(bit uint64) bool {
	word := bitmapWordIndex(bit)
	mask := uint64(1) << (bit & 63)
	b.mu.RLock()
	defer b.mu.RUnlock()
	return word < len(b.words) && b.words[word]&mask != 0
}

// Count returns the number of set bits.
func (b *Bitmap) Count() uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	var count uint64
	for _, word := range b.words {
		count += uint64(bits.OnesCount64(word))
	}
	return count
}

// Snapshot returns a copy of the bitmap words in little bit order.
func (b *Bitmap) Snapshot() []uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return append([]uint64(nil), b.words...)
}

// Clone returns an independent copy of b.
func (b *Bitmap) Clone() *Bitmap { return &Bitmap{words: b.Snapshot()} }

// Or sets every bit present in other.
func (b *Bitmap) Or(other *Bitmap) {
	if other == nil || b == other {
		return
	}
	words := other.Snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(words) > len(b.words) {
		b.words = append(b.words, make([]uint64, len(words)-len(b.words))...)
	}
	for index, word := range words {
		b.words[index] |= word
	}
}

// And retains only bits also present in other.
func (b *Bitmap) And(other *Bitmap) {
	if other == nil {
		b.mu.Lock()
		clear(b.words)
		b.mu.Unlock()
		return
	}
	if b == other {
		return
	}
	words := other.Snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	common := min(len(b.words), len(words))
	for index := 0; index < common; index++ {
		b.words[index] &= words[index]
	}
	clear(b.words[common:])
}

// AndNot clears every bit present in other.
func (b *Bitmap) AndNot(other *Bitmap) {
	if other == nil {
		return
	}
	if b == other {
		b.mu.Lock()
		clear(b.words)
		b.mu.Unlock()
		return
	}
	words := other.Snapshot()
	b.mu.Lock()
	defer b.mu.Unlock()
	for index, word := range words {
		if index >= len(b.words) {
			break
		}
		b.words[index] &^= word
	}
}

// Range calls yield for set bits in ascending order and stops when yield
// returns false. The callback runs against a snapshot and may mutate b.
func (b *Bitmap) Range(yield func(bit uint64) bool) {
	if yield == nil {
		return
	}
	for wordIndex, word := range b.Snapshot() {
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			if !yield(uint64(wordIndex)*64 + uint64(bit)) {
				return
			}
			word &= word - 1
		}
	}
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
