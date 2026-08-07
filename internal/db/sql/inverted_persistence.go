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

package sql

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"

	"github.com/gorse-io/zvec/internal/ailego"
)

const (
	invertedArtifactVersion    = uint16(1)
	invertedArtifactHeaderSize = 24
	maxInvertedArtifactSize    = 256 << 20
)

var invertedArtifactMagic = [8]byte{'Z', 'V', 'I', 'N', 'V', 'E', 'R', 'T'}

// ErrCorruptInvertedIndex identifies a malformed or checksummed INVERT
// snapshot. It is distinct from query and schema validation errors.
var ErrCorruptInvertedIndex = errors.New("sql: corrupt inverted index")

type persistedInvertedIndex struct {
	Field        Field                   `json:"field"`
	Rows         []uint64                `json:"rows"`
	Nulls        []uint64                `json:"nulls"`
	NonNull      []uint64                `json:"non_null"`
	Postings     []persistedPosting      `json:"postings,omitempty"`
	ArrayLengths []persistedArrayPosting `json:"array_lengths,omitempty"`
}

type persistedPosting struct {
	Kind     ValueKind `json:"kind"`
	Text     string    `json:"text,omitempty"`
	Boolean  bool      `json:"boolean,omitempty"`
	Signed   int64     `json:"signed,omitempty"`
	Unsigned uint64    `json:"unsigned,omitempty"`
	Bits     uint64    `json:"bits,omitempty"`
	Words    []uint64  `json:"words"`
}

type persistedArrayPosting struct {
	Length uint32   `json:"length"`
	Words  []uint64 `json:"words"`
}

// Encode returns a deterministic checksummed representation of a sealed
// snapshot-local inverted index.
func (i *InvertedIndex) Encode(ctx context.Context) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("sql: nil inverted-index encode context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if i == nil {
		return nil, errors.New("sql: nil inverted index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if !i.sealed {
		return nil, fmt.Errorf("sql: inverted index %q is not sealed", i.field.Name)
	}
	persisted := persistedInvertedIndex{
		Field: i.field, Rows: i.rows.Snapshot(), Nulls: i.nulls.Snapshot(), NonNull: i.nonNull.Snapshot(),
		Postings: make([]persistedPosting, 0, len(i.ordered)), ArrayLengths: make([]persistedArrayPosting, 0, len(i.lengths)),
	}
	for position, key := range i.ordered {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		persisted.Postings = append(persisted.Postings, persistedPosting{
			Kind: key.kind, Text: key.text, Boolean: key.boolean, Signed: key.signed,
			Unsigned: key.unsigned, Bits: key.bits, Words: i.postings[key].Snapshot(),
		})
	}
	for _, length := range i.lengths {
		persisted.ArrayLengths = append(persisted.ArrayLengths, persistedArrayPosting{Length: length, Words: i.arrayLength[length].Snapshot()})
	}
	payload, err := json.Marshal(persisted)
	if err != nil {
		return nil, fmt.Errorf("sql: marshal inverted index: %w", err)
	}
	if len(payload) > maxInvertedArtifactSize {
		return nil, fmt.Errorf("sql: inverted index exceeds artifact limit")
	}
	encoded := make([]byte, invertedArtifactHeaderSize+len(payload))
	copy(encoded[:8], invertedArtifactMagic[:])
	binary.LittleEndian.PutUint16(encoded[8:10], invertedArtifactVersion)
	binary.LittleEndian.PutUint16(encoded[10:12], invertedArtifactHeaderSize)
	binary.LittleEndian.PutUint64(encoded[12:20], uint64(len(payload)))
	binary.LittleEndian.PutUint32(encoded[20:24], ailego.CRC32C(payload))
	copy(encoded[invertedArtifactHeaderSize:], payload)
	return encoded, nil
}

// OpenInvertedIndex validates and opens an immutable encoded INVERT snapshot.
func OpenInvertedIndex(ctx context.Context, encoded []byte) (*InvertedIndex, error) {
	if ctx == nil {
		return nil, errors.New("sql: nil inverted-index open context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(encoded) < invertedArtifactHeaderSize || !bytes.Equal(encoded[:8], invertedArtifactMagic[:]) ||
		binary.LittleEndian.Uint16(encoded[8:10]) != invertedArtifactVersion ||
		binary.LittleEndian.Uint16(encoded[10:12]) != invertedArtifactHeaderSize {
		return nil, ErrCorruptInvertedIndex
	}
	payloadLength := binary.LittleEndian.Uint64(encoded[12:20])
	if payloadLength > maxInvertedArtifactSize || payloadLength != uint64(len(encoded)-invertedArtifactHeaderSize) {
		return nil, ErrCorruptInvertedIndex
	}
	payload := encoded[invertedArtifactHeaderSize:]
	if ailego.CRC32C(payload) != binary.LittleEndian.Uint32(encoded[20:24]) {
		return nil, ErrCorruptInvertedIndex
	}
	var persisted persistedInvertedIndex
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&persisted); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorruptInvertedIndex, err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%w: trailing JSON payload", ErrCorruptInvertedIndex)
	}
	index, err := NewInvertedIndex(persisted.Field)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorruptInvertedIndex, err)
	}
	index.rows = bitmapFromPersistedWords(persisted.Rows)
	index.nulls = bitmapFromPersistedWords(persisted.Nulls)
	index.nonNull = bitmapFromPersistedWords(persisted.NonNull)
	if !bitmapPartition(index.rows, index.nulls, index.nonNull) {
		return nil, fmt.Errorf("%w: NULL bitmaps do not partition rows", ErrCorruptInvertedIndex)
	}
	seen := make(map[scalarKey]struct{}, len(persisted.Postings))
	for position, posting := range persisted.Postings {
		if position&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		key := scalarKey{kind: posting.Kind, text: posting.Text, boolean: posting.Boolean, signed: posting.Signed, unsigned: posting.Unsigned, bits: posting.Bits}
		if key.kind != persisted.Field.Kind || !validPersistedScalarKey(key) {
			return nil, fmt.Errorf("%w: invalid posting key", ErrCorruptInvertedIndex)
		}
		if _, duplicate := seen[key]; duplicate {
			return nil, fmt.Errorf("%w: duplicate posting key", ErrCorruptInvertedIndex)
		}
		seen[key] = struct{}{}
		bitmap := bitmapFromPersistedWords(posting.Words)
		if !bitmapSubset(bitmap, index.nonNull) {
			return nil, fmt.Errorf("%w: posting references a NULL or absent row", ErrCorruptInvertedIndex)
		}
		index.postings[key] = bitmap
	}
	seenLengths := make(map[uint32]struct{}, len(persisted.ArrayLengths))
	for _, posting := range persisted.ArrayLengths {
		if !persisted.Field.Array {
			return nil, fmt.Errorf("%w: scalar index contains array lengths", ErrCorruptInvertedIndex)
		}
		if _, duplicate := seenLengths[posting.Length]; duplicate {
			return nil, fmt.Errorf("%w: duplicate array length", ErrCorruptInvertedIndex)
		}
		seenLengths[posting.Length] = struct{}{}
		bitmap := bitmapFromPersistedWords(posting.Words)
		if !bitmapSubset(bitmap, index.nonNull) {
			return nil, fmt.Errorf("%w: array length references a NULL or absent row", ErrCorruptInvertedIndex)
		}
		index.arrayLength[posting.Length] = bitmap
	}
	if err := index.Seal(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorruptInvertedIndex, err)
	}
	return index, nil
}

func validPersistedScalarKey(key scalarKey) bool {
	if !key.kind.valid() {
		return false
	}
	if key.kind == ValueFloat32 || key.kind == ValueFloat64 {
		value := math.Float64frombits(key.bits)
		return !math.IsNaN(value) && !math.IsInf(value, 0)
	}
	return true
}

func bitmapFromPersistedWords(words []uint64) *ailego.Bitmap {
	bitmap := ailego.NewBitmap(uint64(len(words)) * 64)
	for wordIndex, word := range words {
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			bitmap.Set(uint64(wordIndex*64 + bit))
			word &= word - 1
		}
	}
	return bitmap
}

func bitmapSubset(left, right *ailego.Bitmap) bool {
	leftWords, rightWords := left.Snapshot(), right.Snapshot()
	for index, word := range leftWords {
		if index >= len(rightWords) {
			if word != 0 {
				return false
			}
			continue
		}
		if word&^rightWords[index] != 0 {
			return false
		}
	}
	return true
}

func bitmapPartition(rows, nulls, nonNull *ailego.Bitmap) bool {
	rowWords, nullWords, nonNullWords := rows.Snapshot(), nulls.Snapshot(), nonNull.Snapshot()
	length := max(len(rowWords), len(nullWords), len(nonNullWords))
	for index := 0; index < length; index++ {
		var row, null, present uint64
		if index < len(rowWords) {
			row = rowWords[index]
		}
		if index < len(nullWords) {
			null = nullWords[index]
		}
		if index < len(nonNullWords) {
			present = nonNullWords[index]
		}
		if null&present != 0 || null|present != row {
			return false
		}
	}
	return true
}
