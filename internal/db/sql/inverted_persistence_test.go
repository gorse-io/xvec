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
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInvertedIndexPersistenceRoundTripAndCorruption(t *testing.T) {
	field := Field{Name: "tags", Kind: ValueString, Array: true, Nullable: true, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field,
		mustNullValue(t, ValueString, true),
		mustArray(t, ValueString),
		mustArray(t, ValueString, StringValue("a"), StringValue("b")),
		mustArray(t, ValueString, StringValue("b"), StringValue("c")),
	)
	encoded, err := index.Encode(context.Background())
	require.NoError(t, err)

	reopened, err := OpenInvertedIndex(context.Background(), encoded)
	require.NoError(t, err)
	require.Equal(t, field, reopened.Field())
	require.Equal(t, uint64(4), reopened.RowCount())
	predicate, err := NewSetPredicate(PredicateContainAny, false, []Value{StringValue("a"), StringValue("c")})
	require.NoError(t, err)
	result, err := reopened.Search(predicate)
	require.NoError(t, err)
	require.Equal(t, []uint64{2, 3}, bitmapBits(result.Bitmap))
	length, err := NewComparisonPredicate(PredicateEQ, Uint32Value(0))
	require.NoError(t, err)
	result, err = reopened.SearchArrayLength(length)
	require.NoError(t, err)
	require.Equal(t, []uint64{1}, bitmapBits(result.Bitmap))

	corrupt := append([]byte(nil), encoded...)
	corrupt[len(corrupt)-1] ^= 0xff
	_, err = OpenInvertedIndex(context.Background(), corrupt)
	require.ErrorIs(t, err, ErrCorruptInvertedIndex)
}

func FuzzOpenInvertedIndex(f *testing.F) {
	index, err := NewInvertedIndex(Field{Name: "value", Kind: ValueInt64, Filterable: true, Indexed: true, RangeOptimized: true})
	if err != nil {
		f.Fatal(err)
	}
	if err := index.Add(0, Int64Value(42)); err != nil {
		f.Fatal(err)
	}
	if err := index.Seal(); err != nil {
		f.Fatal(err)
	}
	encoded, err := index.Encode(context.Background())
	if err != nil {
		f.Fatal(err)
	}
	f.Add(encoded)
	f.Add([]byte("not an inverted index"))
	f.Fuzz(func(t *testing.T, data []byte) {
		reopened, err := OpenInvertedIndex(context.Background(), data)
		if err == nil {
			reencoded, encodeErr := reopened.Encode(context.Background())
			require.NoError(t, encodeErr)
			require.NotEmpty(t, reencoded)
		}
	})
}
