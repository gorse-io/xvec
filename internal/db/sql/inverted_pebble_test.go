// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
package sql

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInvertedIndexPebbleRoundTripChunksAndCorruption(t *testing.T) {
	field := Field{Name: "tags", Kind: ValueString, Array: true, Nullable: true, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field,
		mustNullValue(t, ValueString, true),
		mustArray(t, ValueString),
		mustArray(t, ValueString, StringValue("a"), StringValue("b")),
		mustArray(t, ValueString, StringValue("b"), StringValue("c")),
	)
	path := filepath.Join(t.TempDir(), "invert.pebble")
	require.NoError(t, index.Save(context.Background(), path))
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.True(t, info.IsDir())

	reopened, err := OpenInvertedIndex(context.Background(), path)
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

	current := filepath.Join(path, "ZVEC-INDEX")
	require.NoError(t, os.WriteFile(current, []byte("corrupt\n"), 0o600))
	_, err = OpenInvertedIndex(context.Background(), path)
	require.ErrorIs(t, err, ErrCorruptInvertedIndex)
}

func TestInvertedIndexPebbleUsesMultipleOrderedPostingKeys(t *testing.T) {
	field := Field{Name: "value", Kind: ValueInt64, Filterable: true, Indexed: true, RangeOptimized: true}
	index, err := NewInvertedIndex(field)
	require.NoError(t, err)
	for row := 0; row < 70_000; row++ {
		require.NoError(t, index.Add(uint64(row), Int64Value(int64(row%2))))
	}
	require.NoError(t, index.Seal())
	path := filepath.Join(t.TempDir(), "chunked.pebble")
	require.NoError(t, index.Save(context.Background(), path))
	keys, err := inspectInvertedStoreKeys(path)
	require.NoError(t, err)
	var postingKeys int
	for _, key := range keys {
		if len(key) > 0 && key[0] == 'p' {
			postingKeys++
		}
	}
	require.Greater(t, postingKeys, len(index.ordered), "postings were stored as one value per term")
}
