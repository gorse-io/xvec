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

package sqlengine

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"sync"
	"testing"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInvertedIntegerRangeBlocks(t *testing.T) {
	for _, kind := range []struct {
		name  string
		kind  ValueKind
		value func(int) Value
	}{
		{"int32", ValueInt32, func(v int) Value { return Int32Value(int32(v - 1024)) }},
		{"int64", ValueInt64, func(v int) Value { return Int64Value(math.MinInt64 + 4096 + int64(v)) }},
		{"uint32", ValueUint32, func(v int) Value { return Uint32Value(uint32(math.MaxUint32 - 4096 + int64(v))) }},
		{"uint64", ValueUint64, func(v int) Value { return Uint64Value(uint64(math.MaxUint64-4096) + uint64(v)) }},
	} {
		for _, terms := range []int{0, 255, 256, 257, 769} {
			t.Run(fmt.Sprintf("%s/%d", kind.name, terms), func(t *testing.T) {
				field := Field{Name: "id", Kind: kind.kind, Nullable: true, Filterable: true, Indexed: true, RangeOptimized: true}
				index, err := NewInvertedIndex(field)
				require.NoError(t, err)
				type rowValue struct {
					row   uint64
					value Value
				}
				rows := []rowValue{{0, mustNullValue(t, kind.kind, false)}}
				for term := 0; term < terms; term++ {
					// Sparse row ordinals run in the opposite order from values.
					row := uint64((terms - term) * 3)
					rows = append(rows, rowValue{row, kind.value(term * 2)})
					if term%7 == 0 {
						rows = append(rows, rowValue{row + 1, kind.value(term * 2)})
					}
				}
				for _, row := range rows {
					require.NoError(t, index.Add(row.row, row.value))
				}
				require.NoError(t, index.Seal())
				require.NoError(t, index.Seal())
				require.Len(t, index.rangeBlocks, terms/invertedRangeBlockSize)

				check := func(t *testing.T, index *InvertedIndex) {
					t.Helper()
					for _, op := range []PredicateOperator{PredicateLT, PredicateLE, PredicateGT, PredicateGE} {
						// Exercise inclusive/exclusive bounds, absent terms, block
						// boundaries, empty results and all non-NULL rows.
						for _, target := range []int{-1, 0, 1, 509, 510, 511, 512, 513, 514, 1023, 1024, terms * 2} {
							predicate, err := NewComparisonPredicate(op, kind.value(target))
							require.NoError(t, err)
							result, err := index.Search(predicate)
							require.NoError(t, err)
							want := container.NewBitmap(0)
							distinct := make(map[scalarKey]struct{})
							for _, row := range rows {
								truth, err := predicate.Evaluate(row.value)
								require.NoError(t, err)
								if truth.Match() {
									want.Set(row.row)
									distinct[keyFromValue(row.value)] = struct{}{}
								}
							}
							require.True(t, result.Supported)
							require.Equal(t, InvertedSortedRange, result.Strategy)
							require.Equal(t, len(distinct), result.Terms)
							require.Equal(t, bitmapBits(want), bitmapBits(result.Bitmap), "op=%v target=%d", op, target)
							// A caller may mutate its result; cached aggregates and
							// postings must remain intact for the next query.
							result.Bitmap.And(nil)
							result.Bitmap.Set(0)
							again, err := index.Search(predicate)
							require.NoError(t, err)
							require.Equal(t, bitmapBits(want), bitmapBits(again.Bitmap))
						}
					}
				}
				check(t, index)
				path := filepath.Join(t.TempDir(), "range.pebble")
				require.NoError(t, index.Save(context.Background(), path))
				reopened, err := OpenInvertedIndex(context.Background(), path)
				require.NoError(t, err)
				require.Len(t, reopened.rangeBlocks, terms/invertedRangeBlockSize)
				check(t, reopened)
			})
		}
	}
}

func TestInvertedIntegerRangeBlocksConcurrent(t *testing.T) {
	field := Field{Name: "id", Kind: ValueInt64, Filterable: true, Indexed: true, RangeOptimized: true}
	values := make([]Value, 2048)
	for row := range values {
		values[row] = Int64Value(int64(row))
	}
	index := mustInvertedIndex(t, field, values...)
	var workers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for iteration := 0; iteration < 32; iteration++ {
				threshold := int64(256 + iteration*17)
				predicate, err := NewComparisonPredicate(PredicateGE, Int64Value(threshold))
				if !assert.NoError(t, err) {
					return
				}
				result, err := index.Search(predicate)
				if !assert.NoError(t, err) {
					return
				}
				assert.Equal(t, uint64(len(values))-uint64(threshold), result.Bitmap.Count())
				assert.True(t, result.Bitmap.Contains(uint64(threshold)))
				assert.False(t, result.Bitmap.Contains(uint64(threshold-1)))
				result.Bitmap.And(nil)
			}
		}()
	}
	workers.Wait()
}

func TestInvertedRangeBlocksDisabled(t *testing.T) {
	for _, field := range []Field{
		{Name: "id", Kind: ValueInt64, Filterable: true, Indexed: true},
		{Name: "id", Kind: ValueString, Filterable: true, Indexed: true, RangeOptimized: true},
		{Name: "id", Kind: ValueInt64, Array: true, Filterable: true, Indexed: true, RangeOptimized: true},
	} {
		index, err := NewInvertedIndex(field)
		require.NoError(t, err)
		for row := 0; row < 512; row++ {
			value := Int64Value(int64(row))
			if field.Kind == ValueString {
				value = StringValue(fmt.Sprintf("%04d", row))
			}
			if field.Array {
				value = mustArray(t, ValueInt64, value)
			}
			require.NoError(t, index.Add(uint64(row), value))
		}
		require.NoError(t, index.Seal())
		require.Empty(t, index.rangeBlocks)
	}
}

func TestInvertedBitmapSubsetSparse(t *testing.T) {
	left, right := container.NewBitmap(0), container.NewBitmap(0)
	require.True(t, bitmapSubset(left, right))
	right.Set(1 << 40)
	left.Set(1 << 40)
	require.True(t, bitmapSubset(left, right))
	left.Set((1 << 40) + 7)
	require.False(t, bitmapSubset(left, right))
	require.Equal(t, uint64(2), left.Count())
	require.Equal(t, uint64(1), right.Count())
}

// Compare identical sorted dictionaries and candidate sets. The per-term case
// retains the previous union loop; these numbers isolate scalar filtering,
// not end-to-end vector-search QPS.
func BenchmarkInvertedIntegerRange(b *testing.B) {
	const rows = 100_000
	for _, layout := range []string{"ordered", "permuted"} {
		b.Run(layout, func(b *testing.B) {
			field := Field{Name: "id", Kind: ValueInt64, Filterable: true, Indexed: true, RangeOptimized: true}
			index, err := NewInvertedIndex(field)
			if err != nil {
				b.Fatal(err)
			}
			for row := 0; row < rows; row++ {
				value := row
				if layout == "permuted" {
					value = row * 7919 % rows
				}
				if err := index.Add(uint64(row), Int64Value(int64(value))); err != nil {
					b.Fatal(err)
				}
			}
			if err := index.Seal(); err != nil {
				b.Fatal(err)
			}
			for _, matches := range []int{100, 10_000, 20_000, 50_000, 90_000} {
				for _, mode := range []string{"per-term", "blocks"} {
					b.Run(fmt.Sprintf("matches=%d/%s", matches, mode), func(b *testing.B) {
						predicate, err := NewComparisonPredicate(PredicateGE, Int64Value(int64(rows-matches)))
						if err != nil {
							b.Fatal(err)
						}
						b.ReportAllocs()
						b.ResetTimer()
						for n := 0; n < b.N; n++ {
							var result InvertedResult
							if mode == "blocks" {
								result, err = index.Search(predicate)
							} else {
								index.mu.RLock()
								start, end, boundsErr := index.orderedBounds(predicate.right, predicate.operator)
								err = boundsErr
								bitmap := container.NewBitmap(0)
								for _, key := range index.ordered[start:end] {
									bitmap.Or(index.postings[key])
								}
								result = supportedBitmap(bitmap, InvertedSortedRange, end-start)
								index.mu.RUnlock()
							}
							if err != nil {
								b.Fatal(err)
							}
							if result.Bitmap.Count() != uint64(matches) {
								b.Fatal("incorrect candidate count")
							}
						}
					})
				}
			}
		})
	}
}
