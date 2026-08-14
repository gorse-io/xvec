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
	"math"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/gorse-io/xvec/internal/ailego/container"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInvertedScalarPredicatesAndRangeStrategies(t *testing.T) {
	field := Field{Name: "number", Kind: ValueInt32, Nullable: true, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field,
		mustNullValue(t, ValueInt32, false), Int32Value(1), Int32Value(2), Int32Value(2), Int32Value(4),
	)

	comparison := func(operator PredicateOperator, value int32) BoundPredicate {
		predicate, err := NewComparisonPredicate(operator, Int32Value(value))
		require.NoError(t, err)

		return predicate
	}
	for _, testCase := range []struct {
		name     string
		query    BoundPredicate
		want     []uint64
		strategy InvertedStrategy
		terms    int
	}{
		{"equal", comparison(PredicateEQ, 2), []uint64{2, 3}, InvertedExact, 1},
		{"not-equal", comparison(PredicateNE, 2), []uint64{1, 4}, InvertedExact, 1},
		{"less", comparison(PredicateLT, 2), []uint64{1}, InvertedSortedRange, 1},
		{"less-equal", comparison(PredicateLE, 2), []uint64{1, 2, 3}, InvertedSortedRange, 2},
		{"greater", comparison(PredicateGT, 2), []uint64{4}, InvertedSortedRange, 1},
		{"greater-equal", comparison(PredicateGE, 2), []uint64{2, 3, 4}, InvertedSortedRange, 2},
		{"null", NewNullPredicate(false), []uint64{0}, InvertedNull, 1},
		{"not-null", NewNullPredicate(true), []uint64{1, 2, 3, 4}, InvertedNull, 1},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			result, err := index.Search(testCase.query)
			require.NoError(t, err)

			assertInvertedResult(t, result, testCase.want, testCase.strategy, testCase.terms)
		})
	}

	in, err := NewSetPredicate(PredicateIn, false, []Value{Int32Value(1), Int32Value(4)})
	require.NoError(t, err)

	result, err := index.Search(in)
	require.NoError(t, err)

	assertInvertedResult(t, result, []uint64{1, 4}, InvertedPostingUnion, 2)
	notIn, err := NewSetPredicate(PredicateIn, true, []Value{Int32Value(1), Int32Value(4)})
	require.NoError(t, err)

	result, err = index.Search(notIn)
	require.NoError(t, err)

	assertInvertedResult(t, result, []uint64{2, 3}, InvertedPostingUnion, 2)

	field.RangeOptimized = false
	scan := mustInvertedIndex(t, field, Int32Value(1), Int32Value(2), Int32Value(4))
	result, err = scan.Search(comparison(PredicateGE, 2))
	require.NoError(t, err)

	assertInvertedResult(t, result, []uint64{1, 2}, InvertedTermScan, 2)
}

func TestInvertedArrayContainAndLength(t *testing.T) {
	field := Field{Name: "tags", Kind: ValueString, Array: true, Nullable: true, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field,
		mustNullValue(t, ValueString, true),
		mustArray(t, ValueString),
		mustArray(t, ValueString, StringValue("a"), StringValue("b"), StringValue("b")),
		mustArray(t, ValueString, StringValue("b"), StringValue("c")),
		mustArray(t, ValueString, StringValue("a")),
	)
	for _, testCase := range []struct {
		name    string
		op      PredicateOperator
		negated bool
		values  []Value
		want    []uint64
	}{
		{"all", PredicateContainAll, false, []Value{StringValue("a"), StringValue("b")}, []uint64{2}},
		{"any", PredicateContainAny, false, []Value{StringValue("a"), StringValue("c")}, []uint64{2, 3, 4}},
		{"not-any", PredicateContainAny, true, []Value{StringValue("a")}, []uint64{1, 3}},
		{"duplicates-do-not-change-postings", PredicateContainAll, false, []Value{StringValue("b")}, []uint64{2, 3}},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			predicate, err := NewSetPredicate(testCase.op, testCase.negated, testCase.values)
			require.NoError(t, err)

			result, err := index.Search(predicate)
			require.NoError(t, err)

			assertInvertedResult(t, result, testCase.want, InvertedPostingUnion, len(testCase.values))
		})
	}

	length := func(operator PredicateOperator, value uint32) BoundPredicate {
		predicate, err := NewComparisonPredicate(operator, Uint32Value(value))
		require.NoError(t, err)

		return predicate
	}
	for _, testCase := range []struct {
		operator PredicateOperator
		value    uint32
		want     []uint64
	}{
		{PredicateEQ, 0, []uint64{1}},
		{PredicateNE, 2, []uint64{1, 2, 4}},
		{PredicateLT, 2, []uint64{1, 4}},
		{PredicateGE, 2, []uint64{2, 3}},
	} {
		result, err := index.SearchArrayLength(length(testCase.operator, testCase.value))
		require.NoError(t, err)
		{
			got := bitmapBits(result.Bitmap)
			assert.Equal(t, testCase.want, got)
		}
	}
}

func TestInvertedFloatCanonicalizesSignedZero(t *testing.T) {
	negativeZero, err := Float64Value(math.Copysign(0, -1))
	require.NoError(t, err)

	positiveZero, err := Float64Value(0.0)
	require.NoError(t, err)

	one, err := Float64Value(1)
	require.NoError(t, err)

	field := Field{Name: "number", Kind: ValueFloat64, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field, negativeZero, positiveZero, one)
	predicate, err := NewComparisonPredicate(PredicateEQ, positiveZero)
	require.NoError(t, err)

	result, err := index.Search(predicate)
	require.NoError(t, err)

	assertInvertedResult(t, result, []uint64{0, 1}, InvertedExact, 1)
}

func TestInvertedLikeRouting(t *testing.T) {
	values := []Value{
		mustNullValue(t, ValueString, false), StringValue("alpha"), StringValue("alphabet"),
		StringValue("beta"), StringValue("gamma-alpha"),
	}
	base := Field{Name: "name", Kind: ValueString, Nullable: true, Filterable: true, Indexed: true}
	plain := mustInvertedIndex(t, base, values...)
	extendedField := base
	extendedField.ExtendedWildcard = true
	extended := mustInvertedIndex(t, extendedField, values...)

	search := func(index *InvertedIndex, pattern string) InvertedResult {
		predicate, err := NewLikePredicate(pattern)
		require.NoError(t, err)

		result, err := index.Search(predicate)
		require.NoError(t, err)

		return result
	}
	assertInvertedResult(t, search(plain, "alpha"), []uint64{1}, InvertedExact, 1)
	assertInvertedResult(t, search(plain, "%"), []uint64{1, 2, 3, 4}, InvertedPrefix, 4)
	assertInvertedResult(t, search(plain, "alp%"), []uint64{1, 2}, InvertedPrefix, 2)
	{
		result := search(plain, "%alpha")
		require.False(t, result.Supported,
			"suffix search used index without extended wildcard")
	}

	assertInvertedResult(t, search(extended, "%alpha"), []uint64{1, 4}, InvertedSuffix, 2)
	assertInvertedResult(t, search(extended, "al%bet"), []uint64{2}, InvertedPrefixSuffix, 1)
	for _, pattern := range []string{"%pha%", "a_pha", "a%b%t", "a%%bet", "%%"} {
		{
			result := search(extended, pattern)
			assert.False(t, result.Supported)
		}
	}
}

func TestPlanCandidateCompositionNeverDropsPossibleMatches(t *testing.T) {
	fields := []Field{
		{Name: "number", Kind: ValueInt32, Filterable: true, Indexed: true, RangeOptimized: true},
		{Name: "name", Kind: ValueString, Filterable: true},
		{Name: "tags", Kind: ValueString, Array: true, Filterable: true, Indexed: true},
	}
	schema, err := NewSchema(fields)
	require.NoError(t, err)

	indexes := IndexSet{
		"number": mustInvertedIndex(t, fields[0], Int32Value(1), Int32Value(2), Int32Value(3)),
		"tags": mustInvertedIndex(t, fields[2],
			mustArray(t, ValueString), mustArray(t, ValueString, StringValue("a")),
			mustArray(t, ValueString, StringValue("a"), StringValue("b"))),
	}
	for _, testCase := range []struct {
		filter string
		want   []uint64
		used   bool
		exact  bool
	}{
		{"number>=2 AND name='x'", []uint64{1, 2}, true, false},
		{"number=1 OR name='x'", nil, false, false},
		{"number=1 OR number=3", []uint64{0, 2}, true, true},
		{"array_length(tags)>=2", []uint64{2}, true, true},
		{"tags CONTAIN_ANY ()", []uint64{}, true, true},
	} {
		plan, err := BuildPlan(testCase.filter, schema)
		require.NoError(t, err)

		bitmap, used, exact, err := plan.Candidates(indexes, 3)
		require.NoError(t, err)

		var got []uint64
		if bitmap != nil {
			got = bitmapBits(bitmap)
		}
		assert.Equal(t, testCase.want, got)
		assert.Equal(t, testCase.used, used)
		assert.Equal(t, testCase.exact, exact)
	}
}

func TestInvertedIndexLifecycleAndConcurrentSearch(t *testing.T) {
	field := Field{Name: "number", Kind: ValueInt32, Filterable: true, Indexed: true}
	{
		_, err := NewInvertedIndex(Field{Name: "bad", Kind: ValueInt32, Filterable: true})
		require.Error(t, err,
			"unindexed field succeeded")
	}

	index, err := NewInvertedIndex(field)
	require.NoError(t, err)
	{
		_, err := index.Search(NewNullPredicate(false))
		require.Error(t, err,
			"unsealed search succeeded")
	}
	{
		err := index.Add(0, Int32Value(1))
		require.NoError(t, err)
	}
	{
		err := index.Add(0, Int32Value(1))
		require.Error(t, err,
			"duplicate row succeeded")
	}
	{
		err := index.Seal()
		require.NoError(t, err)
	}
	{
		err := index.Add(1, Int32Value(2))
		require.Error(t, err,
			"add after seal succeeded")
	}

	stringIndex, err := NewInvertedIndex(Field{Name: "text", Kind: ValueString, Filterable: true, Indexed: true})
	require.NoError(t, err)
	{
		err := stringIndex.Add(0, StringValue(string([]byte{0xff})))
		require.Error(t, err,
			"invalid UTF-8 string succeeded")
	}

	predicate, err := NewComparisonPredicate(PredicateEQ, Int32Value(1))
	require.NoError(t, err)

	var wait sync.WaitGroup
	for worker := 0; worker < 16; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				result, searchErr := index.Search(predicate)
				if !assert.NoError(t, searchErr) {
					return
				}
				if !assert.Equal(t, uint64(1), result.Bitmap.Count()) {
					return
				}
			}
		}()
	}
	wait.Wait()
}

func FuzzInvertedLikeCandidate(f *testing.F) {
	for _, seed := range [][2]string{
		{"alpha", "alp%"}, {"gamma-alpha", "%alpha"}, {"alphabet", "al%bet"},
		{"a%b", `a\%b`}, {"世界", "世_"}, {"", "%"},
	} {
		f.Add(seed[0], seed[1])
	}
	f.Fuzz(func(t *testing.T, text, pattern string) {
		if len(text) > 256 || len(pattern) > 256 || !utf8.ValidString(text) {
			t.Skip()
		}
		predicate, err := NewLikePredicate(pattern)
		if err != nil {
			return
		}
		field := Field{Name: "text", Kind: ValueString, Filterable: true, Indexed: true, ExtendedWildcard: true}
		index := mustInvertedIndex(t, field, StringValue(text))
		result, err := index.Search(predicate)
		require.NoError(t, err)

		if !result.Supported {
			return
		}
		truth, err := predicate.Evaluate(StringValue(text))
		require.NoError(t, err)
		require.Equal(t, truth.Match(), result.Bitmap.Contains(0))
	})
}

func mustInvertedIndex(t *testing.T, field Field, values ...Value) *InvertedIndex {
	t.Helper()
	index, err := NewInvertedIndex(field)
	require.NoError(t, err)

	for row, value := range values {
		{
			err := index.Add(uint64(row), value)
			require.NoError(t, err)
		}
	}
	{
		err := index.Seal()
		require.NoError(t, err)
	}

	return index
}

func mustNullValue(t *testing.T, kind ValueKind, array bool) Value {
	t.Helper()
	value, err := NullValue(kind, array)
	require.NoError(t, err)

	return value
}

func assertInvertedResult(t *testing.T, result InvertedResult, want []uint64, strategy InvertedStrategy, terms int) {
	t.Helper()
	require.True(t, result.Supported)
	require.Equal(t, strategy, result.Strategy)
	require.Equal(t, terms, result.Terms)
	{
		got := bitmapBits(result.Bitmap)
		require.Equal(t, want, got)
	}
}

func bitmapBits(bitmap *container.Bitmap) []uint64 {
	if bitmap == nil {
		return nil
	}
	bits := make([]uint64, 0, bitmap.Count())
	bitmap.Range(func(bit uint64) bool {
		bits = append(bits, bit)
		return true
	})
	return bits
}
