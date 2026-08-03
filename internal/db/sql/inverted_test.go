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
	"math"
	"reflect"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/gorse-io/zvec/internal/ailego"
)

func TestInvertedScalarPredicatesAndRangeStrategies(t *testing.T) {
	field := Field{Name: "number", Kind: ValueInt32, Nullable: true, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field,
		mustNullValue(t, ValueInt32, false), Int32Value(1), Int32Value(2), Int32Value(2), Int32Value(4),
	)

	comparison := func(operator PredicateOperator, value int32) BoundPredicate {
		predicate, err := NewComparisonPredicate(operator, Int32Value(value))
		if err != nil {
			t.Fatal(err)
		}
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
			if err != nil {
				t.Fatal(err)
			}
			assertInvertedResult(t, result, testCase.want, testCase.strategy, testCase.terms)
		})
	}

	in, err := NewSetPredicate(PredicateIn, false, []Value{Int32Value(1), Int32Value(4)})
	if err != nil {
		t.Fatal(err)
	}
	result, err := index.Search(in)
	if err != nil {
		t.Fatal(err)
	}
	assertInvertedResult(t, result, []uint64{1, 4}, InvertedPostingUnion, 2)
	notIn, err := NewSetPredicate(PredicateIn, true, []Value{Int32Value(1), Int32Value(4)})
	if err != nil {
		t.Fatal(err)
	}
	result, err = index.Search(notIn)
	if err != nil {
		t.Fatal(err)
	}
	assertInvertedResult(t, result, []uint64{2, 3}, InvertedPostingUnion, 2)

	field.RangeOptimized = false
	scan := mustInvertedIndex(t, field, Int32Value(1), Int32Value(2), Int32Value(4))
	result, err = scan.Search(comparison(PredicateGE, 2))
	if err != nil {
		t.Fatal(err)
	}
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
			if err != nil {
				t.Fatal(err)
			}
			result, err := index.Search(predicate)
			if err != nil {
				t.Fatal(err)
			}
			assertInvertedResult(t, result, testCase.want, InvertedPostingUnion, len(testCase.values))
		})
	}

	length := func(operator PredicateOperator, value uint32) BoundPredicate {
		predicate, err := NewComparisonPredicate(operator, Uint32Value(value))
		if err != nil {
			t.Fatal(err)
		}
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
		if err != nil {
			t.Fatal(err)
		}
		if got := bitmapBits(result.Bitmap); !reflect.DeepEqual(got, testCase.want) {
			t.Errorf("array_length %s %d = %v, want %v", testCase.operator, testCase.value, got, testCase.want)
		}
	}
}

func TestInvertedFloatCanonicalizesSignedZero(t *testing.T) {
	negativeZero, err := Float64Value(math.Copysign(0, -1))
	if err != nil {
		t.Fatal(err)
	}
	positiveZero, err := Float64Value(0.0)
	if err != nil {
		t.Fatal(err)
	}
	one, err := Float64Value(1)
	if err != nil {
		t.Fatal(err)
	}
	field := Field{Name: "number", Kind: ValueFloat64, Filterable: true, Indexed: true, RangeOptimized: true}
	index := mustInvertedIndex(t, field, negativeZero, positiveZero, one)
	predicate, err := NewComparisonPredicate(PredicateEQ, positiveZero)
	if err != nil {
		t.Fatal(err)
	}
	result, err := index.Search(predicate)
	if err != nil {
		t.Fatal(err)
	}
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
		if err != nil {
			t.Fatal(err)
		}
		result, err := index.Search(predicate)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	assertInvertedResult(t, search(plain, "alpha"), []uint64{1}, InvertedExact, 1)
	assertInvertedResult(t, search(plain, "%"), []uint64{1, 2, 3, 4}, InvertedPrefix, 4)
	assertInvertedResult(t, search(plain, "alp%"), []uint64{1, 2}, InvertedPrefix, 2)
	if result := search(plain, "%alpha"); result.Supported {
		t.Fatal("suffix search used index without extended wildcard")
	}
	assertInvertedResult(t, search(extended, "%alpha"), []uint64{1, 4}, InvertedSuffix, 2)
	assertInvertedResult(t, search(extended, "al%bet"), []uint64{2}, InvertedPrefixSuffix, 1)
	for _, pattern := range []string{"%pha%", "a_pha", "a%b%t", "a%%bet", "%%"} {
		if result := search(extended, pattern); result.Supported {
			t.Errorf("general pattern %q unexpectedly used index", pattern)
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
	if err != nil {
		t.Fatal(err)
	}
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
		if err != nil {
			t.Fatal(err)
		}
		bitmap, used, exact, err := plan.Candidates(indexes, 3)
		if err != nil {
			t.Fatal(err)
		}
		var got []uint64
		if bitmap != nil {
			got = bitmapBits(bitmap)
		}
		if !reflect.DeepEqual(got, testCase.want) || used != testCase.used || exact != testCase.exact {
			t.Errorf("Candidates(%q) = %v used=%t exact=%t; want %v used=%t exact=%t", testCase.filter, got, used, exact, testCase.want, testCase.used, testCase.exact)
		}
	}
}

func TestInvertedIndexLifecycleAndConcurrentSearch(t *testing.T) {
	field := Field{Name: "number", Kind: ValueInt32, Filterable: true, Indexed: true}
	if _, err := NewInvertedIndex(Field{Name: "bad", Kind: ValueInt32, Filterable: true}); err == nil {
		t.Fatal("unindexed field succeeded")
	}
	if _, err := NewInvertedIndex(Field{Name: "binary", Kind: ValueBinary, Filterable: true, Indexed: true}); err == nil {
		t.Fatal("binary field succeeded")
	}
	index, err := NewInvertedIndex(field)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := index.Search(NewNullPredicate(false)); err == nil {
		t.Fatal("unsealed search succeeded")
	}
	if err := index.Add(0, Int32Value(1)); err != nil {
		t.Fatal(err)
	}
	if err := index.Add(0, Int32Value(1)); err == nil {
		t.Fatal("duplicate row succeeded")
	}
	if err := index.Seal(); err != nil {
		t.Fatal(err)
	}
	if err := index.Add(1, Int32Value(2)); err == nil {
		t.Fatal("add after seal succeeded")
	}
	stringIndex, err := NewInvertedIndex(Field{Name: "text", Kind: ValueString, Filterable: true, Indexed: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := stringIndex.Add(0, StringValue(string([]byte{0xff}))); err == nil {
		t.Fatal("invalid UTF-8 string succeeded")
	}
	predicate, err := NewComparisonPredicate(PredicateEQ, Int32Value(1))
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	for worker := 0; worker < 16; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				result, searchErr := index.Search(predicate)
				if searchErr != nil || result.Bitmap.Count() != 1 {
					t.Errorf("concurrent search count=%d error=%v", result.Bitmap.Count(), searchErr)
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
		if err != nil {
			t.Fatal(err)
		}
		if !result.Supported {
			return
		}
		truth, err := predicate.Evaluate(StringValue(text))
		if err != nil {
			t.Fatal(err)
		}
		if result.Bitmap.Contains(0) != truth.Match() {
			t.Fatalf("indexed LIKE mismatch: text=%q pattern=%q bitmap=%t truth=%s", text, pattern, result.Bitmap.Contains(0), truth)
		}
	})
}

func mustInvertedIndex(t *testing.T, field Field, values ...Value) *InvertedIndex {
	t.Helper()
	index, err := NewInvertedIndex(field)
	if err != nil {
		t.Fatal(err)
	}
	for row, value := range values {
		if err := index.Add(uint64(row), value); err != nil {
			t.Fatal(err)
		}
	}
	if err := index.Seal(); err != nil {
		t.Fatal(err)
	}
	return index
}

func mustNullValue(t *testing.T, kind ValueKind, array bool) Value {
	t.Helper()
	value, err := NullValue(kind, array)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func assertInvertedResult(t *testing.T, result InvertedResult, want []uint64, strategy InvertedStrategy, terms int) {
	t.Helper()
	if !result.Supported || result.Strategy != strategy || result.Terms != terms {
		t.Fatalf("result = supported=%t strategy=%s terms=%d; want true %s %d", result.Supported, result.Strategy, result.Terms, strategy, terms)
	}
	if got := bitmapBits(result.Bitmap); !reflect.DeepEqual(got, want) {
		t.Fatalf("bitmap = %v, want %v", got, want)
	}
}

func bitmapBits(bitmap *ailego.Bitmap) []uint64 {
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
