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

import "testing"

func TestTruthKleeneLogic(t *testing.T) {
	values := []Truth{TruthFalse, TruthUnknown, TruthTrue}
	wantAnd := [][]Truth{
		{TruthFalse, TruthFalse, TruthFalse},
		{TruthFalse, TruthUnknown, TruthUnknown},
		{TruthFalse, TruthUnknown, TruthTrue},
	}
	wantOr := [][]Truth{
		{TruthFalse, TruthUnknown, TruthTrue},
		{TruthUnknown, TruthUnknown, TruthTrue},
		{TruthTrue, TruthTrue, TruthTrue},
	}
	for leftIndex, left := range values {
		for rightIndex, right := range values {
			if got := left.And(right); got != wantAnd[leftIndex][rightIndex] {
				t.Errorf("%s AND %s = %s", left, right, got)
			}
			if got := left.Or(right); got != wantOr[leftIndex][rightIndex] {
				t.Errorf("%s OR %s = %s", left, right, got)
			}
		}
	}
	if TruthTrue.Not() != TruthFalse || TruthFalse.Not() != TruthTrue || TruthUnknown.Not() != TruthUnknown {
		t.Fatal("invalid NOT truth table")
	}
	if !TruthTrue.Match() || TruthFalse.Match() || TruthUnknown.Match() {
		t.Fatal("filter Match must retain only TRUE")
	}
}

func TestComparisonPredicatesExactTypesAndNull(t *testing.T) {
	tests := []struct {
		operator PredicateOperator
		left     Value
		right    Value
		want     Truth
	}{
		{PredicateEQ, BinaryValue([]byte{0, 255}), BinaryValue([]byte{0, 255}), TruthTrue},
		{PredicateLT, BinaryValue([]byte{0}), BinaryValue([]byte{1}), TruthTrue},
		{PredicateGT, StringValue("z"), StringValue("a"), TruthTrue},
		{PredicateEQ, BoolValue(true), BoolValue(true), TruthTrue},
		{PredicateNE, BoolValue(false), BoolValue(true), TruthTrue},
		{PredicateLT, Int32Value(-2), Int32Value(-1), TruthTrue},
		{PredicateLE, Int64Value(5), Int64Value(5), TruthTrue},
		{PredicateGT, Uint32Value(^uint32(0)), Uint32Value(1), TruthTrue},
		{PredicateGE, Uint64Value(^uint64(0)), Uint64Value(^uint64(0)), TruthTrue},
	}
	float32Left, _ := Float32Value(1.25)
	float32Right, _ := Float32Value(2.5)
	tests = append(tests, struct {
		operator PredicateOperator
		left     Value
		right    Value
		want     Truth
	}{PredicateLT, float32Left, float32Right, TruthTrue})
	float64Left, _ := Float64Value(3.5)
	float64Right, _ := Float64Value(3.5)
	tests = append(tests, struct {
		operator PredicateOperator
		left     Value
		right    Value
		want     Truth
	}{PredicateEQ, float64Left, float64Right, TruthTrue})

	for _, testCase := range tests {
		predicate, err := NewComparisonPredicate(testCase.operator, testCase.right)
		if err != nil {
			t.Fatal(err)
		}
		got, err := predicate.Evaluate(testCase.left)
		if err != nil || got != testCase.want {
			t.Fatalf("%s %s: got %s, error=%v", testCase.left.describe(), testCase.operator, got, err)
		}
	}

	predicate, _ := NewComparisonPredicate(PredicateEQ, Int32Value(1))
	null, _ := NullValue(ValueInt32, false)
	if got, err := predicate.Evaluate(null); err != nil || got != TruthUnknown {
		t.Fatalf("NULL comparison = %s, %v", got, err)
	}
	if _, err := predicate.Evaluate(Int64Value(1)); err == nil {
		t.Fatal("mixed-width comparison succeeded")
	}
	boolOrder, _ := NewComparisonPredicate(PredicateLT, BoolValue(true))
	if _, err := boolOrder.Evaluate(BoolValue(false)); err == nil {
		t.Fatal("ordered BOOL comparison succeeded")
	}
}

func TestInPredicateAndDefensiveCopy(t *testing.T) {
	values := []Value{StringValue("a"), StringValue("b")}
	predicate, err := NewSetPredicate(PredicateIn, false, values)
	if err != nil {
		t.Fatal(err)
	}
	values[0] = StringValue("changed")
	for input, want := range map[string]Truth{"a": TruthTrue, "b": TruthTrue, "c": TruthFalse} {
		got, evalErr := predicate.Evaluate(StringValue(input))
		if evalErr != nil || got != want {
			t.Fatalf("IN(%q) = %s, %v; want %s", input, got, evalErr, want)
		}
	}
	notIn, _ := NewSetPredicate(PredicateIn, true, []Value{StringValue("a")})
	if got, _ := notIn.Evaluate(StringValue("b")); got != TruthTrue {
		t.Fatalf("NOT IN = %s", got)
	}
	null, _ := NullValue(ValueString, false)
	if got, _ := notIn.Evaluate(null); got != TruthUnknown {
		t.Fatalf("NULL NOT IN = %s", got)
	}
}

func TestContainPredicatesIncludingEmptyAndNull(t *testing.T) {
	array, err := ArrayValue(ValueInt32, Int32Value(1), Int32Value(2), Int32Value(2))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		operator PredicateOperator
		negated  bool
		set      []Value
		want     Truth
	}{
		{PredicateContainAll, false, []Value{Int32Value(1), Int32Value(2)}, TruthTrue},
		{PredicateContainAll, false, []Value{Int32Value(1), Int32Value(3)}, TruthFalse},
		{PredicateContainAny, false, []Value{Int32Value(3), Int32Value(2)}, TruthTrue},
		{PredicateContainAny, false, []Value{Int32Value(3)}, TruthFalse},
		{PredicateContainAll, false, nil, TruthTrue},
		{PredicateContainAll, true, nil, TruthFalse},
		{PredicateContainAny, false, nil, TruthFalse},
		{PredicateContainAny, true, nil, TruthTrue},
	}
	for _, testCase := range tests {
		predicate, bindErr := NewSetPredicate(testCase.operator, testCase.negated, testCase.set)
		if bindErr != nil {
			t.Fatal(bindErr)
		}
		got, evalErr := predicate.Evaluate(array)
		if evalErr != nil || got != testCase.want {
			t.Fatalf("%s negated=%t set=%v = %s, %v; want %s", testCase.operator, testCase.negated, testCase.set, got, evalErr, testCase.want)
		}
	}
	null, _ := NullValue(ValueInt32, true)
	predicate, _ := NewSetPredicate(PredicateContainAny, true, []Value{Int32Value(1)})
	if got, _ := predicate.Evaluate(null); got != TruthUnknown {
		t.Fatalf("NULL NOT CONTAIN_ANY = %s", got)
	}
}

func TestNullAndLikePredicates(t *testing.T) {
	null, _ := NullValue(ValueString, false)
	for _, testCase := range []struct {
		predicate BoundPredicate
		value     Value
		want      Truth
	}{
		{NewNullPredicate(false), null, TruthTrue},
		{NewNullPredicate(true), null, TruthFalse},
		{NewNullPredicate(false), StringValue(""), TruthFalse},
		{NewNullPredicate(true), StringValue(""), TruthTrue},
	} {
		got, err := testCase.predicate.Evaluate(testCase.value)
		if err != nil || got != testCase.want {
			t.Fatalf("null predicate = %s, %v; want %s", got, err, testCase.want)
		}
	}
	like, err := NewLikePredicate(`user-\_%`)
	if err != nil || like.Operator() != PredicateLike || like.LikePattern().Mode() != LikePrefix {
		t.Fatalf("LIKE bind = %#v, %v", like, err)
	}
	if got, _ := like.Evaluate(StringValue("user-_22")); got != TruthTrue {
		t.Fatalf("LIKE = %s", got)
	}
	if got, _ := like.Evaluate(null); got != TruthUnknown {
		t.Fatalf("NULL LIKE = %s", got)
	}
}

func TestPredicateBindingRejectsInvalidForms(t *testing.T) {
	null, _ := NullValue(ValueInt32, false)
	array, _ := ArrayValue(ValueInt32, Int32Value(1))
	tests := []func() error{
		func() error { _, err := NewComparisonPredicate(PredicateLike, Int32Value(1)); return err },
		func() error { _, err := NewComparisonPredicate(PredicateEQ, null); return err },
		func() error { _, err := NewComparisonPredicate(PredicateEQ, array); return err },
		func() error { _, err := NewSetPredicate(PredicateEQ, false, nil); return err },
		func() error { _, err := NewSetPredicate(PredicateIn, false, nil); return err },
		func() error {
			_, err := NewSetPredicate(PredicateIn, false, []Value{Int32Value(1), Int64Value(1)})
			return err
		},
		func() error {
			values := make([]Value, MaxContainValues+1)
			for index := range values {
				values[index] = Int32Value(int32(index))
			}
			_, err := NewSetPredicate(PredicateContainAll, false, values)
			return err
		},
	}
	for index, testCase := range tests {
		if err := testCase(); err == nil {
			t.Fatalf("invalid binding %d succeeded", index)
		}
	}
	predicate := NewNullPredicate(false)
	if _, err := predicate.Evaluate(Value{}); err == nil {
		t.Fatal("invalid runtime value succeeded")
	}
}
