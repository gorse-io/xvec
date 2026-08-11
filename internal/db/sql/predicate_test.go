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

package sql

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
			{
				got := left.And(right)
				assert.Equal(t, wantAnd[leftIndex][rightIndex], got)
			}
			{
				got := left.Or(right)
				assert.Equal(t, wantOr[leftIndex][rightIndex], got)
			}
		}
	}
	require.Equal(t, TruthFalse, TruthTrue.Not(),
		"invalid NOT truth table")
	require.Equal(t, TruthTrue, TruthFalse.Not(),
		"invalid NOT truth table")
	require.Equal(t, TruthUnknown, TruthUnknown.Not(),
		"invalid NOT truth table")
	require.True(t, TruthTrue.Match(),
		"filter Match must retain only TRUE")
	require.False(t, TruthFalse.Match(),
		"filter Match must retain only TRUE")
	require.False(t, TruthUnknown.Match(),
		"filter Match must retain only TRUE")
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
		require.NoError(t, err)

		got, err := predicate.Evaluate(testCase.left)
		require.NoError(t, err)
		require.Equal(t, testCase.want, got)
	}

	predicate, _ := NewComparisonPredicate(PredicateEQ, Int32Value(1))
	null, _ := NullValue(ValueInt32, false)
	{
		got, err := predicate.Evaluate(null)
		require.NoError(t, err)
		require.Equal(t, TruthUnknown, got)
	}
	{
		_, err := predicate.Evaluate(Int64Value(1))
		require.Error(t, err,
			"mixed-width comparison succeeded")
	}

	boolOrder, _ := NewComparisonPredicate(PredicateLT, BoolValue(true))
	{
		_, err := boolOrder.Evaluate(BoolValue(false))
		require.Error(t, err,
			"ordered BOOL comparison succeeded")
	}
}

func TestInPredicateAndDefensiveCopy(t *testing.T) {
	values := []Value{StringValue("a"), StringValue("b")}
	predicate, err := NewSetPredicate(PredicateIn, false, values)
	require.NoError(t, err)

	values[0] = StringValue("changed")
	for input, want := range map[string]Truth{"a": TruthTrue, "b": TruthTrue, "c": TruthFalse} {
		got, evalErr := predicate.Evaluate(StringValue(input))
		require.NoError(t, evalErr)
		require.Equal(t, want, got)
	}
	notIn, _ := NewSetPredicate(PredicateIn, true, []Value{StringValue("a")})
	{
		got, _ := notIn.Evaluate(StringValue("b"))
		require.Equal(t, TruthTrue, got)
	}

	null, _ := NullValue(ValueString, false)
	{
		got, _ := notIn.Evaluate(null)
		require.Equal(t, TruthUnknown, got)
	}
}

func TestContainPredicatesIncludingEmptyAndNull(t *testing.T) {
	array, err := ArrayValue(ValueInt32, Int32Value(1), Int32Value(2), Int32Value(2))
	require.NoError(t, err)

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
		require.NoError(t, bindErr)

		got, evalErr := predicate.Evaluate(array)
		require.NoError(t, evalErr)
		require.Equal(t, testCase.want, got)
	}
	null, _ := NullValue(ValueInt32, true)
	predicate, _ := NewSetPredicate(PredicateContainAny, true, []Value{Int32Value(1)})
	{
		got, _ := predicate.Evaluate(null)
		require.Equal(t, TruthUnknown, got)
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
		require.NoError(t, err)
		require.Equal(t, testCase.want, got)
	}
	like, err := NewLikePredicate(`user-\_%`)
	require.NoError(t, err)
	require.Equal(t, PredicateLike, like.Operator())
	require.Equal(t, LikePrefix, like.LikePattern().Mode())
	{
		got, _ := like.Evaluate(StringValue("user-_22"))
		require.Equal(t, TruthTrue, got)
	}
	{
		got, _ := like.Evaluate(null)
		require.Equal(t, TruthUnknown, got)
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
	for _, testCase := range tests {
		{
			err := testCase()
			require.Error(t, err)
		}
	}
	predicate := NewNullPredicate(false)
	{
		_, err := predicate.Evaluate(Value{})
		require.Error(t, err,
			"invalid runtime value succeeded")
	}
}
