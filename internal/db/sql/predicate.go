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

import "fmt"

// MaxContainValues matches the baseline maximum array field length.
const MaxContainValues = 32

// Truth is a SQL three-valued Boolean. A filter retains only TruthTrue;
// comparisons against NULL propagate TruthUnknown through logical operators.
type Truth uint8

const (
	TruthUnknown Truth = iota
	TruthFalse
	TruthTrue
)

func (t Truth) String() string {
	switch t {
	case TruthUnknown:
		return "UNKNOWN"
	case TruthFalse:
		return "FALSE"
	case TruthTrue:
		return "TRUE"
	default:
		return fmt.Sprintf("Truth(%d)", t)
	}
}

func (t Truth) Match() bool { return t == TruthTrue }

func (t Truth) Not() Truth {
	switch t {
	case TruthTrue:
		return TruthFalse
	case TruthFalse:
		return TruthTrue
	default:
		return TruthUnknown
	}
}

func (t Truth) And(other Truth) Truth {
	if t == TruthFalse || other == TruthFalse {
		return TruthFalse
	}
	if t == TruthTrue && other == TruthTrue {
		return TruthTrue
	}
	return TruthUnknown
}

func (t Truth) Or(other Truth) Truth {
	if t == TruthTrue || other == TruthTrue {
		return TruthTrue
	}
	if t == TruthFalse && other == TruthFalse {
		return TruthFalse
	}
	return TruthUnknown
}

// BoundPredicate is a schema-typed predicate kernel. The analyzer constructs
// it from the syntax AST; Evaluate then performs no parsing or numeric casts.
type BoundPredicate struct {
	operator PredicateOperator
	negated  bool
	right    Value
	set      []Value
	like     *LikePattern
}

func NewComparisonPredicate(operator PredicateOperator, right Value) (BoundPredicate, error) {
	if operator < PredicateEQ || operator > PredicateGE {
		return BoundPredicate{}, fmt.Errorf("sql: %s is not a comparison operator", operator)
	}
	if right.null || right.array || !right.kind.valid() {
		return BoundPredicate{}, fmt.Errorf("sql: comparison right operand must be a non-null scalar")
	}
	return BoundPredicate{operator: operator, right: right.clone()}, nil
}

func NewLikePredicate(pattern string) (BoundPredicate, error) {
	compiled, err := CompileLike(pattern)
	if err != nil {
		return BoundPredicate{}, err
	}
	return BoundPredicate{operator: PredicateLike, like: compiled}, nil
}

// NewSetPredicate binds IN or a contain predicate. IN requires a non-empty
// set; contain predicates intentionally retain baseline empty-set semantics.
func NewSetPredicate(operator PredicateOperator, negated bool, values []Value) (BoundPredicate, error) {
	if operator != PredicateIn && operator != PredicateContainAll && operator != PredicateContainAny {
		return BoundPredicate{}, fmt.Errorf("sql: %s is not a set predicate", operator)
	}
	if operator == PredicateIn && len(values) == 0 {
		return BoundPredicate{}, fmt.Errorf("sql: IN requires at least one value")
	}
	if operator != PredicateIn && len(values) > MaxContainValues {
		return BoundPredicate{}, fmt.Errorf("sql: %s supports at most %d values", operator, MaxContainValues)
	}
	clone := make([]Value, len(values))
	var kind ValueKind
	for index, value := range values {
		if value.null || value.array || !value.kind.valid() {
			return BoundPredicate{}, fmt.Errorf("sql: set value %d must be a non-null scalar", index)
		}
		if index == 0 {
			kind = value.kind
		} else if value.kind != kind {
			return BoundPredicate{}, fmt.Errorf("sql: set value %d is %s, want %s", index, value.kind, kind)
		}
		clone[index] = value.clone()
	}
	return BoundPredicate{operator: operator, negated: negated, set: clone}, nil
}

func NewNullPredicate(negated bool) BoundPredicate {
	return BoundPredicate{operator: PredicateIsNull, negated: negated}
}

func (p BoundPredicate) Operator() PredicateOperator { return p.operator }
func (p BoundPredicate) Negated() bool               { return p.negated }

func (p BoundPredicate) LikePattern() *LikePattern {
	return p.like
}

// Evaluate applies a bound predicate to a field value.
func (p BoundPredicate) Evaluate(left Value) (Truth, error) {
	if !left.kind.valid() {
		return TruthUnknown, fmt.Errorf("sql: predicate input has invalid value kind %d", left.kind)
	}
	switch p.operator {
	case PredicateIsNull:
		result := TruthFalse
		if left.null {
			result = TruthTrue
		}
		if p.negated {
			result = result.Not()
		}
		return result, nil
	case PredicateLike:
		if left.null {
			return TruthUnknown, nil
		}
		if left.array || left.kind != ValueString || p.like == nil {
			return TruthUnknown, fmt.Errorf("sql: LIKE requires a STRING scalar and compiled pattern")
		}
		if p.like.Match(left.text) {
			return TruthTrue, nil
		}
		return TruthFalse, nil
	case PredicateIn:
		return p.evaluateIn(left)
	case PredicateContainAll, PredicateContainAny:
		return p.evaluateContain(left)
	case PredicateEQ, PredicateNE, PredicateLT, PredicateLE, PredicateGT, PredicateGE:
		if left.null {
			return TruthUnknown, nil
		}
		if left.kind == ValueBool && p.operator != PredicateEQ && p.operator != PredicateNE {
			return TruthUnknown, fmt.Errorf("sql: BOOL supports only = and !=")
		}
		comparison, err := compareValues(left, p.right)
		if err != nil {
			return TruthUnknown, err
		}
		matched := false
		switch p.operator {
		case PredicateEQ:
			matched = comparison == 0
		case PredicateNE:
			matched = comparison != 0
		case PredicateLT:
			matched = comparison < 0
		case PredicateLE:
			matched = comparison <= 0
		case PredicateGT:
			matched = comparison > 0
		case PredicateGE:
			matched = comparison >= 0
		}
		return truthFromBool(matched), nil
	default:
		return TruthUnknown, fmt.Errorf("sql: invalid bound predicate operator %d", p.operator)
	}
}

func (p BoundPredicate) evaluateIn(left Value) (Truth, error) {
	if left.null {
		return TruthUnknown, nil
	}
	if left.array {
		return TruthUnknown, fmt.Errorf("sql: IN requires a scalar field")
	}
	matched := false
	for _, candidate := range p.set {
		comparison, err := compareValues(left, candidate)
		if err != nil {
			return TruthUnknown, err
		}
		if comparison == 0 {
			matched = true
			break
		}
	}
	result := truthFromBool(matched)
	if p.negated {
		result = result.Not()
	}
	return result, nil
}

func (p BoundPredicate) evaluateContain(left Value) (Truth, error) {
	if left.null {
		return TruthUnknown, nil
	}
	if !left.array {
		return TruthUnknown, fmt.Errorf("sql: %s requires an array field", p.operator)
	}
	if len(p.set) > 0 && left.kind != p.set[0].kind {
		return TruthUnknown, fmt.Errorf("sql: array element kind %s does not match set kind %s", left.kind, p.set[0].kind)
	}
	matched := p.operator == PredicateContainAll
	if p.operator == PredicateContainAll {
		for _, needle := range p.set {
			found, err := containsValue(left.elements, needle)
			if err != nil {
				return TruthUnknown, err
			}
			if !found {
				matched = false
				break
			}
		}
	} else {
		for _, needle := range p.set {
			found, err := containsValue(left.elements, needle)
			if err != nil {
				return TruthUnknown, err
			}
			if found {
				matched = true
				break
			}
		}
	}
	result := truthFromBool(matched)
	if p.negated {
		result = result.Not()
	}
	return result, nil
}

func containsValue(values []Value, needle Value) (bool, error) {
	for _, value := range values {
		comparison, err := compareValues(value, needle)
		if err != nil {
			return false, err
		}
		if comparison == 0 {
			return true, nil
		}
	}
	return false, nil
}

func truthFromBool(value bool) Truth {
	if value {
		return TruthTrue
	}
	return TruthFalse
}
