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
	"strconv"
	"strings"
)

// BuildPlan parses, rewrites, analyzes, and optimizes one complete filter.
func BuildPlan(input string, schema Schema) (*Plan, error) {
	expression, err := ParseFilter(input)
	if err != nil {
		return nil, err
	}
	rewritten, rewriteStats := RewriteFilter(expression)
	analyzer := filterAnalyzer{schema: schema, seen: make(map[string]struct{})}
	root, err := analyzer.bindExpression(rewritten)
	if err != nil {
		return nil, err
	}
	stats := PlanStats{EqualitySets: rewriteStats.EqualitySets}
	var optimizer filterOptimizer = forwardFilterOptimizer{}
	root = optimizer.optimize(root, &stats)
	return &Plan{
		root: root, fields: analyzer.fields, normalized: Format(rewritten), stats: stats,
	}, nil
}

type filterAnalyzer struct {
	schema Schema
	seen   map[string]struct{}
	fields []Field
}

func (a *filterAnalyzer) bindExpression(expression Expr) (planNode, error) {
	switch expression := expression.(type) {
	case *LogicalExpr:
		left, err := a.bindExpression(expression.Left)
		if err != nil {
			return nil, err
		}
		right, err := a.bindExpression(expression.Right)
		if err != nil {
			return nil, err
		}
		return &logicalPlanNode{operator: expression.Operator, left: left, right: right}, nil
	case *PredicateExpr:
		return a.bindPredicate(expression)
	default:
		return nil, analysisError(expression.NodeSpan().Start, "expected logical or predicate expression, got %T", expression)
	}
}

func (a *filterAnalyzer) bindPredicate(expression *PredicateExpr) (planNode, error) {
	source, field, err := a.bindSource(expression.Left)
	if err != nil {
		return nil, err
	}
	operator := expression.Operator
	if _, isArrayLength := source.(arrayLengthSource); isArrayLength &&
		(operator < PredicateEQ || operator > PredicateGE) {
		return nil, analysisError(expression.Range.Start, "array_length supports only comparison operators")
	}
	if field.Array && operator != PredicateContainAll && operator != PredicateContainAny && operator != PredicateIsNull {
		return nil, analysisError(expression.Range.Start, "array field %q supports only CONTAIN_ALL, CONTAIN_ANY, and IS [NOT] NULL", field.Name)
	}
	if !field.Array && (operator == PredicateContainAll || operator == PredicateContainAny) {
		return nil, analysisError(expression.Range.Start, "%s requires an array field", operator)
	}

	var predicate BoundPredicate
	switch operator {
	case PredicateIsNull:
		predicate = NewNullPredicate(expression.Negated)
	case PredicateLike:
		if field.Array || field.Kind != ValueString {
			return nil, analysisError(expression.Range.Start, "LIKE requires a STRING scalar field")
		}
		literal, ok := expression.Right.(*LiteralExpr)
		if !ok || literal.Kind != LiteralString {
			return nil, analysisError(expression.Right.NodeSpan().Start, "LIKE requires a string literal")
		}
		predicate, err = NewLikePredicate(literal.Text)
	case PredicateIn:
		if field.Array {
			return nil, analysisError(expression.Range.Start, "IN requires a scalar field")
		}
		predicate, err = a.bindSetPredicate(expression, field.Kind)
	case PredicateContainAll, PredicateContainAny:
		predicate, err = a.bindSetPredicate(expression, field.Kind)
	case PredicateEQ, PredicateNE, PredicateLT, PredicateLE, PredicateGT, PredicateGE:
		if field.Array {
			return nil, analysisError(expression.Range.Start, "comparison operators do not support array fields")
		}
		if field.Kind == ValueBool && operator != PredicateEQ && operator != PredicateNE {
			return nil, analysisError(expression.Range.Start, "BOOL fields support only = and !=")
		}
		literal, ok := expression.Right.(*LiteralExpr)
		if !ok {
			return nil, analysisError(expression.Right.NodeSpan().Start, "comparison right operand must be a scalar literal")
		}
		value, valueErr := bindLiteral(literal, field.Kind)
		if valueErr != nil {
			return nil, valueErr
		}
		predicate, err = NewComparisonPredicate(operator, value)
	default:
		return nil, analysisError(expression.Range.Start, "unsupported predicate operator %s", operator)
	}
	if err != nil {
		return nil, analysisError(expression.Range.Start, "%v", err)
	}
	return &predicatePlanNode{source: source, predicate: predicate, text: Format(expression)}, nil
}

func (a *filterAnalyzer) bindSetPredicate(expression *PredicateExpr, kind ValueKind) (BoundPredicate, error) {
	list, ok := expression.Right.(*ListExpr)
	if !ok {
		return BoundPredicate{}, analysisError(expression.Right.NodeSpan().Start, "%s requires a literal list", expression.Operator)
	}
	values := make([]Value, len(list.Values))
	for index, literal := range list.Values {
		value, err := bindLiteral(literal, kind)
		if err != nil {
			return BoundPredicate{}, err
		}
		values[index] = value
	}
	predicate, err := NewSetPredicate(expression.Operator, expression.Negated, values)
	if err != nil {
		return BoundPredicate{}, analysisError(list.Range.Start, "%v", err)
	}
	return predicate, nil
}

func (a *filterAnalyzer) bindSource(expression ValueExpr) (valueSource, Field, error) {
	switch expression := expression.(type) {
	case *IdentifierExpr:
		field, err := a.requireField(expression.Name, expression.Range.Start)
		if err != nil {
			return nil, Field{}, err
		}
		return fieldSource{field: field}, field, nil
	case *CallExpr:
		if expression.Name != "array_length" {
			return nil, Field{}, analysisError(expression.Range.Start, "function %q is not supported", expression.Name)
		}
		if len(expression.Arguments) != 1 {
			return nil, Field{}, analysisError(expression.Range.Start, "array_length requires exactly one field argument")
		}
		identifier, ok := expression.Arguments[0].(*IdentifierExpr)
		if !ok {
			return nil, Field{}, analysisError(expression.Arguments[0].NodeSpan().Start, "array_length argument must be a field name")
		}
		field, err := a.requireField(identifier.Name, identifier.Range.Start)
		if err != nil {
			return nil, Field{}, err
		}
		if !field.Array {
			return nil, Field{}, analysisError(identifier.Range.Start, "array_length requires an array field, got %q", field.Name)
		}
		result := Field{Name: expression.String(), Kind: ValueUint32, Filterable: true, Nullable: field.Nullable}
		return arrayLengthSource{field: field}, result, nil
	default:
		return nil, Field{}, analysisError(expression.NodeSpan().Start, "predicate left operand must be a field or supported function")
	}
}

func (a *filterAnalyzer) requireField(name string, position Position) (Field, error) {
	field, found := a.schema.Field(name)
	if !found {
		return Field{}, analysisError(position, "field %q does not exist in schema", name)
	}
	if !field.Filterable {
		return Field{}, analysisError(position, "field %q cannot be used in scalar filters", name)
	}
	if _, seen := a.seen[name]; !seen {
		a.seen[name] = struct{}{}
		a.fields = append(a.fields, field)
	}
	return field, nil
}

func bindLiteral(literal *LiteralExpr, kind ValueKind) (Value, error) {
	typeMismatch := func(expected string) (Value, error) {
		return Value{}, analysisError(literal.Range.Start, "%s literal %q is required for %s", expected, literal.Text, kind)
	}
	switch kind {
	case ValueBinary:
		if literal.Kind != LiteralString {
			return typeMismatch("string")
		}
		return BinaryValue([]byte(literal.Text)), nil
	case ValueString:
		if literal.Kind != LiteralString {
			return typeMismatch("string")
		}
		return StringValue(literal.Text), nil
	case ValueBool:
		if literal.Kind != LiteralBool {
			return typeMismatch("Boolean")
		}
		return BoolValue(literal.Text == "true"), nil
	case ValueInt32:
		if literal.Kind != LiteralInteger {
			return typeMismatch("integer")
		}
		value, err := strconv.ParseInt(literal.Text, 10, 32)
		if err != nil {
			return Value{}, invalidNumericLiteral(literal, kind, err)
		}
		return Int32Value(int32(value)), nil
	case ValueInt64:
		if literal.Kind != LiteralInteger {
			return typeMismatch("integer")
		}
		value, err := strconv.ParseInt(literal.Text, 10, 64)
		if err != nil {
			return Value{}, invalidNumericLiteral(literal, kind, err)
		}
		return Int64Value(value), nil
	case ValueUint32:
		if literal.Kind != LiteralInteger {
			return typeMismatch("integer")
		}
		value, err := strconv.ParseUint(literal.Text, 10, 32)
		if err != nil {
			return Value{}, invalidNumericLiteral(literal, kind, err)
		}
		return Uint32Value(uint32(value)), nil
	case ValueUint64:
		if literal.Kind != LiteralInteger {
			return typeMismatch("integer")
		}
		value, err := strconv.ParseUint(literal.Text, 10, 64)
		if err != nil {
			return Value{}, invalidNumericLiteral(literal, kind, err)
		}
		return Uint64Value(value), nil
	case ValueFloat32, ValueFloat64:
		if literal.Kind != LiteralInteger && literal.Kind != LiteralFloat {
			return typeMismatch("numeric")
		}
		text := strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(literal.Text, "f"), "F"), "d"), "D")
		bits := 64
		if kind == ValueFloat32 {
			bits = 32
		}
		value, err := strconv.ParseFloat(text, bits)
		if err != nil {
			return Value{}, invalidNumericLiteral(literal, kind, err)
		}
		if kind == ValueFloat32 {
			bound, bindErr := Float32Value(float32(value))
			if bindErr != nil {
				return Value{}, analysisError(literal.Range.Start, "invalid %s literal %q: %v", kind, literal.Text, bindErr)
			}
			return bound, nil
		}
		bound, bindErr := Float64Value(value)
		if bindErr != nil {
			return Value{}, analysisError(literal.Range.Start, "invalid %s literal %q: %v", kind, literal.Text, bindErr)
		}
		return bound, nil
	default:
		return Value{}, analysisError(literal.Range.Start, "unsupported field kind %d", kind)
	}
}

func invalidNumericLiteral(literal *LiteralExpr, kind ValueKind, err error) error {
	return analysisError(literal.Range.Start, "invalid %s literal %q: %v", kind, literal.Text, err)
}
