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
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestParseFilterLogicalPrecedenceAndAssociativity(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"a=1 or b=2 and c=3", "(a = 1 OR (b = 2 AND c = 3))"},
		{"a=1 and b=2 or c=3", "((a = 1 AND b = 2) OR c = 3)"},
		{"a=1 or b=2 or c=3", "((a = 1 OR b = 2) OR c = 3)"},
		{"a=1 and (b=2 or c=3)", "(a = 1 AND (b = 2 OR c = 3))"},
	}
	for _, testCase := range tests {
		expression, err := ParseFilter(testCase.input)
		if err != nil {
			t.Fatalf("ParseFilter(%q): %v", testCase.input, err)
		}
		if got := Format(expression); got != testCase.want {
			t.Fatalf("Format(%q) = %q, want %q", testCase.input, got, testCase.want)
		}
	}
}

func TestParseFilterPredicates(t *testing.T) {
	tests := []struct {
		input    string
		operator PredicateOperator
		negated  bool
		want     string
	}{
		{"a = 1", PredicateEQ, false, "a = 1"},
		{"a != 1", PredicateNE, false, "a != 1"},
		{"a < 1", PredicateLT, false, "a < 1"},
		{"a < = 1", PredicateLE, false, "a <= 1"},
		{"a > 1", PredicateGT, false, "a > 1"},
		{"a > = 1", PredicateGE, false, "a >= 1"},
		{"name LIKE 'A\\'b%'", PredicateLike, false, `name LIKE 'A\'b%'`},
		{"a IN (1, -2.5, true, 'x')", PredicateIn, false, "a IN (1, -2.5, true, 'x')"},
		{"a NOT IN (1)", PredicateIn, true, "a NOT IN (1)"},
		{"a CONTAIN_ALL ()", PredicateContainAll, false, "a CONTAIN_ALL ()"},
		{"a NOT CONTAIN_ANY ('x')", PredicateContainAny, true, "a NOT CONTAIN_ANY ('x')"},
		{"a IS NULL", PredicateIsNull, false, "a IS NULL"},
		{"a IS NOT NULL", PredicateIsNull, true, "a IS NOT NULL"},
	}
	for _, testCase := range tests {
		t.Run(testCase.input, func(t *testing.T) {
			expression, err := ParseFilter(testCase.input)
			if err != nil {
				t.Fatal(err)
			}
			predicate, ok := expression.(*PredicateExpr)
			if !ok || predicate.Operator != testCase.operator || predicate.Negated != testCase.negated {
				t.Fatalf("predicate = %#v", expression)
			}
			if got := Format(expression); got != testCase.want {
				t.Fatalf("format = %q, want %q", got, testCase.want)
			}
		})
	}
}

func TestParseFilterPreservesIdentifierAndLiteralForms(t *testing.T) {
	expression, err := ParseFilter(`1-dash_Field = 18446744073709551615 AND bool = FALSE AND text = "A\"b"`)
	if err != nil {
		t.Fatal(err)
	}
	root := expression.(*LogicalExpr)
	left := root.Left.(*LogicalExpr).Left.(*PredicateExpr)
	identifier := left.Left.(*IdentifierExpr)
	literal := left.Right.(*LiteralExpr)
	if identifier.Name != "1-dash_Field" || literal.Kind != LiteralInteger || literal.Text != "18446744073709551615" {
		t.Fatalf("preserved nodes = %#v, %#v", identifier, literal)
	}
	right := root.Right.(*PredicateExpr).Right.(*LiteralExpr)
	if right.Kind != LiteralString || right.Text != `A"b` {
		t.Fatalf("normalized string = %#v", right)
	}
}

func TestParseFilterFunctionCallsAndVectors(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"array_length(tags) >= 2", "array_length(tags) >= 2"},
		{"distance(vec, [1, -2.5]) < 0.5", "distance(vec, [1, -2.5]) < 0.5"},
		{"distance(vec, [[1, 2], [3, 4]]) = 0", "distance(vec, [[1, 2], [3, 4]]) = 0"},
		{"outer(inner(value), 1) = 2", "outer(inner(value), 1) = 2"},
	}
	for _, testCase := range tests {
		expression, err := ParseFilter(testCase.input)
		if err != nil {
			t.Fatalf("ParseFilter(%q): %v", testCase.input, err)
		}
		if got := Format(expression); got != testCase.want {
			t.Fatalf("format = %q, want %q", got, testCase.want)
		}
	}
	expression, _ := ParseFilter("distance(vec, [[1, 2], [3, 4]]) = 0")
	call := expression.(*PredicateExpr).Left.(*CallExpr)
	vector := call.Arguments[1].(*VectorExpr)
	if !vector.Matrix || len(vector.Rows) != 2 || len(vector.Rows[0]) != 2 {
		t.Fatalf("matrix AST = %#v", vector)
	}
}

func TestParseFilterKeywordIdentifiersMatchBaseline(t *testing.T) {
	for _, input := range []string{"and=1", "not=1", "where=1", "select=1", "order=1"} {
		if _, err := ParseFilter(input); err != nil {
			t.Fatalf("keyword identifier %q: %v", input, err)
		}
	}
	for _, input := range []string{"from=1", "true=1", "null=1", "contain_all=1"} {
		if _, err := ParseFilter(input); err == nil {
			t.Fatalf("reserved identifier %q succeeded", input)
		}
	}
}

func TestParseFilterErrorsCarryPosition(t *testing.T) {
	tests := []struct {
		input  string
		line   int
		column int
		offset int
	}{
		{input: "", line: 1, column: 1, offset: 0},
		{input: "a =", line: 1, column: 4, offset: 3},
		{input: "a IN ()", line: 1, column: 7, offset: 6},
		{input: "a NOT LIKE 'x'", line: 1, column: 7, offset: 6},
		{input: "a IS TRUE", line: 1, column: 6, offset: 5},
		{input: "(a=1", line: 1, column: 5, offset: 4},
		{input: "a=1 trailing", line: 1, column: 5, offset: 4},
		{input: "a=1\nAND", line: 2, column: 4, offset: 7},
	}
	for _, testCase := range tests {
		t.Run(testCase.input, func(t *testing.T) {
			_, err := ParseFilter(testCase.input)
			var parseErr *ParseError
			if !errors.As(err, &parseErr) {
				t.Fatalf("error = %T %v", err, err)
			}
			position := parseErr.Position
			if position.Line != testCase.line || position.Column != testCase.column || position.Offset != testCase.offset {
				t.Fatalf("position = %#v; error = %v", position, err)
			}
		})
	}
}

func TestParseFilterSpanAndNestingLimit(t *testing.T) {
	input := "alpha = 1 AND beta IS NOT NULL"
	expression, err := ParseFilter(input)
	if err != nil {
		t.Fatal(err)
	}
	want := Span{Start: Position{Offset: 0, Line: 1, Column: 1}, End: Position{Offset: len(input), Line: 1, Column: len(input) + 1}}
	if !reflect.DeepEqual(expression.NodeSpan(), want) {
		t.Fatalf("span = %#v, want %#v", expression.NodeSpan(), want)
	}
	deep := strings.Repeat("(", MaxParseDepth+1) + "a=1" + strings.Repeat(")", MaxParseDepth+1)
	if _, err := ParseFilter(deep); err == nil || !strings.Contains(err.Error(), "nesting") {
		t.Fatalf("deep parse error = %v", err)
	}
}

func TestParseFilterRejectsInvalidForms(t *testing.T) {
	inputs := []string{
		"a BETWEEN 1", "a == 1", "a IN (1,)", "a IN ([1])",
		"a LIKE", "a IS", "a CONTAIN_ANY (func(x))", "a = bare",
		"a = []", "a = [[1],]", "NOT a=1", "a=1;",
		"_=1", "-=1", "array_length(tags) LIKE '2%'",
		"array_length(tags) IN (2)", "array_length(tags) CONTAIN_ANY (2)",
		"array_length(tags) IS NULL",
	}
	for _, input := range inputs {
		if _, err := ParseFilter(input); err == nil {
			t.Fatalf("invalid filter %q succeeded", input)
		}
	}
}

func FuzzParseFilter(f *testing.F) {
	seeds := []string{
		"a=1", "a=1 OR b=2 AND c=3", "tags NOT IN ('a', 'b')",
		"array_length(tags) >= 2", "a IS NULL", "a CONTAIN_ANY ()",
		"distance(vec, [[1, 2], [3, 4]]) < 1", "a = 'unterminated",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		expression, err := ParseFilter(input)
		if err != nil {
			return
		}
		formatted := Format(expression)
		reparsed, err := ParseFilter(formatted)
		if err != nil {
			t.Fatalf("formatted successful AST does not parse: input=%q formatted=%q error=%v", input, formatted, err)
		}
		if Format(reparsed) != formatted {
			t.Fatalf("format is not stable: %q != %q", Format(reparsed), formatted)
		}
	})
}
