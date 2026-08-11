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

// Package sql parses and evaluates the SQL-style scalar filter language used
// by xvec.
package sql

import (
	"strings"
)

// Position identifies a UTF-8 source location. Offset is a zero-based byte
// offset; Line and Column are one-based Unicode code-point coordinates.
type Position struct {
	Offset int
	Line   int
	Column int
}

// Span is a half-open source range [Start, End).
type Span struct {
	Start Position
	End   Position
}

// Expr is any filter AST expression.
type Expr interface {
	NodeSpan() Span
	String() string
	exprNode()
}

// ValueExpr is a relation operand or function argument.
type ValueExpr interface {
	Expr
	valueExpr()
}

// LogicalOperator joins Boolean filter expressions.
type LogicalOperator uint8

const (
	LogicalAnd LogicalOperator = iota + 1
	LogicalOr
)

func (o LogicalOperator) String() string {
	if o == LogicalAnd {
		return "AND"
	}
	if o == LogicalOr {
		return "OR"
	}
	return "UNKNOWN"
}

// PredicateOperator identifies one relation predicate.
type PredicateOperator uint8

const (
	PredicateEQ PredicateOperator = iota + 1
	PredicateNE
	PredicateLT
	PredicateLE
	PredicateGT
	PredicateGE
	PredicateLike
	PredicateIn
	PredicateContainAll
	PredicateContainAny
	PredicateIsNull
)

func (o PredicateOperator) String() string {
	switch o {
	case PredicateEQ:
		return "="
	case PredicateNE:
		return "!="
	case PredicateLT:
		return "<"
	case PredicateLE:
		return "<="
	case PredicateGT:
		return ">"
	case PredicateGE:
		return ">="
	case PredicateLike:
		return "LIKE"
	case PredicateIn:
		return "IN"
	case PredicateContainAll:
		return "CONTAIN_ALL"
	case PredicateContainAny:
		return "CONTAIN_ANY"
	case PredicateIsNull:
		return "IS NULL"
	default:
		return "UNKNOWN"
	}
}

// LiteralKind distinguishes untyped syntax literals before schema analysis.
type LiteralKind uint8

const (
	LiteralInteger LiteralKind = iota + 1
	LiteralFloat
	LiteralString
	LiteralBool
)

func (k LiteralKind) String() string {
	switch k {
	case LiteralInteger:
		return "INTEGER"
	case LiteralFloat:
		return "FLOAT"
	case LiteralString:
		return "STRING"
	case LiteralBool:
		return "BOOL"
	default:
		return "UNKNOWN"
	}
}

// LogicalExpr is a binary AND or OR expression.
type LogicalExpr struct {
	Operator LogicalOperator
	Left     Expr
	Right    Expr
	Range    Span
}

func (*LogicalExpr) exprNode()        {}
func (e *LogicalExpr) NodeSpan() Span { return e.Range }
func (e *LogicalExpr) String() string { return Format(e) }

// PredicateExpr compares a field or function call with a value. Negated is
// valid for IN, CONTAIN_ALL, CONTAIN_ANY, and IS NULL.
type PredicateExpr struct {
	Operator PredicateOperator
	Left     ValueExpr
	Right    ValueExpr
	Negated  bool
	Range    Span
}

func (*PredicateExpr) exprNode()        {}
func (e *PredicateExpr) NodeSpan() Span { return e.Range }
func (e *PredicateExpr) String() string { return Format(e) }

// IdentifierExpr preserves the source spelling of a field or function name.
type IdentifierExpr struct {
	Name  string
	Range Span
}

func (*IdentifierExpr) exprNode()        {}
func (*IdentifierExpr) valueExpr()       {}
func (e *IdentifierExpr) NodeSpan() Span { return e.Range }
func (e *IdentifierExpr) String() string { return Format(e) }

// CallExpr is a function invocation. Arguments may be literals, identifiers,
// vectors, or nested calls; semantic validation happens in the analyzer.
type CallExpr struct {
	Name      string
	Arguments []ValueExpr
	Range     Span
}

func (*CallExpr) exprNode()        {}
func (*CallExpr) valueExpr()       {}
func (e *CallExpr) NodeSpan() Span { return e.Range }
func (e *CallExpr) String() string { return Format(e) }

// LiteralExpr retains a normalized literal Text and its original source Raw.
// Integer and float values intentionally remain textual until schema analysis
// so the full UINT64 range is representable.
type LiteralExpr struct {
	Kind  LiteralKind
	Text  string
	Raw   string
	Range Span
}

func (*LiteralExpr) exprNode()        {}
func (*LiteralExpr) valueExpr()       {}
func (e *LiteralExpr) NodeSpan() Span { return e.Range }
func (e *LiteralExpr) String() string { return Format(e) }

// ListExpr contains IN or contain literals. Parser rules keep IN lists
// non-empty while contain lists may be empty.
type ListExpr struct {
	Values []*LiteralExpr
	Range  Span
}

func (*ListExpr) exprNode()        {}
func (*ListExpr) valueExpr()       {}
func (e *ListExpr) NodeSpan() Span { return e.Range }
func (e *ListExpr) String() string { return Format(e) }

// VectorExpr preserves one vector or a matrix of numeric literal rows.
type VectorExpr struct {
	Rows   [][]*LiteralExpr
	Matrix bool
	Range  Span
}

func (*VectorExpr) exprNode()        {}
func (*VectorExpr) valueExpr()       {}
func (e *VectorExpr) NodeSpan() Span { return e.Range }
func (e *VectorExpr) String() string { return Format(e) }

// Format renders a deterministic normalized filter for diagnostics.
func Format(expression Expr) string {
	switch expression := expression.(type) {
	case *LogicalExpr:
		return "(" + Format(expression.Left) + " " + expression.Operator.String() + " " + Format(expression.Right) + ")"
	case *PredicateExpr:
		left := Format(expression.Left)
		switch expression.Operator {
		case PredicateIsNull:
			if expression.Negated {
				return left + " IS NOT NULL"
			}
			return left + " IS NULL"
		case PredicateIn, PredicateContainAll, PredicateContainAny:
			negated := ""
			if expression.Negated {
				negated = "NOT "
			}
			return left + " " + negated + expression.Operator.String() + " " + Format(expression.Right)
		default:
			return left + " " + expression.Operator.String() + " " + Format(expression.Right)
		}
	case *IdentifierExpr:
		return expression.Name
	case *CallExpr:
		arguments := make([]string, len(expression.Arguments))
		for index := range expression.Arguments {
			arguments[index] = Format(expression.Arguments[index])
		}
		return expression.Name + "(" + strings.Join(arguments, ", ") + ")"
	case *LiteralExpr:
		if expression.Kind == LiteralString {
			value := strings.ReplaceAll(expression.Text, `\`, `\\`)
			value = strings.ReplaceAll(value, `'`, `\'`)
			return "'" + value + "'"
		}
		return expression.Text
	case *ListExpr:
		values := make([]string, len(expression.Values))
		for index := range expression.Values {
			values[index] = Format(expression.Values[index])
		}
		return "(" + strings.Join(values, ", ") + ")"
	case *VectorExpr:
		rows := make([]string, len(expression.Rows))
		for rowIndex, row := range expression.Rows {
			values := make([]string, len(row))
			for index := range row {
				values[index] = Format(row[index])
			}
			rows[rowIndex] = "[" + strings.Join(values, ", ") + "]"
		}
		if expression.Matrix {
			return "[" + strings.Join(rows, ", ") + "]"
		}
		if len(rows) == 1 {
			return rows[0]
		}
		return "[]"
	default:
		return "<nil>"
	}
}
