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
	"strings"

	antlrRuntime "github.com/antlr4-go/antlr/v4"
	sqlantlr "github.com/gorse-io/xvec/internal/db/sqlengine/antlr"
)

const MaxParseDepth = 256

// ParseFilter parses one complete SQL-style filter expression with the
// generated ANTLR parser and converts its parse tree into xvec's filter AST.
func ParseFilter(input string) (Expr, error) {
	positions, err := validateFilter(input)
	if err != nil {
		return nil, err
	}

	inputStream := antlrRuntime.NewInputStream(input)
	lexer := sqlantlr.NewSQLLexer(inputStream)
	errors := newSyntaxErrorListener(positions)
	lexer.RemoveErrorListeners()
	lexer.AddErrorListener(errors)

	tokens := antlrRuntime.NewCommonTokenStream(lexer, antlrRuntime.TokenDefaultChannel)
	tokens.Fill()
	if errors.err != nil {
		return nil, errors.err
	}
	if err := checkParseDepth(tokens.GetAllTokens(), positions); err != nil {
		return nil, err
	}
	tokens.Seek(0)

	parser := sqlantlr.NewSQLParser(tokens)
	parser.RemoveErrorListeners()
	parser.AddErrorListener(errors)
	tree := parser.Logic_expr_unit()
	if errors.err != nil {
		return nil, errors.err
	}
	return buildLogicExpr(tree.Logic_expr(), positions), nil
}

func checkParseDepth(tokens []antlrRuntime.Token, positions *sourcePositions) error {
	depth := 0
	for _, token := range tokens {
		switch token.GetTokenType() {
		case sqlantlr.SQLLexerLP:
			depth++
			if depth > MaxParseDepth {
				return parseError(positions.atRune(token.GetStart()), "filter nesting exceeds %d", MaxParseDepth)
			}
		case sqlantlr.SQLLexerRP:
			if depth > 0 {
				depth--
			}
		}
	}
	return nil
}

func buildLogicExpr(context sqlantlr.ILogic_exprContext, positions *sourcePositions) Expr {
	return buildOrExpr(context.Or_expr(), positions)
}

func buildOrExpr(context sqlantlr.IOr_exprContext, positions *sourcePositions) Expr {
	operands := context.AllAnd_expr()
	left := buildAndExpr(operands[0], positions)
	for _, operand := range operands[1:] {
		right := buildAndExpr(operand, positions)
		left = &LogicalExpr{
			Operator: LogicalOr,
			Left:     left,
			Right:    right,
			Range:    Span{Start: left.NodeSpan().Start, End: right.NodeSpan().End},
		}
	}
	return left
}

func buildAndExpr(context sqlantlr.IAnd_exprContext, positions *sourcePositions) Expr {
	operands := context.AllPrimary_expr()
	left := buildPrimaryExpr(operands[0], positions)
	for _, operand := range operands[1:] {
		right := buildPrimaryExpr(operand, positions)
		left = &LogicalExpr{
			Operator: LogicalAnd,
			Left:     left,
			Right:    right,
			Range:    Span{Start: left.NodeSpan().Start, End: right.NodeSpan().End},
		}
	}
	return left
}

func buildPrimaryExpr(context sqlantlr.IPrimary_exprContext, positions *sourcePositions) Expr {
	if relation := context.Relation_expr(); relation != nil {
		return buildRelationExpr(relation, positions)
	}
	return buildLogicExpr(context.Logic_expr(), positions)
}

func buildRelationExpr(context sqlantlr.IRelation_exprContext, positions *sourcePositions) Expr {
	var left ValueExpr
	if call := context.Function_call(); call != nil {
		left = buildFunctionCall(call, positions)
	} else {
		left = buildIdentifier(context.Identifier(), positions)
	}

	predicate := &PredicateExpr{
		Left:    left,
		Negated: context.NOT() != nil,
		Range:   ruleSpan(context, positions),
	}
	switch {
	case context.Rel_oper() != nil:
		predicate.Operator = relationOperator(context.Rel_oper().GetText())
		predicate.Right = buildValueExpr(context.Value_expr(), positions)
	case context.LIKE() != nil:
		predicate.Operator = PredicateLike
		predicate.Right = buildValueExpr(context.Value_expr(), positions)
	case context.IN() != nil:
		predicate.Operator = PredicateIn
		predicate.Right = buildListExpr(context, positions)
	case context.CONTAIN_ALL() != nil:
		predicate.Operator = PredicateContainAll
		predicate.Right = buildListExpr(context, positions)
	case context.CONTAIN_ANY() != nil:
		predicate.Operator = PredicateContainAny
		predicate.Right = buildListExpr(context, positions)
	case context.IS() != nil:
		predicate.Operator = PredicateIsNull
	}
	return predicate
}

func relationOperator(text string) PredicateOperator {
	switch strings.ReplaceAll(text, " ", "") {
	case "=":
		return PredicateEQ
	case "!=":
		return PredicateNE
	case "<":
		return PredicateLT
	case "<=":
		return PredicateLE
	case ">":
		return PredicateGT
	case ">=":
		return PredicateGE
	default:
		return 0
	}
}

func buildValueExpr(context sqlantlr.IValue_exprContext, positions *sourcePositions) ValueExpr {
	if call := context.Function_call(); call != nil {
		return buildFunctionCall(call, positions)
	}
	return buildConstant(context.Constant(), positions)
}

func buildConstant(context sqlantlr.IConstantContext, positions *sourcePositions) ValueExpr {
	switch {
	case context.Numeric() != nil:
		return buildNumeric(context.Numeric(), positions)
	case context.Quoted_string() != nil:
		return buildQuotedString(context.Quoted_string(), positions)
	case context.Bool_value() != nil:
		return buildBool(context.Bool_value(), positions)
	default:
		return buildVectorExpr(context.Vector_expr(), positions)
	}
}

func buildListExpr(context sqlantlr.IRelation_exprContext, positions *sourcePositions) *ListExpr {
	list := &ListExpr{
		Range: Span{
			Start: positions.tokenSpan(context.LP().GetSymbol()).Start,
			End:   positions.tokenSpan(context.RP().GetSymbol()).End,
		},
	}
	if values := context.In_value_expr_list(); values != nil {
		for _, value := range values.AllIn_value_expr() {
			switch {
			case value.Numeric() != nil:
				list.Values = append(list.Values, buildNumeric(value.Numeric(), positions))
			case value.Quoted_string() != nil:
				list.Values = append(list.Values, buildQuotedString(value.Quoted_string(), positions))
			default:
				list.Values = append(list.Values, buildBool(value.Bool_value(), positions))
			}
		}
	}
	return list
}

func buildVectorExpr(context sqlantlr.IVector_exprContext, positions *sourcePositions) *VectorExpr {
	vectors := context.AllVector()
	result := &VectorExpr{
		Matrix: len(vectors) > 1 || context.LMP() != nil,
		Range:  ruleSpan(context, positions),
	}
	for _, vector := range vectors {
		var row []*LiteralExpr
		for _, numeric := range vector.AllNumeric() {
			row = append(row, buildNumeric(numeric, positions))
		}
		result.Rows = append(result.Rows, row)
	}
	return result
}

func buildFunctionCall(context sqlantlr.IFunction_callContext, positions *sourcePositions) *CallExpr {
	call := &CallExpr{
		Name:  context.Identifier().GetText(),
		Range: ruleSpan(context, positions),
	}
	for _, argument := range context.AllFunction_value_expr() {
		if value := argument.Value_expr(); value != nil {
			call.Arguments = append(call.Arguments, buildValueExpr(value, positions))
		} else {
			call.Arguments = append(call.Arguments, buildIdentifier(argument.Identifier(), positions))
		}
	}
	return call
}

func buildIdentifier(context sqlantlr.IIdentifierContext, positions *sourcePositions) *IdentifierExpr {
	return &IdentifierExpr{Name: context.GetText(), Range: ruleSpan(context, positions)}
}

func buildNumeric(context sqlantlr.INumericContext, positions *sourcePositions) *LiteralExpr {
	token := context.GetStart()
	kind := LiteralInteger
	if token.GetTokenType() == sqlantlr.SQLLexerFLOAT {
		kind = LiteralFloat
	}
	return &LiteralExpr{
		Kind:  kind,
		Text:  token.GetText(),
		Raw:   token.GetText(),
		Range: positions.tokenSpan(token),
	}
}

func buildQuotedString(context sqlantlr.IQuoted_stringContext, positions *sourcePositions) *LiteralExpr {
	token := context.GetStart()
	return &LiteralExpr{
		Kind:  LiteralString,
		Text:  normalizeQuotedString(token.GetText()),
		Raw:   token.GetText(),
		Range: positions.tokenSpan(token),
	}
}

func buildBool(context sqlantlr.IBool_valueContext, positions *sourcePositions) *LiteralExpr {
	token := context.GetStart()
	return &LiteralExpr{
		Kind:  LiteralBool,
		Text:  strings.ToLower(token.GetText()),
		Raw:   token.GetText(),
		Range: positions.tokenSpan(token),
	}
}

func normalizeQuotedString(raw string) string {
	if len(raw) >= 2 {
		raw = raw[1 : len(raw)-1]
	}
	raw = strings.ReplaceAll(raw, `\"`, `"`)
	return strings.ReplaceAll(raw, `\'`, `'`)
}

func ruleSpan(context antlrRuntime.ParserRuleContext, positions *sourcePositions) Span {
	return Span{
		Start: positions.tokenSpan(context.GetStart()).Start,
		End:   positions.tokenSpan(context.GetStop()).End,
	}
}

var _ error = (*ParseError)(nil)
