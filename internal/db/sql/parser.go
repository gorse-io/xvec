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
	"strings"
)

const MaxParseDepth = 256

// ParseFilter parses one complete SQL-style filter expression.
func ParseFilter(input string) (Expr, error) {
	tokens, err := Lex(input)
	if err != nil {
		return nil, err
	}
	parser := filterParser{tokens: tokens}
	expression, err := parser.parseOr()
	if err != nil {
		return nil, err
	}
	if parser.current().Kind != TokenEOF {
		return nil, parser.unexpected("end of input")
	}
	return expression, nil
}

type filterParser struct {
	tokens []Token
	index  int
	depth  int
}

func (p *filterParser) parseOr() (Expr, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.match(TokenOr) {
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = &LogicalExpr{
			Operator: LogicalOr, Left: left, Right: right,
			Range: Span{Start: left.NodeSpan().Start, End: right.NodeSpan().End},
		}
	}
	return left, nil
}

func (p *filterParser) parseAnd() (Expr, error) {
	left, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	for p.match(TokenAnd) {
		right, err := p.parsePrimary()
		if err != nil {
			return nil, err
		}
		left = &LogicalExpr{
			Operator: LogicalAnd, Left: left, Right: right,
			Range: Span{Start: left.NodeSpan().Start, End: right.NodeSpan().End},
		}
	}
	return left, nil
}

func (p *filterParser) parsePrimary() (Expr, error) {
	if p.current().Kind != TokenLeftParen {
		return p.parsePredicate()
	}
	if err := p.enter(p.current().Span.Start); err != nil {
		return nil, err
	}
	defer p.leave()
	p.advance()
	expression, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	if _, err := p.expect(TokenRightParen); err != nil {
		return nil, err
	}
	return expression, nil
}

func (p *filterParser) parsePredicate() (Expr, error) {
	left, err := p.parseRelationLeft()
	if err != nil {
		return nil, err
	}
	start := left.NodeSpan().Start
	predicate := &PredicateExpr{Left: left}
	if _, functionCall := left.(*CallExpr); functionCall {
		switch p.current().Kind {
		case TokenEqual, TokenNotEqual, TokenLessEqual, TokenGreaterEqual, TokenLess, TokenGreater:
		default:
			return nil, p.unexpected("comparison after function call")
		}
	}
	switch p.current().Kind {
	case TokenEqual, TokenNotEqual, TokenLessEqual, TokenGreaterEqual:
		token := p.advance()
		switch token.Kind {
		case TokenEqual:
			predicate.Operator = PredicateEQ
		case TokenNotEqual:
			predicate.Operator = PredicateNE
		case TokenLessEqual:
			predicate.Operator = PredicateLE
		case TokenGreaterEqual:
			predicate.Operator = PredicateGE
		}
		predicate.Right, err = p.parseValue(false)
	case TokenLess, TokenGreater:
		token := p.advance()
		if p.match(TokenEqual) {
			if token.Kind == TokenLess {
				predicate.Operator = PredicateLE
			} else {
				predicate.Operator = PredicateGE
			}
		} else if token.Kind == TokenLess {
			predicate.Operator = PredicateLT
		} else {
			predicate.Operator = PredicateGT
		}
		predicate.Right, err = p.parseValue(false)
	case TokenLike:
		p.advance()
		predicate.Operator = PredicateLike
		predicate.Right, err = p.parseValue(false)
	case TokenNot:
		p.advance()
		predicate.Negated = true
		switch p.current().Kind {
		case TokenIn:
			p.advance()
			predicate.Operator = PredicateIn
			predicate.Right, err = p.parseList(false)
		case TokenContainAll, TokenContainAny:
			kind := p.advance().Kind
			predicate.Operator = PredicateContainAll
			if kind == TokenContainAny {
				predicate.Operator = PredicateContainAny
			}
			predicate.Right, err = p.parseList(true)
		default:
			return nil, p.unexpected("IN, CONTAIN_ALL, or CONTAIN_ANY after NOT")
		}
	case TokenIn:
		p.advance()
		predicate.Operator = PredicateIn
		predicate.Right, err = p.parseList(false)
	case TokenContainAll, TokenContainAny:
		kind := p.advance().Kind
		predicate.Operator = PredicateContainAll
		if kind == TokenContainAny {
			predicate.Operator = PredicateContainAny
		}
		predicate.Right, err = p.parseList(true)
	case TokenIs:
		p.advance()
		predicate.Operator = PredicateIsNull
		predicate.Negated = p.match(TokenNot)
		if _, expectErr := p.expect(TokenNull); expectErr != nil {
			return nil, expectErr
		}
		predicate.Range = Span{Start: start, End: p.previous().Span.End}
		return predicate, nil
	default:
		return nil, p.unexpected("comparison, LIKE, [NOT] IN, [NOT] CONTAIN_ALL/ANY, or IS [NOT] NULL")
	}
	if err != nil {
		return nil, err
	}
	predicate.Range = Span{Start: start, End: predicate.Right.NodeSpan().End}
	return predicate, nil
}

func (p *filterParser) parseRelationLeft() (ValueExpr, error) {
	identifier, err := p.parseIdentifier()
	if err != nil {
		return nil, err
	}
	if p.current().Kind != TokenLeftParen {
		return identifier, nil
	}
	return p.parseCall(identifier)
}

func (p *filterParser) parseValue(allowIdentifier bool) (ValueExpr, error) {
	token := p.current()
	switch token.Kind {
	case TokenInteger, TokenFloat, TokenString, TokenTrue, TokenFalse:
		p.advance()
		return literalFromToken(token), nil
	case TokenLeftBracket:
		return p.parseVector()
	default:
		if !identifierTokenAllowed(token.Kind) {
			return nil, p.unexpected("literal or function call")
		}
		identifier, err := p.parseIdentifier()
		if err != nil {
			return nil, err
		}
		if p.current().Kind == TokenLeftParen {
			return p.parseCall(identifier)
		}
		if allowIdentifier {
			return identifier, nil
		}
		return nil, parseError(identifier.Range.Start, "bare identifier %q is not a relation value", identifier.Name)
	}
}

func (p *filterParser) parseIdentifier() (*IdentifierExpr, error) {
	token := p.current()
	if !identifierTokenAllowed(token.Kind) {
		return nil, p.unexpected("identifier")
	}
	p.advance()
	return &IdentifierExpr{Name: token.Text, Range: token.Span}, nil
}

func (p *filterParser) parseCall(identifier *IdentifierExpr) (*CallExpr, error) {
	if err := p.enter(identifier.Range.Start); err != nil {
		return nil, err
	}
	defer p.leave()
	if _, err := p.expect(TokenLeftParen); err != nil {
		return nil, err
	}
	call := &CallExpr{Name: identifier.Name}
	if p.current().Kind != TokenRightParen {
		for {
			argument, err := p.parseValue(true)
			if err != nil {
				return nil, err
			}
			call.Arguments = append(call.Arguments, argument)
			if !p.match(TokenComma) {
				break
			}
		}
	}
	right, err := p.expect(TokenRightParen)
	if err != nil {
		return nil, err
	}
	call.Range = Span{Start: identifier.Range.Start, End: right.Span.End}
	return call, nil
}

func (p *filterParser) parseList(allowEmpty bool) (*ListExpr, error) {
	left, err := p.expect(TokenLeftParen)
	if err != nil {
		return nil, err
	}
	list := &ListExpr{}
	if p.current().Kind == TokenRightParen {
		if !allowEmpty {
			return nil, p.unexpected("at least one IN value")
		}
		right := p.advance()
		list.Range = Span{Start: left.Span.Start, End: right.Span.End}
		return list, nil
	}
	for {
		token := p.current()
		switch token.Kind {
		case TokenInteger, TokenFloat, TokenString, TokenTrue, TokenFalse:
			p.advance()
			list.Values = append(list.Values, literalFromToken(token))
		default:
			return nil, p.unexpected("numeric, string, or Boolean list value")
		}
		if !p.match(TokenComma) {
			break
		}
	}
	right, err := p.expect(TokenRightParen)
	if err != nil {
		return nil, err
	}
	list.Range = Span{Start: left.Span.Start, End: right.Span.End}
	return list, nil
}

func (p *filterParser) parseVector() (*VectorExpr, error) {
	if err := p.enter(p.current().Span.Start); err != nil {
		return nil, err
	}
	defer p.leave()
	left, err := p.expect(TokenLeftBracket)
	if err != nil {
		return nil, err
	}
	vector := &VectorExpr{}
	if p.current().Kind == TokenLeftBracket {
		vector.Matrix = true
		for {
			row, _, rowErr := p.parseVectorRow()
			if rowErr != nil {
				return nil, rowErr
			}
			vector.Rows = append(vector.Rows, row)
			if !p.match(TokenComma) {
				break
			}
		}
		right, expectErr := p.expect(TokenRightBracket)
		if expectErr != nil {
			return nil, expectErr
		}
		vector.Range = Span{Start: left.Span.Start, End: right.Span.End}
		return vector, nil
	}
	row, right, err := p.parseVectorElements(TokenRightBracket)
	if err != nil {
		return nil, err
	}
	vector.Rows = [][]*LiteralExpr{row}
	vector.Range = Span{Start: left.Span.Start, End: right.Span.End}
	return vector, nil
}

func (p *filterParser) parseVectorRow() ([]*LiteralExpr, Token, error) {
	if _, err := p.expect(TokenLeftBracket); err != nil {
		return nil, Token{}, err
	}
	return p.parseVectorElements(TokenRightBracket)
}

func (p *filterParser) parseVectorElements(end TokenKind) ([]*LiteralExpr, Token, error) {
	if p.current().Kind == end {
		return nil, Token{}, p.unexpected("at least one vector number")
	}
	var values []*LiteralExpr
	for {
		token := p.current()
		if token.Kind != TokenInteger && token.Kind != TokenFloat {
			return nil, Token{}, p.unexpected("numeric vector value")
		}
		p.advance()
		values = append(values, literalFromToken(token))
		if !p.match(TokenComma) {
			break
		}
	}
	right, err := p.expect(end)
	return values, right, err
}

func literalFromToken(token Token) *LiteralExpr {
	literal := &LiteralExpr{Raw: token.Text, Text: token.Text, Range: token.Span}
	switch token.Kind {
	case TokenInteger:
		literal.Kind = LiteralInteger
	case TokenFloat:
		literal.Kind = LiteralFloat
	case TokenString:
		literal.Kind = LiteralString
		literal.Text = normalizeQuotedString(token.Text)
	case TokenTrue:
		literal.Kind = LiteralBool
		literal.Text = "true"
	case TokenFalse:
		literal.Kind = LiteralBool
		literal.Text = "false"
	}
	return literal
}

func normalizeQuotedString(raw string) string {
	if len(raw) >= 2 {
		raw = raw[1 : len(raw)-1]
	}
	raw = strings.ReplaceAll(raw, `\"`, `"`)
	return strings.ReplaceAll(raw, `\'`, `'`)
}

func identifierTokenAllowed(kind TokenKind) bool {
	switch kind {
	case TokenIdentifier, TokenOr, TokenAnd, TokenNot, TokenIn, TokenBetween,
		TokenLike, TokenWhere, TokenSelect, TokenAs, TokenBy, TokenOrder,
		TokenAsc, TokenDesc, TokenLimit:
		return true
	default:
		return false
	}
}

func (p *filterParser) current() Token {
	if p.index >= len(p.tokens) {
		return p.tokens[len(p.tokens)-1]
	}
	return p.tokens[p.index]
}

func (p *filterParser) previous() Token {
	if p.index == 0 {
		return p.current()
	}
	return p.tokens[p.index-1]
}

func (p *filterParser) advance() Token {
	token := p.current()
	if token.Kind != TokenEOF {
		p.index++
	}
	return token
}

func (p *filterParser) match(kind TokenKind) bool {
	if p.current().Kind != kind {
		return false
	}
	p.advance()
	return true
}

func (p *filterParser) expect(kind TokenKind) (Token, error) {
	if p.current().Kind != kind {
		return Token{}, p.unexpected(kind.String())
	}
	return p.advance(), nil
}

func (p *filterParser) unexpected(expected string) error {
	token := p.current()
	if token.Kind == TokenEOF {
		return parseError(token.Span.Start, "expected %s, found end of input", expected)
	}
	return parseError(token.Span.Start, "expected %s, found %s %q", expected, token.Kind, token.Text)
}

func (p *filterParser) enter(position Position) error {
	p.depth++
	if p.depth > MaxParseDepth {
		p.depth--
		return parseError(position, "filter nesting exceeds %d", MaxParseDepth)
	}
	return nil
}

func (p *filterParser) leave() { p.depth-- }

var _ error = (*ParseError)(nil)
