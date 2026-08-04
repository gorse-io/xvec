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

package core

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var (
	// ErrFTSQuerySyntax identifies a query that does not satisfy the FTS
	// expression grammar.
	ErrFTSQuerySyntax = errors.New("core: invalid FTS query syntax")
	// ErrUnsupportedFTSQuery identifies recognized baseline syntax that the
	// pinned execution surface intentionally rejects, currently field prefixes
	// and boosts.
	ErrUnsupportedFTSQuery = errors.New("core: unsupported FTS query")
)

// FTSQueryParseError reports a byte/rune source location and supports
// errors.Is through Kind.
type FTSQueryParseError struct {
	Kind    error
	Offset  uint32
	Line    uint32
	Column  uint32
	Message string
}

func (e *FTSQueryParseError) Error() string {
	if e == nil {
		return "<nil>"
	}
	kind := e.Kind
	if kind == nil {
		kind = ErrFTSQuerySyntax
	}
	if e.Message == "" {
		return fmt.Sprintf("%v at %d:%d", kind, e.Line, e.Column)
	}
	return fmt.Sprintf("%v at %d:%d: %s", kind, e.Line, e.Column, e.Message)
}

// Unwrap exposes the error category to errors.Is.
func (e *FTSQueryParseError) Unwrap() error {
	if e == nil || e.Kind == nil {
		return ErrFTSQuerySyntax
	}
	return e.Kind
}

// FTSDefaultOperator controls how adjacent query atoms and multi-token bare
// terms are combined.
type FTSDefaultOperator uint8

const (
	FTSDefaultOperatorOR FTSDefaultOperator = iota + 1
	FTSDefaultOperatorAND
)

func (o FTSDefaultOperator) String() string {
	switch o {
	case FTSDefaultOperatorOR:
		return "OR"
	case FTSDefaultOperatorAND:
		return "AND"
	default:
		return "UNKNOWN(" + strconv.FormatUint(uint64(o), 10) + ")"
	}
}

// ParseFTSDefaultOperator parses the public empty/OR/AND spelling. Empty means
// OR for baseline compatibility.
func ParseFTSDefaultOperator(value string) (FTSDefaultOperator, error) {
	switch strings.ToUpper(value) {
	case "", "OR":
		return FTSDefaultOperatorOR, nil
	case "AND":
		return FTSDefaultOperatorAND, nil
	default:
		return 0, fmt.Errorf("%w: default operator must be OR or AND", ErrInvalidFTSQuery)
	}
}

// AnalyzeFTSMatchQuery analyzes natural-language text without interpreting
// boolean syntax. Multiple tokens use defaultOperator; an empty analyzed stream
// returns FTSEmptyQueryNode.
func AnalyzeFTSMatchQuery(ctx context.Context, text string, analyzer FTSAnalyzer, defaultOperator FTSDefaultOperator) (FTSQueryNode, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil match-query context", ErrInvalidFTSQuery)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if ftsNilInterface(analyzer) {
		return nil, fmt.Errorf("%w: analyzer is required", ErrInvalidFTSQuery)
	}
	if defaultOperator != FTSDefaultOperatorOR && defaultOperator != FTSDefaultOperatorAND {
		return nil, fmt.Errorf("%w: unknown default operator %d", ErrInvalidFTSQuery, defaultOperator)
	}
	tokens, err := analyzer.Analyze(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("analyze FTS match query: %w", err)
	}
	if len(tokens) > MaxFTSQueryTokens {
		return nil, &FTSQueryParseError{
			Kind: ErrFTSQueryTooComplex, Offset: 0, Line: 1, Column: 0,
			Message: "analyzed token limit exceeded",
		}
	}
	node, err := buildAnalyzedFTSTerm(ctx, tokens, defaultOperator)
	if err != nil {
		return nil, err
	}
	if node == nil {
		return &FTSEmptyQueryNode{Flags: defaultFTSQueryModifier()}, nil
	}
	return node, nil
}

// ParseFTSQuery lexes, parses, and analyzes query. A syntactically valid query
// whose bare terms are all removed by analysis returns FTSEmptyQueryNode.
func ParseFTSQuery(ctx context.Context, query string, analyzer FTSAnalyzer, defaultOperator FTSDefaultOperator) (FTSQueryNode, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil parser context", ErrInvalidFTSQuery)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if ftsNilInterface(analyzer) {
		return nil, fmt.Errorf("%w: analyzer is required", ErrInvalidFTSQuery)
	}
	if defaultOperator != FTSDefaultOperatorOR && defaultOperator != FTSDefaultOperatorAND {
		return nil, fmt.Errorf("%w: unknown default operator %d", ErrInvalidFTSQuery, defaultOperator)
	}
	tokens, err := LexFTSQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	parser := ftsQueryParser{tokens: tokens}
	raw, err := parser.parseOr(0)
	if err != nil {
		return nil, err
	}
	if parser.current().Type != FTSQueryTokenEOF {
		return nil, parser.syntax(parser.current(), "unexpected "+parser.current().Type.String())
	}
	analyzedTokens := 0
	node, err := buildFTSQueryAST(ctx, raw, analyzer, defaultOperator, &analyzedTokens)
	if err != nil {
		return nil, err
	}
	if node == nil {
		return &FTSEmptyQueryNode{Flags: defaultFTSQueryModifier()}, nil
	}
	return node, nil
}

type ftsQueryParser struct {
	tokens []FTSQueryToken
	index  int
}

type ftsRawQueryNodeType uint8

const (
	ftsRawQueryTerm ftsRawQueryNodeType = iota + 1
	ftsRawQueryPhrase
	ftsRawQuerySequence
	ftsRawQueryAnd
	ftsRawQueryOr
)

type ftsRawQueryNode struct {
	type_       ftsRawQueryNodeType
	token       FTSQueryToken
	text        string
	children    []*ftsRawQueryNode
	must        bool
	mustNot     bool
	fieldPrefix bool
	boost       bool
	boostToken  FTSQueryToken
}

func (p *ftsQueryParser) parseOr(depth int) (*ftsRawQueryNode, error) {
	first, err := p.parseAnd(depth)
	if err != nil {
		return nil, err
	}
	children := []*ftsRawQueryNode{first}
	hadOperator := false
	for p.current().Type == FTSQueryTokenOR {
		hadOperator = true
		p.advance()
		child, err := p.parseAnd(depth)
		if err != nil {
			return nil, err
		}
		children = append(children, child)
	}
	if !hadOperator {
		return first, nil
	}
	return &ftsRawQueryNode{type_: ftsRawQueryOr, token: first.token, children: children}, nil
}

func (p *ftsQueryParser) parseAnd(depth int) (*ftsRawQueryNode, error) {
	first, err := p.parseSequence(depth)
	if err != nil {
		return nil, err
	}
	children := []*ftsRawQueryNode{first}
	hadOperator := false
	for p.current().Type == FTSQueryTokenAND || p.current().Type == FTSQueryTokenNOT {
		hadOperator = true
		mustNot := p.current().Type == FTSQueryTokenNOT
		p.advance()
		if !mustNot && p.current().Type == FTSQueryTokenNOT {
			mustNot = true
			p.advance()
		}
		child, err := p.parseSequence(depth)
		if err != nil {
			return nil, err
		}
		child.mustNot = child.mustNot || mustNot
		children = append(children, child)
	}
	if !hadOperator {
		return first, nil
	}
	return &ftsRawQueryNode{type_: ftsRawQueryAnd, token: first.token, children: children}, nil
}

func (p *ftsQueryParser) parseSequence(depth int) (*ftsRawQueryNode, error) {
	if !ftsCanStartUnary(p.current().Type) {
		return nil, p.syntax(p.current(), "expected a term, phrase, modifier, or '('")
	}
	children := make([]*ftsRawQueryNode, 0, 2)
	for ftsCanStartUnary(p.current().Type) {
		child, err := p.parseUnary(depth)
		if err != nil {
			return nil, err
		}
		children = append(children, child)
	}
	if len(children) == 1 {
		return children[0], nil
	}
	return &ftsRawQueryNode{type_: ftsRawQuerySequence, token: children[0].token, children: children}, nil
}

func (p *ftsQueryParser) parseUnary(depth int) (*ftsRawQueryNode, error) {
	must, mustNot := false, false
	switch p.current().Type {
	case FTSQueryTokenPlus:
		must = true
		p.advance()
	case FTSQueryTokenMinus:
		mustNot = true
		p.advance()
	}
	if !ftsCanStartAtom(p.current().Type) {
		return nil, p.syntax(p.current(), "expected an atom after modifier")
	}
	node, err := p.parseAtom(depth)
	if err != nil {
		return nil, err
	}
	node.must = node.must || must
	node.mustNot = node.mustNot || mustNot
	return node, nil
}

func (p *ftsQueryParser) parseAtom(depth int) (*ftsRawQueryNode, error) {
	start := p.current()
	fieldPrefix := false
	if p.current().Type == FTSQueryTokenRegularID && p.peek(1).Type == FTSQueryTokenColon {
		fieldPrefix = true
		p.advance()
		p.advance()
	}

	node, err := p.parsePrimary(depth)
	if err != nil {
		return nil, err
	}
	if p.current().Type == FTSQueryTokenCaret {
		node.boostToken = p.current()
		p.advance()
		if p.current().Type != FTSQueryTokenNumber {
			return nil, p.syntax(p.current(), "expected a number after '^'")
		}
		node.boost = true
		p.advance()
	}
	node.fieldPrefix = fieldPrefix
	if fieldPrefix {
		node.token = start
	}
	return node, nil
}

func (p *ftsQueryParser) parsePrimary(depth int) (*ftsRawQueryNode, error) {
	token := p.current()
	switch token.Type {
	case FTSQueryTokenTerm, FTSQueryTokenRegularID, FTSQueryTokenNumber:
		p.advance()
		return &ftsRawQueryNode{type_: ftsRawQueryTerm, token: token, text: ftsUnescapeQuery(token.Text)}, nil
	case FTSQueryTokenDefault:
		var builder strings.Builder
		for p.current().Type == FTSQueryTokenDefault {
			builder.WriteString(p.current().Text)
			p.advance()
		}
		return &ftsRawQueryNode{type_: ftsRawQueryTerm, token: token, text: ftsUnescapeQuery(builder.String())}, nil
	case FTSQueryTokenPhrase:
		p.advance()
		body := token.Text[1 : len(token.Text)-1]
		return &ftsRawQueryNode{type_: ftsRawQueryPhrase, token: token, text: ftsUnescapeQuery(body)}, nil
	case FTSQueryTokenLeftParen:
		if depth >= MaxFTSQueryDepth {
			return nil, &FTSQueryParseError{
				Kind: ErrFTSQueryTooComplex, Offset: token.Offset, Line: token.Line,
				Column: token.Column, Message: "parenthesis depth limit exceeded",
			}
		}
		p.advance()
		node, err := p.parseOr(depth + 1)
		if err != nil {
			return nil, err
		}
		if p.current().Type != FTSQueryTokenRightParen {
			return nil, p.syntax(p.current(), "expected ')'")
		}
		p.advance()
		return node, nil
	default:
		return nil, p.syntax(token, "expected a term, phrase, or '('")
	}
}

func buildFTSQueryAST(ctx context.Context, raw *ftsRawQueryNode, analyzer FTSAnalyzer, defaultOperator FTSDefaultOperator, analyzedTokenCount *int) (FTSQueryNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if raw.fieldPrefix {
		return nil, &FTSQueryParseError{
			Kind: ErrUnsupportedFTSQuery, Offset: raw.token.Offset, Line: raw.token.Line,
			Column: raw.token.Column, Message: "field-prefixed queries are not supported",
		}
	}
	if raw.boost {
		return nil, &FTSQueryParseError{
			Kind: ErrUnsupportedFTSQuery, Offset: raw.boostToken.Offset, Line: raw.boostToken.Line,
			Column: raw.boostToken.Column, Message: "boost queries are not supported",
		}
	}

	var node FTSQueryNode
	switch raw.type_ {
	case ftsRawQueryTerm:
		tokens, err := analyzeFTSQueryText(ctx, analyzer, raw.token, raw.text, analyzedTokenCount)
		if err != nil {
			return nil, err
		}
		node, err = buildAnalyzedFTSTerm(ctx, tokens, defaultOperator)
		if err != nil {
			return nil, err
		}
	case ftsRawQueryPhrase:
		tokens, err := analyzeFTSQueryText(ctx, analyzer, raw.token, raw.text, analyzedTokenCount)
		if err != nil {
			return nil, err
		}
		terms := make([]string, len(tokens))
		for index := range tokens {
			if index&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			terms[index] = strings.Clone(tokens[index].Text)
		}
		node = &FTSPhraseQueryNode{Flags: defaultFTSQueryModifier(), Terms: terms}
	case ftsRawQuerySequence, ftsRawQueryAnd, ftsRawQueryOr:
		children := make([]FTSQueryNode, 0, len(raw.children))
		for _, rawChild := range raw.children {
			child, err := buildFTSQueryAST(ctx, rawChild, analyzer, defaultOperator, analyzedTokenCount)
			if err != nil {
				return nil, err
			}
			if child != nil {
				children = append(children, child)
			}
		}
		switch raw.type_ {
		case ftsRawQueryAnd:
			if len(children) == 0 {
				return nil, nil
			}
			if len(children) == 1 {
				node = children[0]
			} else {
				node = &FTSAndQueryNode{Flags: defaultFTSQueryModifier(), Children: children}
			}
		case ftsRawQueryOr:
			if len(children) == 1 {
				node = children[0]
			} else {
				node = &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: children}
			}
		case ftsRawQuerySequence:
			if len(children) == 1 {
				node = children[0]
			} else if defaultOperator == FTSDefaultOperatorAND {
				node = &FTSAndQueryNode{Flags: defaultFTSQueryModifier(), Children: children}
			} else {
				node = &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: children}
			}
		}
	default:
		return nil, fmt.Errorf("%w: unknown raw node type %d", ErrInvalidFTSQuery, raw.type_)
	}
	applyFTSQueryModifier(node, raw.must, raw.mustNot)
	return node, nil
}

func buildAnalyzedFTSTerm(ctx context.Context, tokens []Token, defaultOperator FTSDefaultOperator) (FTSQueryNode, error) {
	if len(tokens) == 0 {
		return nil, nil
	}
	if len(tokens) == 1 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: strings.Clone(tokens[0].Text)}, nil
	}
	children := make([]FTSQueryNode, len(tokens))
	for index := range tokens {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		children[index] = &FTSTermQueryNode{Flags: defaultFTSQueryModifier(), Term: strings.Clone(tokens[index].Text)}
	}
	if defaultOperator == FTSDefaultOperatorAND {
		return &FTSAndQueryNode{Flags: defaultFTSQueryModifier(), Children: children}, nil
	}
	return &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: children}, nil
}

func analyzeFTSQueryText(ctx context.Context, analyzer FTSAnalyzer, token FTSQueryToken, text string, analyzedTokenCount *int) ([]Token, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	tokens, err := analyzer.Analyze(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("analyze FTS query at %d:%d: %w", token.Line, token.Column, err)
	}
	if len(tokens) > MaxFTSQueryTokens-*analyzedTokenCount {
		return nil, &FTSQueryParseError{
			Kind: ErrFTSQueryTooComplex, Offset: token.Offset, Line: token.Line,
			Column: token.Column, Message: "analyzed token limit exceeded",
		}
	}
	*analyzedTokenCount += len(tokens)
	return tokens, nil
}

func (p *ftsQueryParser) current() FTSQueryToken {
	return p.peek(0)
}

func (p *ftsQueryParser) peek(distance int) FTSQueryToken {
	index := p.index + distance
	if index < 0 || index >= len(p.tokens) {
		return FTSQueryToken{Type: FTSQueryTokenEOF}
	}
	return p.tokens[index]
}

func (p *ftsQueryParser) advance() {
	if p.index < len(p.tokens)-1 {
		p.index++
	}
}

func (p *ftsQueryParser) syntax(token FTSQueryToken, message string) error {
	return &FTSQueryParseError{
		Kind: ErrFTSQuerySyntax, Offset: token.Offset, Line: token.Line,
		Column: token.Column, Message: message,
	}
}

func ftsCanStartUnary(tokenType FTSQueryTokenType) bool {
	return tokenType == FTSQueryTokenPlus || tokenType == FTSQueryTokenMinus || ftsCanStartAtom(tokenType)
}

func ftsCanStartAtom(tokenType FTSQueryTokenType) bool {
	switch tokenType {
	case FTSQueryTokenTerm, FTSQueryTokenRegularID, FTSQueryTokenNumber,
		FTSQueryTokenDefault, FTSQueryTokenPhrase, FTSQueryTokenLeftParen:
		return true
	default:
		return false
	}
}

func ftsUnescapeQuery(text string) string {
	write := 0
	buffer := []byte(text)
	for read := 0; read < len(buffer); read++ {
		if buffer[read] == '\\' && read+1 < len(buffer) {
			read++
		}
		buffer[write] = buffer[read]
		write++
	}
	return string(buffer[:write])
}
