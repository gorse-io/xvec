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
	"fmt"
	"unicode/utf8"

	antlrRuntime "github.com/antlr4-go/antlr/v4"
	sqlantlr "github.com/gorse-io/xvec/internal/db/sqlengine/antlr"
)

const MaxFilterBytes = 1 << 20

// TokenKind classifies filter syntax tokens.
type TokenKind uint8

const (
	TokenEOF TokenKind = iota
	TokenIdentifier
	TokenInteger
	TokenFloat
	TokenString
	TokenTrue
	TokenFalse
	TokenOr
	TokenAnd
	TokenNot
	TokenIn
	TokenContainAll
	TokenContainAny
	TokenBetween
	TokenLike
	TokenWhere
	TokenSelect
	TokenFrom
	TokenAs
	TokenBy
	TokenOrder
	TokenAsc
	TokenDesc
	TokenLimit
	TokenIs
	TokenNull
	TokenEqual
	TokenNotEqual
	TokenLess
	TokenLessEqual
	TokenGreater
	TokenGreaterEqual
	TokenLeftParen
	TokenRightParen
	TokenLeftBracket
	TokenRightBracket
	TokenComma
)

func (k TokenKind) String() string {
	switch k {
	case TokenEOF:
		return "end of input"
	case TokenIdentifier:
		return "identifier"
	case TokenInteger:
		return "integer"
	case TokenFloat:
		return "float"
	case TokenString:
		return "string"
	case TokenTrue:
		return "TRUE"
	case TokenFalse:
		return "FALSE"
	case TokenOr:
		return "OR"
	case TokenAnd:
		return "AND"
	case TokenNot:
		return "NOT"
	case TokenIn:
		return "IN"
	case TokenContainAll:
		return "CONTAIN_ALL"
	case TokenContainAny:
		return "CONTAIN_ANY"
	case TokenBetween:
		return "BETWEEN"
	case TokenLike:
		return "LIKE"
	case TokenWhere:
		return "WHERE"
	case TokenSelect:
		return "SELECT"
	case TokenFrom:
		return "FROM"
	case TokenAs:
		return "AS"
	case TokenBy:
		return "BY"
	case TokenOrder:
		return "ORDER"
	case TokenAsc:
		return "ASC"
	case TokenDesc:
		return "DESC"
	case TokenLimit:
		return "LIMIT"
	case TokenIs:
		return "IS"
	case TokenNull:
		return "NULL"
	case TokenEqual:
		return "="
	case TokenNotEqual:
		return "!="
	case TokenLess:
		return "<"
	case TokenLessEqual:
		return "<="
	case TokenGreater:
		return ">"
	case TokenGreaterEqual:
		return ">="
	case TokenLeftParen:
		return "("
	case TokenRightParen:
		return ")"
	case TokenLeftBracket:
		return "["
	case TokenRightBracket:
		return "]"
	case TokenComma:
		return ","
	default:
		return fmt.Sprintf("token(%d)", k)
	}
}

// Token preserves original source text and its half-open range.
type Token struct {
	Kind TokenKind
	Text string
	Span Span
}

var antlrTokenKinds = map[int]TokenKind{
	antlrRuntime.TokenEOF:          TokenEOF,
	sqlantlr.SQLLexerREGULAR_ID:    TokenIdentifier,
	sqlantlr.SQLLexerINTEGER:       TokenInteger,
	sqlantlr.SQLLexerFLOAT:         TokenFloat,
	sqlantlr.SQLLexerSQUOTA_STRING: TokenString,
	sqlantlr.SQLLexerDQUOTA_STRING: TokenString,
	sqlantlr.SQLLexerTRUE_V:        TokenTrue,
	sqlantlr.SQLLexerFALSE_V:       TokenFalse,
	sqlantlr.SQLLexerOR:            TokenOr,
	sqlantlr.SQLLexerAND:           TokenAnd,
	sqlantlr.SQLLexerNOT:           TokenNot,
	sqlantlr.SQLLexerIN:            TokenIn,
	sqlantlr.SQLLexerCONTAIN_ALL:   TokenContainAll,
	sqlantlr.SQLLexerCONTAIN_ANY:   TokenContainAny,
	sqlantlr.SQLLexerBETWEEN:       TokenBetween,
	sqlantlr.SQLLexerLIKE:          TokenLike,
	sqlantlr.SQLLexerWHERE:         TokenWhere,
	sqlantlr.SQLLexerSELECT:        TokenSelect,
	sqlantlr.SQLLexerFROM:          TokenFrom,
	sqlantlr.SQLLexerAS:            TokenAs,
	sqlantlr.SQLLexerBY:            TokenBy,
	sqlantlr.SQLLexerORDER:         TokenOrder,
	sqlantlr.SQLLexerASC:           TokenAsc,
	sqlantlr.SQLLexerDESC:          TokenDesc,
	sqlantlr.SQLLexerLIMIT:         TokenLimit,
	sqlantlr.SQLLexerIS:            TokenIs,
	sqlantlr.SQLLexerNULL_V:        TokenNull,
	sqlantlr.SQLLexerE_OP:          TokenEqual,
	sqlantlr.SQLLexerNE_OP:         TokenNotEqual,
	sqlantlr.SQLLexerL_OP:          TokenLess,
	sqlantlr.SQLLexerLE_OP:         TokenLessEqual,
	sqlantlr.SQLLexerG_OP:          TokenGreater,
	sqlantlr.SQLLexerGE_OP:         TokenGreaterEqual,
	sqlantlr.SQLLexerLP:            TokenLeftParen,
	sqlantlr.SQLLexerRP:            TokenRightParen,
	sqlantlr.SQLLexerLMP:           TokenLeftBracket,
	sqlantlr.SQLLexerRMP:           TokenRightBracket,
	sqlantlr.SQLLexerCOMMA:         TokenComma,
}

// Lex tokenizes one filter with the generated ANTLR lexer. Keywords are
// case-insensitive while token text and identifiers preserve their spelling.
func Lex(input string) ([]Token, error) {
	positions, err := validateFilter(input)
	if err != nil {
		return nil, err
	}

	inputStream := antlrRuntime.NewInputStream(input)
	lexer := sqlantlr.NewSQLLexer(inputStream)
	errors := newSyntaxErrorListener(positions)
	lexer.RemoveErrorListeners()
	lexer.AddErrorListener(errors)

	var tokens []Token
	for {
		token := lexer.NextToken()
		if errors.err != nil {
			return nil, errors.err
		}
		if token.GetChannel() != antlrRuntime.TokenDefaultChannel && token.GetTokenType() != antlrRuntime.TokenEOF {
			continue
		}
		kind, ok := antlrTokenKinds[token.GetTokenType()]
		if !ok {
			return nil, parseError(positions.atRune(token.GetStart()), "unexpected character %q", token.GetText())
		}
		text := token.GetText()
		if token.GetTokenType() == antlrRuntime.TokenEOF {
			text = ""
		}
		tokens = append(tokens, Token{Kind: kind, Text: text, Span: positions.tokenSpan(token)})
		if token.GetTokenType() == antlrRuntime.TokenEOF {
			return tokens, nil
		}
	}
}

func validateFilter(input string) (*sourcePositions, error) {
	if len(input) > MaxFilterBytes {
		return nil, parseError(Position{Line: 1, Column: 1}, "filter is %d bytes, maximum %d", len(input), MaxFilterBytes)
	}
	if !utf8.ValidString(input) {
		offset := firstInvalidUTF8(input)
		return nil, parseError(positionAt(input, offset), "filter is not valid UTF-8")
	}
	return newSourcePositions(input), nil
}

type sourcePositions struct {
	positions []Position
}

func newSourcePositions(input string) *sourcePositions {
	positions := make([]Position, 1, utf8.RuneCountInString(input)+1)
	positions[0] = Position{Line: 1, Column: 1}
	for offset := 0; offset < len(input); {
		r, width := utf8.DecodeRuneInString(input[offset:])
		offset += width
		next := positions[len(positions)-1]
		next.Offset = offset
		if r == '\n' {
			next.Line++
			next.Column = 1
		} else {
			next.Column++
		}
		positions = append(positions, next)
	}
	return &sourcePositions{positions: positions}
}

func (p *sourcePositions) atRune(index int) Position {
	if index < 0 {
		index = 0
	}
	if index >= len(p.positions) {
		index = len(p.positions) - 1
	}
	return p.positions[index]
}

func (p *sourcePositions) atLineColumn(line, zeroBasedColumn int) Position {
	column := zeroBasedColumn + 1
	for _, position := range p.positions {
		if position.Line == line && position.Column == column {
			return position
		}
	}
	return p.positions[len(p.positions)-1]
}

func (p *sourcePositions) tokenSpan(token antlrRuntime.Token) Span {
	start := token.GetStart()
	end := token.GetStop() + 1
	if token.GetTokenType() == antlrRuntime.TokenEOF {
		end = start
	}
	return Span{Start: p.atRune(start), End: p.atRune(end)}
}

type syntaxErrorListener struct {
	*antlrRuntime.DefaultErrorListener
	positions *sourcePositions
	err       error
}

func newSyntaxErrorListener(positions *sourcePositions) *syntaxErrorListener {
	return &syntaxErrorListener{DefaultErrorListener: antlrRuntime.NewDefaultErrorListener(), positions: positions}
}

func (l *syntaxErrorListener) SyntaxError(_ antlrRuntime.Recognizer, offendingSymbol interface{}, line, column int, message string, _ antlrRuntime.RecognitionException) {
	if l.err != nil {
		return
	}
	position := l.positions.atLineColumn(line, column)
	if token, ok := offendingSymbol.(antlrRuntime.Token); ok {
		position = l.positions.atRune(token.GetStart())
	}
	l.err = parseError(position, "%s", message)
}

func firstInvalidUTF8(input string) int {
	for offset := 0; offset < len(input); {
		_, width := utf8.DecodeRuneInString(input[offset:])
		if width == 1 && input[offset] >= utf8.RuneSelf {
			return offset
		}
		offset += width
	}
	return len(input)
}

func positionAt(input string, target int) Position {
	position := Position{Line: 1, Column: 1}
	for offset := 0; offset < target; {
		r, width := utf8.DecodeRuneInString(input[offset:])
		offset += width
		position.Offset = offset
		if r == '\n' {
			position.Line++
			position.Column = 1
		} else {
			position.Column++
		}
	}
	position.Offset = target
	return position
}
