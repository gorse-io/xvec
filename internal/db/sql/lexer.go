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
	"fmt"
	"strings"
	"unicode/utf8"
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

var keywordTokens = map[string]TokenKind{
	"OR": TokenOr, "AND": TokenAnd, "NOT": TokenNot, "IN": TokenIn,
	"CONTAIN_ALL": TokenContainAll, "CONTAIN_ANY": TokenContainAny,
	"BETWEEN": TokenBetween, "LIKE": TokenLike, "WHERE": TokenWhere,
	"SELECT": TokenSelect, "FROM": TokenFrom, "AS": TokenAs, "BY": TokenBy,
	"ORDER": TokenOrder, "ASC": TokenAsc, "DESC": TokenDesc,
	"LIMIT": TokenLimit, "TRUE": TokenTrue, "FALSE": TokenFalse,
	"IS": TokenIs, "NULL": TokenNull,
}

// Lex tokenizes one filter. Keywords are case-insensitive while token text and
// identifiers preserve their original spelling. SQL comments are ignored.
func Lex(input string) ([]Token, error) {
	if len(input) > MaxFilterBytes {
		return nil, parseError(Position{Line: 1, Column: 1}, "filter is %d bytes, maximum %d", len(input), MaxFilterBytes)
	}
	if !utf8.ValidString(input) {
		offset := firstInvalidUTF8(input)
		return nil, parseError(positionAt(input, offset), "filter is not valid UTF-8")
	}
	lexer := filterLexer{input: input, position: Position{Line: 1, Column: 1}}
	return lexer.lex()
}

type filterLexer struct {
	input    string
	offset   int
	position Position
	tokens   []Token
}

func (l *filterLexer) lex() ([]Token, error) {
	for l.offset < len(l.input) {
		if l.skipWhitespace() || l.skipLineComment() {
			continue
		}
		if strings.HasPrefix(l.input[l.offset:], "/*") {
			if err := l.skipBlockComment(); err != nil {
				return nil, err
			}
			continue
		}
		start := l.position
		startOffset := l.offset
		if token, ok := l.lexFixedToken(); ok {
			l.tokens = append(l.tokens, token)
			continue
		}
		if l.input[l.offset] == '\'' || l.input[l.offset] == '"' {
			token, err := l.lexString()
			if err != nil {
				return nil, err
			}
			l.tokens = append(l.tokens, token)
			continue
		}
		identifierEnd := scanIdentifier(l.input, l.offset)
		numericEnd, numericKind := scanNumber(l.input, l.offset)
		if numericEnd > l.offset && numericEnd >= identifierEnd {
			l.advanceTo(numericEnd)
			l.tokens = append(l.tokens, Token{
				Kind: numericKind, Text: l.input[startOffset:numericEnd],
				Span: Span{Start: start, End: l.position},
			})
			continue
		}
		if identifierEnd > l.offset {
			// The pinned lexer emits its earlier MINUS_SIGN/UNDERSCORE rules
			// for these one-byte spellings rather than REGULAR_ID.
			if identifierEnd == l.offset+1 && (l.input[l.offset] == '-' || l.input[l.offset] == '_') {
				return nil, parseError(start, "unexpected character %q", l.input[l.offset:identifierEnd])
			}
			l.advanceTo(identifierEnd)
			text := l.input[startOffset:identifierEnd]
			kind := TokenIdentifier
			if keyword, found := keywordTokens[strings.ToUpper(text)]; found {
				kind = keyword
			}
			l.tokens = append(l.tokens, Token{Kind: kind, Text: text, Span: Span{Start: start, End: l.position}})
			continue
		}
		_, width := utf8.DecodeRuneInString(l.input[l.offset:])
		bad := l.input[l.offset : l.offset+width]
		return nil, parseError(start, "unexpected character %q", bad)
	}
	eof := Token{Kind: TokenEOF, Span: Span{Start: l.position, End: l.position}}
	l.tokens = append(l.tokens, eof)
	return l.tokens, nil
}

func (l *filterLexer) lexFixedToken() (Token, bool) {
	start := l.position
	startOffset := l.offset
	fixed := []struct {
		text string
		kind TokenKind
	}{
		{"<=", TokenLessEqual}, {">=", TokenGreaterEqual}, {"!=", TokenNotEqual},
		{"=", TokenEqual}, {"<", TokenLess}, {">", TokenGreater},
		{"(", TokenLeftParen}, {")", TokenRightParen},
		{"[", TokenLeftBracket}, {"]", TokenRightBracket}, {",", TokenComma},
	}
	for _, candidate := range fixed {
		if strings.HasPrefix(l.input[l.offset:], candidate.text) {
			l.advanceTo(l.offset + len(candidate.text))
			return Token{
				Kind: candidate.kind, Text: l.input[startOffset:l.offset],
				Span: Span{Start: start, End: l.position},
			}, true
		}
	}
	return Token{}, false
}

func (l *filterLexer) lexString() (Token, error) {
	start := l.position
	startOffset := l.offset
	quote := l.input[l.offset]
	l.advanceOne()
	for l.offset < len(l.input) {
		current := l.input[l.offset]
		if current == quote {
			l.advanceOne()
			return Token{
				Kind: TokenString, Text: l.input[startOffset:l.offset],
				Span: Span{Start: start, End: l.position},
			}, nil
		}
		if current == '\\' {
			l.advanceOne()
			if l.offset == len(l.input) {
				return Token{}, parseError(start, "unterminated quoted string")
			}
			l.advanceOne()
			continue
		}
		l.advanceOne()
	}
	return Token{}, parseError(start, "unterminated quoted string")
}

func (l *filterLexer) skipWhitespace() bool {
	start := l.offset
	for l.offset < len(l.input) {
		switch l.input[l.offset] {
		case ' ', '\t', '\r', '\n':
			l.advanceOne()
		default:
			return l.offset != start
		}
	}
	return l.offset != start
}

func (l *filterLexer) skipLineComment() bool {
	if !strings.HasPrefix(l.input[l.offset:], "--") {
		return false
	}
	for l.offset < len(l.input) {
		current := l.input[l.offset]
		l.advanceOne()
		if current == '\n' {
			break
		}
	}
	return true
}

func (l *filterLexer) skipBlockComment() error {
	start := l.position
	l.advanceTo(l.offset + 2)
	for l.offset < len(l.input) {
		if strings.HasPrefix(l.input[l.offset:], "*/") {
			l.advanceTo(l.offset + 2)
			return nil
		}
		l.advanceOne()
	}
	return parseError(start, "unterminated block comment")
}

func (l *filterLexer) advanceTo(end int) {
	for l.offset < end {
		l.advanceOne()
	}
}

func (l *filterLexer) advanceOne() {
	r, width := utf8.DecodeRuneInString(l.input[l.offset:])
	l.offset += width
	l.position.Offset = l.offset
	if r == '\n' {
		l.position.Line++
		l.position.Column = 1
	} else {
		l.position.Column++
	}
}

func scanIdentifier(input string, offset int) int {
	end := offset
	for end < len(input) && isIdentifierByte(input[end]) {
		end++
	}
	return end
}

func isIdentifierByte(value byte) bool {
	return value >= 'a' && value <= 'z' || value >= 'A' && value <= 'Z' ||
		value >= '0' && value <= '9' || value == '_' || value == '-'
}

func scanNumber(input string, offset int) (int, TokenKind) {
	position := offset
	if position < len(input) && input[position] == '-' {
		position++
	}
	startDigits := position
	for position < len(input) && isDigit(input[position]) {
		position++
	}
	integerDigits := position - startDigits
	hasDot := false
	if position < len(input) && input[position] == '.' {
		dotPosition := position
		hasDot = true
		position++
		fractionStart := position
		for position < len(input) && isDigit(input[position]) {
			position++
		}
		if position == fractionStart {
			if integerDigits != 0 {
				return dotPosition, TokenInteger
			}
			return offset, TokenEOF
		}
	}
	if integerDigits == 0 && !hasDot {
		return offset, TokenEOF
	}
	hasExponent := false
	if position < len(input) && (input[position] == 'e' || input[position] == 'E') {
		exponentMarker := position
		position++
		if position < len(input) && (input[position] == '+' || input[position] == '-') {
			position++
		}
		exponentDigitsBefore := position
		for position < len(input) && isDigit(input[position]) {
			position++
		}
		if position < len(input) && input[position] == '.' {
			position++
			fractionStart := position
			for position < len(input) && isDigit(input[position]) {
				position++
			}
			if position == fractionStart {
				position = exponentMarker
			} else {
				hasExponent = true
			}
		} else if position > exponentDigitsBefore {
			hasExponent = true
		} else {
			position = exponentMarker
		}
	}
	hasSuffix := false
	if position < len(input) && (input[position] == 'd' || input[position] == 'D' || input[position] == 'f' || input[position] == 'F') {
		hasSuffix = true
		position++
	}
	if hasDot || hasExponent || hasSuffix {
		return position, TokenFloat
	}
	return position, TokenInteger
}

func isDigit(value byte) bool { return value >= '0' && value <= '9' }

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
