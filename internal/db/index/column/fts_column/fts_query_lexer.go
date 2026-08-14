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

package ftscolumn

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ErrInvalidFTSQuery identifies invalid parser configuration or input.
var ErrInvalidFTSQuery = errors.New("core: invalid FTS query")

// ErrFTSQueryTooComplex identifies a query whose token count or nesting depth
// exceeds the parser's explicit resource bounds.
var ErrFTSQueryTooComplex = errors.New("core: FTS query is too complex")

const (
	// MaxFTSQueryTokens bounds lexer/parser memory consumption.
	MaxFTSQueryTokens = 1 << 20
	// MaxFTSQueryDepth prevents recursive parenthesis parsing from exhausting
	// the Go stack.
	MaxFTSQueryDepth = 256
)

// FTSQueryTokenType identifies one token from the baseline-compatible query
// lexer. Whitespace is skipped and an EOF token terminates every successful
// result.
type FTSQueryTokenType uint8

const (
	FTSQueryTokenEOF FTSQueryTokenType = iota
	FTSQueryTokenOR
	FTSQueryTokenAND
	FTSQueryTokenNOT
	FTSQueryTokenPlus
	FTSQueryTokenMinus
	FTSQueryTokenColon
	FTSQueryTokenCaret
	FTSQueryTokenLeftParen
	FTSQueryTokenRightParen
	FTSQueryTokenPhrase
	FTSQueryTokenRegularID
	FTSQueryTokenNumber
	FTSQueryTokenTerm
	FTSQueryTokenDefault
)

var ftsQueryTokenTypeNames = map[FTSQueryTokenType]string{
	FTSQueryTokenEOF:        "EOF",
	FTSQueryTokenOR:         "OR",
	FTSQueryTokenAND:        "AND",
	FTSQueryTokenNOT:        "NOT",
	FTSQueryTokenPlus:       "PLUS",
	FTSQueryTokenMinus:      "MINUS",
	FTSQueryTokenColon:      "COLON",
	FTSQueryTokenCaret:      "CARET",
	FTSQueryTokenLeftParen:  "LPAREN",
	FTSQueryTokenRightParen: "RPAREN",
	FTSQueryTokenPhrase:     "PHRASE",
	FTSQueryTokenRegularID:  "REGULAR_ID",
	FTSQueryTokenNumber:     "NUMBER",
	FTSQueryTokenTerm:       "TERM",
	FTSQueryTokenDefault:    "DEFAULT",
}

func (t FTSQueryTokenType) String() string {
	if name, ok := ftsQueryTokenTypeNames[t]; ok {
		return name
	}
	return "UNKNOWN(" + strconv.FormatUint(uint64(t), 10) + ")"
}

// FTSQueryToken records an owned token and its byte/rune source location.
// End is an exclusive byte offset. Line is one-based and Column is zero-based.
type FTSQueryToken struct {
	Type   FTSQueryTokenType
	Text   string
	Offset uint32
	End    uint32
	Line   uint32
	Column uint32
}

// LexFTSQuery applies the pinned FtsLexer.g4 longest-match rules. The catch-all
// DEFAULT rule means every byte sequence can be tokenized; syntactic validity
// is determined by ParseFTSQuery.
func LexFTSQuery(ctx context.Context, query string) ([]FTSQueryToken, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil lexer context", ErrInvalidFTSQuery)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(len(query)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: input exceeds uint32 offsets", ErrInvalidFTSQuery)
	}
	tokens := make([]FTSQueryToken, 0, min(len(query)/3+1, 256))
	offset := 0
	line, column := uint32(1), uint32(0)
	for offset < len(query) {
		if offset&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if ftsQuerySpace(query[offset]) {
			start := offset
			for offset < len(query) && ftsQuerySpace(query[offset]) {
				if (offset-start)&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				offset++
			}
			if err := advanceFTSQueryPosition(ctx, query[start:offset], &line, &column); err != nil {
				return nil, err
			}
			continue
		}

		start, startLine, startColumn := offset, line, column
		tokenType, end, err := matchFTSQueryToken(ctx, query, offset)
		if err != nil {
			return nil, err
		}
		if end <= offset {
			return nil, fmt.Errorf("%w: lexer made no progress at byte %d", ErrInvalidFTSQuery, offset)
		}
		if len(tokens) >= MaxFTSQueryTokens {
			return nil, &FTSQueryParseError{
				Kind: ErrFTSQueryTooComplex, Offset: uint32(start), Line: startLine,
				Column: startColumn, Message: "token limit exceeded",
			}
		}
		tokens = append(tokens, FTSQueryToken{
			Type: tokenType, Text: strings.Clone(query[start:end]),
			Offset: uint32(start), End: uint32(end), Line: startLine, Column: startColumn,
		})
		if err := advanceFTSQueryPosition(ctx, query[start:end], &line, &column); err != nil {
			return nil, err
		}
		offset = end
	}
	tokens = append(tokens, FTSQueryToken{
		Type: FTSQueryTokenEOF, Offset: uint32(len(query)), End: uint32(len(query)),
		Line: line, Column: column,
	})
	return tokens, nil
}

func matchFTSQueryToken(ctx context.Context, query string, offset int) (FTSQueryTokenType, int, error) {
	bestType, bestEnd, bestPriority := FTSQueryTokenDefault, offset, int(^uint(0)>>1)
	consider := func(tokenType FTSQueryTokenType, end, priority int) {
		if end > bestEnd || end == bestEnd && priority < bestPriority {
			bestType, bestEnd, bestPriority = tokenType, end, priority
		}
	}

	// Rule order is significant when two rules match the same longest span.
	for index, keyword := range []struct {
		text      string
		tokenType FTSQueryTokenType
	}{{"OR", FTSQueryTokenOR}, {"AND", FTSQueryTokenAND}, {"NOT", FTSQueryTokenNOT}} {
		if len(query)-offset >= len(keyword.text) && strings.EqualFold(query[offset:offset+len(keyword.text)], keyword.text) {
			consider(keyword.tokenType, offset+len(keyword.text), index)
		}
	}

	switch query[offset] {
	case '+':
		consider(FTSQueryTokenPlus, offset+1, 3)
	case '-':
		consider(FTSQueryTokenMinus, offset+1, 4)
	case ':':
		consider(FTSQueryTokenColon, offset+1, 5)
	case '^':
		consider(FTSQueryTokenCaret, offset+1, 6)
	case '(':
		consider(FTSQueryTokenLeftParen, offset+1, 7)
	case ')':
		consider(FTSQueryTokenRightParen, offset+1, 8)
	case '"':
		end, err := scanFTSQuotedString(ctx, query, offset)
		if err != nil {
			return 0, 0, err
		}
		if end > offset {
			consider(FTSQueryTokenPhrase, end, 9)
		}
	}
	end, err := scanFTSRegularID(ctx, query, offset)
	if err != nil {
		return 0, 0, err
	}
	if end > offset {
		consider(FTSQueryTokenRegularID, end, 10)
	}
	end, err = scanFTSNumber(ctx, query, offset)
	if err != nil {
		return 0, 0, err
	}
	if end > offset {
		consider(FTSQueryTokenNumber, end, 11)
	}
	end, err = scanFTSTerm(ctx, query, offset)
	if err != nil {
		return 0, 0, err
	}
	if end > offset {
		consider(FTSQueryTokenTerm, end, 12)
	}
	_, size := utf8.DecodeRuneInString(query[offset:])
	if size <= 0 {
		size = 1
	}
	consider(FTSQueryTokenDefault, offset+size, 14)
	return bestType, bestEnd, nil
}

func scanFTSQuotedString(ctx context.Context, query string, offset int) (int, error) {
	for index := offset + 1; index < len(query); {
		if (index-offset)&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		switch query[index] {
		case '"':
			return index + 1, nil
		case '\r', '\n':
			return offset, nil
		case '\\':
			index++
			if index == len(query) {
				return offset, nil
			}
			_, size := utf8.DecodeRuneInString(query[index:])
			if size <= 0 {
				size = 1
			}
			index += size
		default:
			_, size := utf8.DecodeRuneInString(query[index:])
			if size <= 0 {
				size = 1
			}
			index += size
		}
	}
	return offset, nil
}

func scanFTSRegularID(ctx context.Context, query string, offset int) (int, error) {
	if offset >= len(query) || !ftsASCIIAlpha(query[offset]) && query[offset] != '_' {
		return offset, nil
	}
	index := offset + 1
	for index < len(query) && (ftsASCIIAlphaNumeric(query[index]) || query[index] == '_' || query[index] == '-') {
		if (index-offset)&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		index++
	}
	return index, nil
}

func scanFTSNumber(ctx context.Context, query string, offset int) (int, error) {
	if offset >= len(query) || query[offset] < '0' || query[offset] > '9' {
		return offset, nil
	}
	index := offset + 1
	for index < len(query) && query[index] >= '0' && query[index] <= '9' {
		if (index-offset)&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		index++
	}
	if index+1 < len(query) && query[index] == '.' && query[index+1] >= '0' && query[index+1] <= '9' {
		index += 2
		for index < len(query) && query[index] >= '0' && query[index] <= '9' {
			if (index-offset)&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return 0, err
				}
			}
			index++
		}
	}
	return index, nil
}

func scanFTSTerm(ctx context.Context, query string, offset int) (int, error) {
	if offset >= len(query) {
		return offset, nil
	}
	_, size, ok := scanFTSTermStart(query, offset)
	if !ok {
		return offset, nil
	}
	index := offset + size
	for index < len(query) {
		if (index-offset)&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		value, bodySize, body := scanFTSTermStart(query, index)
		if body || value < utf8.RuneSelf && strings.ContainsRune("._#/%-'@", value) {
			index += bodySize
			continue
		}
		if query[index] == '\\' && index+1 < len(query) && ftsEscapableQueryCharacter(query[index+1]) {
			index += 2
			continue
		}
		break
	}
	return index, nil
}

func scanFTSTermStart(query string, offset int) (rune, int, bool) {
	value, size := utf8.DecodeRuneInString(query[offset:])
	if value == utf8.RuneError && size == 1 {
		return value, 1, false
	}
	if value < utf8.RuneSelf {
		return value, size, ftsASCIIAlphaNumeric(byte(value)) || value == '_'
	}
	return value, size, value >= 0x80 && value <= 0xffff
}

func ftsEscapableQueryCharacter(value byte) bool {
	return strings.ContainsRune(`-+=&|!(){}[]^"~*?:\/`, rune(value))
}

func ftsASCIIAlpha(value byte) bool {
	return value >= 'A' && value <= 'Z' || value >= 'a' && value <= 'z'
}

func ftsASCIIAlphaNumeric(value byte) bool {
	return ftsASCIIAlpha(value) || value >= '0' && value <= '9'
}

func ftsQuerySpace(value byte) bool {
	switch value {
	case ' ', '\t', '\r', '\n':
		return true
	default:
		return false
	}
}

func advanceFTSQueryPosition(ctx context.Context, text string, line, column *uint32) error {
	for offset := 0; offset < len(text); {
		if offset&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		value, size := utf8.DecodeRuneInString(text[offset:])
		if size <= 0 {
			size = 1
		}
		if value == '\n' {
			(*line)++
			*column = 0
		} else {
			(*column)++
		}
		offset += size
	}
	return nil
}
