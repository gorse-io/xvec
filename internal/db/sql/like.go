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
	"regexp"
	"strings"
	"unicode/utf8"
)

// LikeMode exposes the cheapest exact interpretation of a LIKE pattern. The
// later inverted-index planner uses this classification without changing
// forward-evaluation results.
type LikeMode uint8

const (
	LikeExact LikeMode = iota + 1
	LikeAny
	LikePrefix
	LikeSuffix
	LikeContains
	LikeGeneral
)

func (m LikeMode) String() string {
	switch m {
	case LikeExact:
		return "EXACT"
	case LikeAny:
		return "ANY"
	case LikePrefix:
		return "PREFIX"
	case LikeSuffix:
		return "SUFFIX"
	case LikeContains:
		return "CONTAINS"
	case LikeGeneral:
		return "GENERAL"
	default:
		return "UNKNOWN"
	}
}

type likeTokenKind uint8

const (
	likeLiteral likeTokenKind = iota + 1
	likeAnySequence
	likeAnyRune
)

type likeToken struct {
	kind likeTokenKind
	rune rune
}

// LikePattern is an immutable, concurrency-safe compiled SQL LIKE pattern.
// Percent matches any rune sequence, underscore one rune, and backslash makes
// the following rune literal.
type LikePattern struct {
	raw     string
	mode    LikeMode
	literal string
	regexp  *regexp.Regexp
}

// CompileLike validates UTF-8 and compiles a whole-string LIKE matcher using
// Go's linear-time RE2 engine.
func CompileLike(pattern string) (*LikePattern, error) {
	if len(pattern) > MaxFilterBytes {
		return nil, parseError(Position{Line: 1, Column: 1}, "LIKE pattern is %d bytes, maximum %d", len(pattern), MaxFilterBytes)
	}
	if !utf8.ValidString(pattern) {
		offset := firstInvalidUTF8(pattern)
		return nil, parseError(positionAt(pattern, offset), "LIKE pattern is not valid UTF-8")
	}
	tokens := tokenizeLike(pattern)
	mode, literal := classifyLike(tokens)
	var expression strings.Builder
	expression.WriteString(`(?s)\A`)
	for _, token := range tokens {
		switch token.kind {
		case likeLiteral:
			expression.WriteString(regexp.QuoteMeta(string(token.rune)))
		case likeAnySequence:
			expression.WriteString(`.*`)
		case likeAnyRune:
			expression.WriteByte('.')
		}
	}
	expression.WriteString(`\z`)
	compiled, err := regexp.Compile(expression.String())
	if err != nil {
		return nil, parseError(Position{Line: 1, Column: 1}, "compile LIKE pattern: %v", err)
	}
	return &LikePattern{raw: pattern, mode: mode, literal: literal, regexp: compiled}, nil
}

func (p *LikePattern) Match(value string) bool {
	return p != nil && p.regexp != nil && utf8.ValidString(value) && p.regexp.MatchString(value)
}

func (p *LikePattern) Raw() string {
	if p == nil {
		return ""
	}
	return p.raw
}

func (p *LikePattern) Mode() LikeMode {
	if p == nil {
		return 0
	}
	return p.mode
}

// Literal returns the unescaped exact, prefix, suffix, or contained text. It
// is empty for ANY and GENERAL patterns.
func (p *LikePattern) Literal() string {
	if p == nil {
		return ""
	}
	return p.literal
}

func tokenizeLike(pattern string) []likeToken {
	runes := []rune(pattern)
	tokens := make([]likeToken, 0, len(runes))
	for index := 0; index < len(runes); index++ {
		current := runes[index]
		if current == '\\' && index+1 < len(runes) {
			index++
			tokens = append(tokens, likeToken{kind: likeLiteral, rune: runes[index]})
			continue
		}
		switch current {
		case '%':
			// Consecutive percent wildcards have identical semantics and are
			// collapsed to keep the generated matcher compact.
			if len(tokens) == 0 || tokens[len(tokens)-1].kind != likeAnySequence {
				tokens = append(tokens, likeToken{kind: likeAnySequence})
			}
		case '_':
			tokens = append(tokens, likeToken{kind: likeAnyRune})
		default:
			tokens = append(tokens, likeToken{kind: likeLiteral, rune: current})
		}
	}
	return tokens
}

func classifyLike(tokens []likeToken) (LikeMode, string) {
	wildcards := 0
	for _, token := range tokens {
		if token.kind != likeLiteral {
			wildcards++
		}
	}
	literalBetween := func(start, end int) (string, bool) {
		var value strings.Builder
		for _, token := range tokens[start:end] {
			if token.kind != likeLiteral {
				return "", false
			}
			value.WriteRune(token.rune)
		}
		return value.String(), true
	}
	if wildcards == 0 {
		literal, _ := literalBetween(0, len(tokens))
		return LikeExact, literal
	}
	if len(tokens) == 1 && tokens[0].kind == likeAnySequence {
		return LikeAny, ""
	}
	if tokens[len(tokens)-1].kind == likeAnySequence {
		if literal, ok := literalBetween(0, len(tokens)-1); ok {
			return LikePrefix, literal
		}
	}
	if tokens[0].kind == likeAnySequence {
		if literal, ok := literalBetween(1, len(tokens)); ok {
			return LikeSuffix, literal
		}
	}
	if len(tokens) >= 2 && tokens[0].kind == likeAnySequence && tokens[len(tokens)-1].kind == likeAnySequence {
		if literal, ok := literalBetween(1, len(tokens)-1); ok {
			return LikeContains, literal
		}
	}
	return LikeGeneral, ""
}
