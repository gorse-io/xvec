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

package tokenizer

import (
	"context"
	"errors"
	"sort"
	"strings"
	"unicode/utf8"
)

// TokenFilter transforms an ordered token list without changing source
// offsets or positions.
type TokenFilter interface {
	Name() string
	Filter(ctx context.Context, tokens []Token) ([]Token, error)
}

// LowercaseTokenFilter applies the baseline Unicode 17 simple lowercase
// mapping. It is immutable and safe for concurrent use.
type LowercaseTokenFilter struct{}

// NewLowercaseTokenFilter constructs a stateless lowercase filter.
func NewLowercaseTokenFilter() *LowercaseTokenFilter { return &LowercaseTokenFilter{} }

func (*LowercaseTokenFilter) Name() string { return "lowercase" }

// Filter returns an owned copy of tokens with text lowercased. Malformed UTF-8
// bytes are copied one byte at a time, matching utf8proc_iterate fallback.
func (*LowercaseTokenFilter) Filter(ctx context.Context, tokens []Token) ([]Token, error) {
	if ctx == nil {
		return nil, errors.New("core: nil token filter context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]Token, len(tokens))
	work := 0
	for index, token := range tokens {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		result[index] = token
		var builder strings.Builder
		builder.Grow(len(token.Text))
		for offset := 0; offset < len(token.Text); {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			value, size := utf8.DecodeRuneInString(token.Text[offset:])
			if value == utf8.RuneError && size == 1 {
				builder.WriteByte(token.Text[offset])
				offset++
				work++
				continue
			}
			builder.WriteRune(lowercaseUnicode17(value))
			offset += size
			work++
		}
		result[index].Text = builder.String()
	}
	return result, nil
}

type lowercaseRange struct {
	first rune
	last  rune
	step  rune
	delta rune
}

func lowercaseUnicode17(value rune) rune {
	index := sort.Search(len(lowercaseUnicode17Ranges), func(index int) bool {
		return lowercaseUnicode17Ranges[index].last >= value
	})
	if index == len(lowercaseUnicode17Ranges) {
		return value
	}
	mapping := lowercaseUnicode17Ranges[index]
	if value < mapping.first || (value-mapping.first)%mapping.step != 0 {
		return value
	}
	return value + mapping.delta
}

var _ TokenFilter = (*LowercaseTokenFilter)(nil)
