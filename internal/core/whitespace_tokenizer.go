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
	"math"
	"strings"
)

var ErrTokenizerInputTooLarge = errors.New("core: tokenizer input exceeds uint32 offset capacity")

// WhitespaceTokenizer implements the pinned byte-oriented whitespace split.
// It recognizes the six ASCII whitespace bytes accepted by C isspace in the C
// locale and otherwise preserves source bytes verbatim.
type WhitespaceTokenizer struct{}

func NewWhitespaceTokenizer() *WhitespaceTokenizer { return &WhitespaceTokenizer{} }

func (*WhitespaceTokenizer) Name() string { return "whitespace" }

func (*WhitespaceTokenizer) Tokenize(ctx context.Context, text string) ([]Token, error) {
	if ctx == nil {
		return nil, errors.New("core: nil tokenizer context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(len(text)) > math.MaxUint32 {
		return nil, ErrTokenizerInputTooLarge
	}
	tokens := make([]Token, 0)
	for index := 0; index < len(text); {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for index < len(text) && asciiWhitespace(text[index]) {
			index++
			if index&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
		}
		if index == len(text) {
			break
		}
		start := index
		for index < len(text) && !asciiWhitespace(text[index]) {
			index++
			if index&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
		}
		tokens = append(tokens, Token{
			Text: strings.Clone(text[start:index]), Offset: uint32(start),
			Position: uint32(len(tokens)),
		})
	}
	return tokens, nil
}

func asciiWhitespace(value byte) bool {
	switch value {
	case ' ', '\t', '\n', '\v', '\f', '\r':
		return true
	default:
		return false
	}
}
