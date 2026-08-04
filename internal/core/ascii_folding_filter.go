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
	"math/bits"
	"sort"
	"strings"
	"unicode/utf8"
)

// ASCIIFoldingTokenFilter replaces the baseline's supported Unicode
// codepoints with short ASCII equivalents. It is immutable and safe for
// concurrent use.
type ASCIIFoldingTokenFilter struct{}

// NewASCIIFoldingTokenFilter constructs a stateless ASCII-folding filter.
func NewASCIIFoldingTokenFilter() *ASCIIFoldingTokenFilter {
	return &ASCIIFoldingTokenFilter{}
}

func (*ASCIIFoldingTokenFilter) Name() string { return "ascii_folding" }

// Filter returns owned, non-empty tokens with folded text. It preserves the
// order, offsets, and positions of every retained token. Malformed UTF-8 bytes
// pass through one byte at a time.
func (*ASCIIFoldingTokenFilter) Filter(ctx context.Context, tokens []Token) ([]Token, error) {
	if ctx == nil {
		return nil, errors.New("core: nil token filter context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]Token, 0, len(tokens))
	work := 0
	for _, token := range tokens {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if token.Text == "" {
			continue
		}
		folded, err := foldTokenToASCII(ctx, token.Text, &work)
		if err != nil {
			return nil, err
		}
		if folded == "" {
			continue
		}
		token.Text = folded
		result = append(result, token)
	}
	return result, nil
}

func foldTokenToASCII(ctx context.Context, text string, work *int) (string, error) {
	allASCII := true
	for index := 0; index < len(text); index++ {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return "", err
			}
		}
		if text[index] >= utf8.RuneSelf {
			allASCII = false
			break
		}
	}
	if allASCII {
		*work += len(text)
		return strings.Clone(text), nil
	}
	var builder strings.Builder
	builder.Grow(len(text))
	for offset := 0; offset < len(text); {
		if *work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return "", err
			}
		}
		if text[offset] < utf8.RuneSelf {
			builder.WriteByte(text[offset])
			offset++
			(*work)++
			continue
		}
		value, size := utf8.DecodeRuneInString(text[offset:])
		if value == utf8.RuneError && size == 1 {
			builder.WriteByte(text[offset])
			offset++
			(*work)++
			continue
		}
		if replacement, found := lookupASCIIExtraFold(value); found {
			builder.WriteString(replacement)
		} else if replacement, found := lookupASCIINFKDFold(value); found {
			writePackedASCII(&builder, replacement)
		} else {
			builder.WriteString(text[offset : offset+size])
		}
		offset += size
		(*work)++
	}
	return builder.String(), nil
}

type asciiExtraFold struct {
	codepoint   rune
	replacement string
}

func lookupASCIIExtraFold(value rune) (string, bool) {
	index := sort.Search(len(asciiExtraFolds), func(index int) bool {
		return asciiExtraFolds[index].codepoint >= value
	})
	if index < len(asciiExtraFolds) && asciiExtraFolds[index].codepoint == value {
		return asciiExtraFolds[index].replacement, true
	}
	return "", false
}

func lookupASCIINFKDFold(value rune) (uint32, bool) {
	index := sort.Search(len(asciiNFKDCodepoints), func(index int) bool {
		return asciiNFKDCodepoints[index] >= value
	})
	if index < len(asciiNFKDCodepoints) && asciiNFKDCodepoints[index] == value {
		return asciiNFKDReplacements[index], true
	}
	return 0, false
}

func writePackedASCII(builder *strings.Builder, value uint32) {
	if value == 0 {
		return
	}
	shift := (bits.Len32(value) - 1) / 8 * 8
	for ; shift >= 0; shift -= 8 {
		builder.WriteByte(byte(value >> shift))
	}
}

var _ TokenFilter = (*ASCIIFoldingTokenFilter)(nil)
