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
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	// DefaultNGramLength is the pinned minimum and maximum ngram length.
	DefaultNGramLength uint32 = 2
	// MaxNGramLengthDifference is the largest supported Max-Min range.
	MaxNGramLengthDifference uint32 = 1
)

// NGramTokenChars is a mask of Unicode character classes retained by the
// tokenizer. Zero retains every valid UTF-8 codepoint, matching the baseline.
type NGramTokenChars uint32

const (
	NGramTokenCharLetter NGramTokenChars = 1 << iota
	NGramTokenCharDigit
	NGramTokenCharWhitespace
	NGramTokenCharPunctuation
	NGramTokenCharSymbol

	ngramTokenCharLetter      = NGramTokenCharLetter
	ngramTokenCharDigit       = NGramTokenCharDigit
	ngramTokenCharWhitespace  = NGramTokenCharWhitespace
	ngramTokenCharPunctuation = NGramTokenCharPunctuation
	ngramTokenCharSymbol      = NGramTokenCharSymbol
	ngramTokenCharAll         = NGramTokenCharLetter | NGramTokenCharDigit | NGramTokenCharWhitespace | NGramTokenCharPunctuation | NGramTokenCharSymbol
)

// ErrInvalidNGramTokenizerOptions identifies invalid ngram construction
// options.
var ErrInvalidNGramTokenizerOptions = errors.New("core: invalid ngram tokenizer options")

// NGramTokenizerOptions configures codepoint ngram lengths and optional
// Unicode character-class filtering.
type NGramTokenizerOptions struct {
	Min        uint32
	Max        uint32
	TokenChars NGramTokenChars
}

// DefaultNGramTokenizerOptions returns baseline bigram settings and retains
// all valid UTF-8 codepoints.
func DefaultNGramTokenizerOptions() NGramTokenizerOptions {
	return NGramTokenizerOptions{Min: DefaultNGramLength, Max: DefaultNGramLength}
}

// Validate checks the pinned positive uint32 and one-length-span constraints.
func (o NGramTokenizerOptions) Validate() error {
	if o.Min == 0 {
		return fmt.Errorf("%w: Min must be a positive uint32", ErrInvalidNGramTokenizerOptions)
	}
	if o.Max == 0 {
		return fmt.Errorf("%w: Max must be a positive uint32", ErrInvalidNGramTokenizerOptions)
	}
	if o.Min > o.Max {
		return fmt.Errorf("%w: Min must be less than or equal to Max", ErrInvalidNGramTokenizerOptions)
	}
	if o.Max-o.Min > MaxNGramLengthDifference {
		return fmt.Errorf("%w: Max-Min must be at most %d", ErrInvalidNGramTokenizerOptions, MaxNGramLengthDifference)
	}
	if o.TokenChars&^ngramTokenCharAll != 0 {
		return fmt.Errorf("%w: TokenChars contains unknown class bits 0x%x", ErrInvalidNGramTokenizerOptions, uint32(o.TokenChars&^ngramTokenCharAll))
	}
	return nil
}

// NGramTokenizer emits overlapping UTF-8 codepoint ngrams.
type NGramTokenizer struct {
	minimum    uint32
	maximum    uint32
	tokenChars NGramTokenChars
}

// NewNGramTokenizer constructs a validated tokenizer.
func NewNGramTokenizer(options NGramTokenizerOptions) (*NGramTokenizer, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &NGramTokenizer{minimum: options.Min, maximum: options.Max, tokenChars: options.TokenChars}, nil
}

func (*NGramTokenizer) Name() string { return "ngram" }

func (t *NGramTokenizer) Min() uint32 { return t.minimum }

func (t *NGramTokenizer) Max() uint32 { return t.maximum }

func (t *NGramTokenizer) TokenChars() NGramTokenChars { return t.tokenChars }

type ngramUnicodeClassRange struct {
	first uint32
	last  uint32
	class NGramTokenChars
}

type ngramCodepointSpan struct {
	start uint32
	end   uint32
}

// Tokenize emits ngrams ordered first by start codepoint and then by
// increasing length. Malformed UTF-8 bytes always terminate a segment.
func (t *NGramTokenizer) Tokenize(ctx context.Context, text string) ([]Token, error) {
	if ctx == nil {
		return nil, errors.New("core: nil tokenizer context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(len(text)) > math.MaxUint32 {
		return nil, ErrTokenizerInputTooLarge
	}
	capacity := len(text)/4 + 1
	if capacity > 4096 {
		capacity = 4096
	}
	tokens := make([]Token, 0, capacity)
	segment := make([]ngramCodepointSpan, 0)
	position := uint32(0)

	emit := func() error {
		if err := emitNGramSegment(ctx, text, segment, t.minimum, t.maximum, &position, &tokens); err != nil {
			return err
		}
		segment = segment[:0]
		return nil
	}

	for index := 0; index < len(text); {
		if err := checkNGramContext(ctx, index); err != nil {
			return nil, err
		}
		value, size := utf8.DecodeRuneInString(text[index:])
		if value == utf8.RuneError && size == 1 {
			if err := emit(); err != nil {
				return nil, err
			}
			index++
			continue
		}
		if t.tokenChars == 0 || lookupNGramTokenChar(uint32(value))&t.tokenChars != 0 {
			segment = append(segment, ngramCodepointSpan{start: uint32(index), end: uint32(index + size)})
		} else if err := emit(); err != nil {
			return nil, err
		}
		index += size
	}
	if err := emit(); err != nil {
		return nil, err
	}
	return tokens, nil
}

func lookupNGramTokenChar(value uint32) NGramTokenChars {
	index := sort.Search(len(ngramUnicodeClassRanges), func(index int) bool {
		return ngramUnicodeClassRanges[index].last >= value
	})
	if index < len(ngramUnicodeClassRanges) && ngramUnicodeClassRanges[index].first <= value {
		return ngramUnicodeClassRanges[index].class
	}
	return 0
}

func emitNGramSegment(ctx context.Context, text string, segment []ngramCodepointSpan, minimum, maximum uint32, position *uint32, tokens *[]Token) error {
	if uint64(len(segment)) < uint64(minimum) {
		return nil
	}
	for start := 0; start < len(segment); start++ {
		if err := checkNGramContext(ctx, start); err != nil {
			return err
		}
		remaining := uint64(len(segment) - start)
		upper := uint64(maximum)
		if remaining < upper {
			upper = remaining
		}
		for length := uint64(minimum); length <= upper; length++ {
			first := segment[start]
			last := segment[start+int(length)-1]
			*tokens = append(*tokens, Token{
				Text:     strings.Clone(text[first.start:last.end]),
				Offset:   first.start,
				Position: *position,
			})
			*position++
		}
	}
	return nil
}

func checkNGramContext(ctx context.Context, progress int) error {
	if progress&4095 == 0 {
		return ctx.Err()
	}
	return nil
}

var _ Tokenizer = (*NGramTokenizer)(nil)
