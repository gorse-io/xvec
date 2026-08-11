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

package core

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
	// DefaultStandardMaxTokenLength is the pinned standard-tokenizer default.
	DefaultStandardMaxTokenLength uint32 = 255
	// MinStandardMaxTokenLength is the smallest accepted token length.
	MinStandardMaxTokenLength uint32 = 1
	// MaxStandardMaxTokenLength is the largest accepted token length.
	MaxStandardMaxTokenLength uint32 = 1_048_576

	standardVariationSelector16 uint32 = 0xFE0F
	standardKeycap              uint32 = 0x20E3
)

// ErrInvalidStandardTokenizerOptions identifies invalid standard tokenizer
// construction options.
var ErrInvalidStandardTokenizerOptions = errors.New("core: invalid standard tokenizer options")

// StandardTokenizerOptions configures StandardTokenizer. Use
// DefaultStandardTokenizerOptions when the caller does not supply a value.
type StandardTokenizerOptions struct {
	MaxTokenLength uint32
}

// DefaultStandardTokenizerOptions returns options compatible with zvec
// 58375ff.
func DefaultStandardTokenizerOptions() StandardTokenizerOptions {
	return StandardTokenizerOptions{MaxTokenLength: DefaultStandardMaxTokenLength}
}

// Validate checks the baseline max_token_length range.
func (o StandardTokenizerOptions) Validate() error {
	if o.MaxTokenLength < MinStandardMaxTokenLength || o.MaxTokenLength > MaxStandardMaxTokenLength {
		return fmt.Errorf("%w: MaxTokenLength must be in [%d, %d]", ErrInvalidStandardTokenizerOptions, MinStandardMaxTokenLength, MaxStandardMaxTokenLength)
	}
	return nil
}

// StandardTokenizer implements the Unicode 17 standard tokenizer behavior
// pinned by zvec commit 58375ff.
type StandardTokenizer struct {
	maxTokenLength uint32
}

// NewStandardTokenizer constructs a validated tokenizer.
func NewStandardTokenizer(options StandardTokenizerOptions) (*StandardTokenizer, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}
	return &StandardTokenizer{maxTokenLength: options.MaxTokenLength}, nil
}

func (*StandardTokenizer) Name() string { return "standard" }

// MaxTokenLength returns the configured codepoint limit.
func (t *StandardTokenizer) MaxTokenLength() uint32 { return t.maxTokenLength }

// Tokenize emits owned token text with byte offsets into text and contiguous
// zero-based positions. MaxTokenLength counts decoded codepoints, not bytes.
func (t *StandardTokenizer) Tokenize(ctx context.Context, text string) ([]Token, error) {
	if ctx == nil {
		return nil, errors.New("core: nil tokenizer context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(len(text)) > math.MaxUint32 {
		return nil, ErrTokenizerInputTooLarge
	}
	if isASCIIText(text) {
		return tokenizeStandardASCII(ctx, text, t.maxTokenLength)
	}

	codepoints, err := decodeStandardUTF8(ctx, text)
	if err != nil {
		return nil, err
	}
	tokens := make([]Token, 0)
	position := uint32(0)
	for index := 0; index < len(codepoints); {
		if err := checkStandardContext(ctx, index); err != nil {
			return nil, err
		}
		class := codepoints[index].class
		var end int
		switch class {
		case wordBreakIdeographic, wordBreakHiragana:
			end = scanStandardSingleToken(codepoints, index)
		case wordBreakHangul, wordBreakSoutheastAsian:
			end = scanStandardWordToken(codepoints, index)
		case wordBreakRegionalIndicator:
			end = scanStandardRegionalIndicatorToken(codepoints, index)
		default:
			end = scanStandardKeycapToken(codepoints, index)
			if end == index && class == wordBreakExtendedPictographic {
				end = scanStandardEmojiToken(codepoints, index)
			}
			if end == index && class == wordBreakZWJ {
				end = scanStandardZWJExtendedPictographicToken(codepoints, index)
			}
			if end == index {
				end = scanStandardEmojiModifierToken(codepoints, index)
			}
			if end == index && class == wordBreakExtendNumLet {
				end = scanStandardWordToken(codepoints, index)
				if !standardSpanHasCoreToken(codepoints, index, end) {
					index = end
					continue
				}
			}
			if end == index {
				if !standardIsTokenStart(class) {
					index++
					continue
				}
				end = scanStandardWordToken(codepoints, index)
			}
		}
		if err := emitStandardTokenSpan(ctx, text, codepoints, index, end, t.maxTokenLength, &position, &tokens); err != nil {
			return nil, err
		}
		index = end
	}
	return tokens, nil
}

type standardWordBreakClass uint8

const (
	wordBreakOther standardWordBreakClass = iota
	wordBreakCR
	wordBreakLF
	wordBreakNewline
	wordBreakExtend
	wordBreakFormat
	wordBreakZWJ
	wordBreakALetter
	wordBreakHebrewLetter
	wordBreakNumeric
	wordBreakKatakana
	wordBreakExtendNumLet
	wordBreakRegionalIndicator
	wordBreakWSegSpace
	wordBreakMidLetter
	wordBreakMidNum
	wordBreakMidNumLet
	wordBreakSingleQuote
	wordBreakDoubleQuote
	wordBreakHiragana
	wordBreakHangul
	wordBreakSoutheastAsian
	wordBreakIdeographic
	wordBreakExtendedPictographic
)

type standardUnicodeClassRange struct {
	first uint32
	last  uint32
	class standardWordBreakClass
}

type standardUnicodeRange struct {
	first uint32
	last  uint32
}

type standardCodepoint struct {
	value                uint32
	start                uint32
	end                  uint32
	class                standardWordBreakClass
	extendedPictographic bool
	emojiModifierBase    bool
	emojiModifier        bool
}

type standardCodepointProperties struct {
	class                standardWordBreakClass
	extendedPictographic bool
	emojiModifierBase    bool
	emojiModifier        bool
}

func lookupStandardClassRange(ranges []standardUnicodeClassRange, value uint32) standardWordBreakClass {
	index := sort.Search(len(ranges), func(index int) bool { return ranges[index].last >= value })
	if index < len(ranges) && ranges[index].first <= value {
		return ranges[index].class
	}
	return wordBreakOther
}

func containsStandardRange(ranges []standardUnicodeRange, value uint32) bool {
	index := sort.Search(len(ranges), func(index int) bool { return ranges[index].last >= value })
	return index < len(ranges) && ranges[index].first <= value
}

func standardASCIIClass(value uint32) standardWordBreakClass {
	switch {
	case value == '\n':
		return wordBreakLF
	case value == '\r':
		return wordBreakCR
	case value == '\v' || value == '\f':
		return wordBreakNewline
	case value == ' ':
		return wordBreakWSegSpace
	case value == '"':
		return wordBreakDoubleQuote
	case value == '\'':
		return wordBreakSingleQuote
	case value == ',':
		return wordBreakMidNum
	case value == '.':
		return wordBreakMidNumLet
	case value == ':':
		return wordBreakMidLetter
	case value == ';':
		return wordBreakMidNum
	case value == '_':
		return wordBreakExtendNumLet
	case value >= '0' && value <= '9':
		return wordBreakNumeric
	case value >= 'A' && value <= 'Z', value >= 'a' && value <= 'z':
		return wordBreakALetter
	default:
		return wordBreakOther
	}
}

func classifyStandardCodepoint(value uint32) standardWordBreakClass {
	if value <= 0x7f {
		return standardASCIIClass(value)
	}
	scriptClass := lookupStandardClassRange(standardScriptClassRanges[:], value)
	if scriptClass == wordBreakIdeographic || scriptClass == wordBreakHiragana {
		return scriptClass
	}
	if value >= 0xac00 && value <= 0xd7a3 {
		return wordBreakHangul
	}
	wordBreakClass := lookupStandardClassRange(standardWordBreakRanges[:], value)
	if wordBreakClass == wordBreakALetter || wordBreakClass == wordBreakKatakana || wordBreakClass == wordBreakOther {
		if containsStandardRange(standardLineBreakComplexContextRanges[:], value) {
			return wordBreakSoutheastAsian
		}
		if scriptClass != wordBreakOther {
			if scriptClass == wordBreakHangul && !standardIsAHLetter(wordBreakClass) {
				return wordBreakClass
			}
			return scriptClass
		}
	}
	if wordBreakClass != wordBreakOther {
		return wordBreakClass
	}
	if containsStandardRange(standardExtendedPictographicRanges[:], value) {
		return wordBreakExtendedPictographic
	}
	return wordBreakOther
}

func lookupStandardCodepointProperties(value uint32) standardCodepointProperties {
	properties := standardCodepointProperties{class: classifyStandardCodepoint(value)}
	properties.extendedPictographic = properties.class == wordBreakExtendedPictographic
	if !properties.extendedPictographic && value >= 0x00a9 {
		properties.extendedPictographic = containsStandardRange(standardExtendedPictographicRanges[:], value)
	}
	if value >= 0x261d {
		properties.emojiModifierBase = containsStandardRange(standardEmojiModifierBaseRanges[:], value)
	}
	if value >= 0x1f3fb {
		properties.emojiModifier = containsStandardRange(standardEmojiModifierRanges[:], value)
	}
	return properties
}

func decodeStandardUTF8(ctx context.Context, text string) ([]standardCodepoint, error) {
	codepoints := make([]standardCodepoint, 0, len(text)/2+1)
	for index := 0; index < len(text); {
		if err := checkStandardContext(ctx, len(codepoints)); err != nil {
			return nil, err
		}
		value, size := utf8.DecodeRuneInString(text[index:])
		if value == utf8.RuneError && size == 1 {
			codepoints = append(codepoints, standardCodepoint{start: uint32(index), end: uint32(index + 1), class: wordBreakOther})
			index++
			continue
		}
		properties := lookupStandardCodepointProperties(uint32(value))
		codepoints = append(codepoints, standardCodepoint{
			value:                uint32(value),
			start:                uint32(index),
			end:                  uint32(index + size),
			class:                properties.class,
			extendedPictographic: properties.extendedPictographic,
			emojiModifierBase:    properties.emojiModifierBase,
			emojiModifier:        properties.emojiModifier,
		})
		index += size
	}
	return codepoints, nil
}

func standardIsAHLetter(class standardWordBreakClass) bool {
	return class == wordBreakALetter || class == wordBreakHebrewLetter
}

func standardIsIgnored(class standardWordBreakClass) bool {
	return class == wordBreakExtend || class == wordBreakFormat || class == wordBreakZWJ
}

func standardIsExtendOrFormat(class standardWordBreakClass) bool {
	return class == wordBreakExtend || class == wordBreakFormat
}

func standardIsExtendedPictographic(codepoint standardCodepoint) bool {
	return codepoint.class == wordBreakExtendedPictographic || codepoint.extendedPictographic
}

func standardIsTokenStart(class standardWordBreakClass) bool {
	return standardIsAHLetter(class) || class == wordBreakNumeric || class == wordBreakKatakana ||
		class == wordBreakRegionalIndicator || class == wordBreakHiragana || class == wordBreakHangul ||
		class == wordBreakSoutheastAsian || class == wordBreakIdeographic || class == wordBreakExtendedPictographic
}

func standardIsConnector(class standardWordBreakClass) bool {
	return standardIsAHLetter(class) || class == wordBreakNumeric || class == wordBreakKatakana || class == wordBreakExtendNumLet
}

func standardNextSignificant(codepoints []standardCodepoint, index int) int {
	for index < len(codepoints) && standardIsIgnored(codepoints[index].class) {
		index++
	}
	return index
}

func standardPunctuationConnects(left, punctuation, right standardWordBreakClass) bool {
	if standardIsAHLetter(left) && standardIsAHLetter(right) &&
		(punctuation == wordBreakMidLetter || punctuation == wordBreakMidNumLet || punctuation == wordBreakSingleQuote) {
		return true
	}
	if left == wordBreakHebrewLetter && right == wordBreakHebrewLetter && punctuation == wordBreakDoubleQuote {
		return true
	}
	return left == wordBreakNumeric && right == wordBreakNumeric &&
		(punctuation == wordBreakMidNum || punctuation == wordBreakMidNumLet || punctuation == wordBreakSingleQuote)
}

func standardSignificantConnects(left, right standardWordBreakClass) bool {
	if standardIsAHLetter(left) && standardIsAHLetter(right) {
		return true
	}
	if standardIsAHLetter(left) && right == wordBreakNumeric || left == wordBreakNumeric && standardIsAHLetter(right) {
		return true
	}
	if left == wordBreakNumeric && right == wordBreakNumeric || left == wordBreakKatakana && right == wordBreakKatakana {
		return true
	}
	if standardIsConnector(left) && right == wordBreakExtendNumLet || left == wordBreakExtendNumLet && standardIsConnector(right) {
		return true
	}
	if left == wordBreakExtendNumLet && right == wordBreakExtendNumLet {
		return true
	}
	return left == wordBreakHangul && right == wordBreakHangul ||
		left == wordBreakSoutheastAsian && right == wordBreakSoutheastAsian
}

func scanStandardWordToken(codepoints []standardCodepoint, start int) int {
	end := start + 1
	lastSignificant := start
	for end < len(codepoints) {
		class := codepoints[end].class
		if standardIsIgnored(class) {
			end++
			continue
		}
		if class == wordBreakExtendNumLet {
			if !standardSignificantConnects(codepoints[lastSignificant].class, class) {
				break
			}
			lastSignificant = end
			end++
			continue
		}
		if class == wordBreakSingleQuote && codepoints[lastSignificant].class == wordBreakHebrewLetter {
			lastSignificant = end
			end++
			continue
		}
		if standardIsExtendedPictographic(codepoints[end]) && codepoints[end-1].class == wordBreakZWJ {
			end = scanStandardEmojiToken(codepoints, end)
			lastSignificant = end - 1
			continue
		}
		if class == wordBreakIdeographic || class == wordBreakExtendedPictographic || class == wordBreakRegionalIndicator || !standardIsTokenStart(class) {
			right := standardNextSignificant(codepoints, end+1)
			if right < len(codepoints) && standardPunctuationConnects(codepoints[lastSignificant].class, class, codepoints[right].class) {
				end = right + 1
				lastSignificant = right
				continue
			}
			break
		}
		if !standardSignificantConnects(codepoints[lastSignificant].class, class) {
			break
		}
		lastSignificant = end
		end++
	}
	return end
}

func standardIsKeycapBase(codepoint standardCodepoint) bool {
	return codepoint.value >= '0' && codepoint.value <= '9' || codepoint.value == '#' || codepoint.value == '*'
}

func consumeStandardExtendOrFormat(codepoints []standardCodepoint, index int) int {
	for index < len(codepoints) && standardIsExtendOrFormat(codepoints[index].class) {
		index++
	}
	return index
}

func consumeStandardExtendFormatAndModifier(codepoints []standardCodepoint, index int) int {
	index = consumeStandardExtendOrFormat(codepoints, index)
	if index < len(codepoints) && codepoints[index].emojiModifier {
		index = consumeStandardExtendOrFormat(codepoints, index+1)
	}
	return index
}

func scanStandardKeycapToken(codepoints []standardCodepoint, start int) int {
	if !standardIsKeycapBase(codepoints[start]) {
		return start
	}
	index := start + 1
	if index < len(codepoints) && codepoints[index].value == standardVariationSelector16 {
		index++
	}
	if index >= len(codepoints) || codepoints[index].value != standardKeycap {
		return start
	}
	return consumeStandardExtendOrFormat(codepoints, index+1)
}

func scanStandardEmojiModifierToken(codepoints []standardCodepoint, start int) int {
	if codepoints[start].emojiModifier {
		return consumeStandardExtendOrFormat(codepoints, start+1)
	}
	if !codepoints[start].emojiModifierBase {
		return start
	}
	index := start + 1
	for index < len(codepoints) && standardIsExtendOrFormat(codepoints[index].class) && !codepoints[index].emojiModifier {
		index++
	}
	if index >= len(codepoints) || !codepoints[index].emojiModifier {
		return start
	}
	return consumeStandardExtendOrFormat(codepoints, index+1)
}

func scanStandardEmojiToken(codepoints []standardCodepoint, start int) int {
	index := consumeStandardExtendFormatAndModifier(codepoints, start+1)
	for index < len(codepoints) {
		if codepoints[index].class != wordBreakZWJ {
			break
		}
		index++
		for index < len(codepoints) && standardIsExtendOrFormat(codepoints[index].class) {
			index++
		}
		if index >= len(codepoints) || !standardIsExtendedPictographic(codepoints[index]) {
			break
		}
		index = consumeStandardExtendFormatAndModifier(codepoints, index+1)
	}
	return index
}

func scanStandardZWJExtendedPictographicToken(codepoints []standardCodepoint, start int) int {
	index := start + 1
	if index >= len(codepoints) || !standardIsExtendedPictographic(codepoints[index]) {
		return start
	}
	return scanStandardEmojiToken(codepoints, index)
}

func scanStandardRegionalIndicatorToken(codepoints []standardCodepoint, start int) int {
	index := standardNextSignificant(codepoints, start+1)
	if index < len(codepoints) && codepoints[index].class == wordBreakRegionalIndicator {
		index = standardNextSignificant(codepoints, index+1)
	}
	return index
}

func scanStandardSingleToken(codepoints []standardCodepoint, start int) int {
	return standardNextSignificant(codepoints, start+1)
}

func standardSpanHasCoreToken(codepoints []standardCodepoint, start, end int) bool {
	for index := start; index < end; index++ {
		if standardIsTokenStart(codepoints[index].class) {
			return true
		}
	}
	return false
}

func trimStandardNonCoreSuffix(codepoints []standardCodepoint, start, end int) int {
	for end > start {
		class := codepoints[end-1].class
		if standardIsTokenStart(class) || standardIsIgnored(class) {
			break
		}
		end--
	}
	return end
}

func emitStandardCoreSpan(text string, codepoints []standardCodepoint, start, end int, position *uint32, tokens *[]Token) {
	if start >= end || !standardSpanHasCoreToken(codepoints, start, end) {
		return
	}
	byteStart := codepoints[start].start
	byteEnd := codepoints[end-1].end
	*tokens = append(*tokens, Token{Text: strings.Clone(text[byteStart:byteEnd]), Offset: byteStart, Position: *position})
	*position++
}

func emitStandardTokenSpan(ctx context.Context, text string, codepoints []standardCodepoint, start, end int, maxTokenLength uint32, position *uint32, tokens *[]Token) error {
	if uint64(end-start) <= uint64(maxTokenLength) {
		byteStart := codepoints[start].start
		byteEnd := codepoints[end-1].end
		*tokens = append(*tokens, Token{Text: strings.Clone(text[byteStart:byteEnd]), Offset: byteStart, Position: *position})
		*position++
		return nil
	}
	tokenStart := start
	codepointCount := uint32(0)
	for index := start; index < end; {
		if err := checkStandardContext(ctx, index); err != nil {
			return err
		}
		if codepointCount >= maxTokenLength && standardIsTokenStart(codepoints[index].class) {
			emitStandardCoreSpan(text, codepoints, tokenStart, trimStandardNonCoreSuffix(codepoints, tokenStart, index), position, tokens)
			tokenStart = index
			codepointCount = 0
			continue
		}
		codepointCount++
		index++
	}
	tokenEnd := end
	if uint64(end-tokenStart) > uint64(maxTokenLength) {
		tokenEnd = trimStandardNonCoreSuffix(codepoints, tokenStart, end)
	}
	emitStandardCoreSpan(text, codepoints, tokenStart, tokenEnd, position, tokens)
	return nil
}

func isASCIIText(text string) bool {
	for index := 0; index < len(text); index++ {
		if text[index]&0x80 != 0 {
			return false
		}
	}
	return true
}

func standardASCIIIsLetterOrDigit(value byte) bool {
	return value >= '0' && value <= '9' || value >= 'A' && value <= 'Z' || value >= 'a' && value <= 'z'
}

func standardASCIIIsWordBody(value byte) bool {
	return standardASCIIIsLetterOrDigit(value) || value == '_'
}

func scanStandardASCIIWordToken(ctx context.Context, text string, start int) (int, error) {
	end := start + 1
	lastSignificant := start
	for end < len(text) {
		if err := checkStandardContext(ctx, end); err != nil {
			return 0, err
		}
		value := text[end]
		if standardASCIIIsWordBody(value) {
			lastSignificant = end
			end++
			continue
		}
		class := standardASCIIClass(uint32(value))
		if class == wordBreakExtendNumLet {
			if !standardSignificantConnects(standardASCIIClass(uint32(text[lastSignificant])), class) {
				break
			}
			lastSignificant = end
			end++
			continue
		}
		if !standardIsTokenStart(class) {
			right := end + 1
			if right < len(text) && standardPunctuationConnects(
				standardASCIIClass(uint32(text[lastSignificant])), class, standardASCIIClass(uint32(text[right])),
			) {
				end = right + 1
				lastSignificant = right
				continue
			}
			break
		}
		left := standardASCIIClass(uint32(text[lastSignificant]))
		if !standardSignificantConnects(left, class) {
			break
		}
		lastSignificant = end
		end++
	}
	return end, nil
}

func standardASCIISpanHasCore(text string, start, end int) bool {
	for index := start; index < end; index++ {
		if standardASCIIIsLetterOrDigit(text[index]) {
			return true
		}
	}
	return false
}

func trimStandardASCIINonCoreSuffix(text string, start, end int) int {
	for end > start && !standardASCIIIsLetterOrDigit(text[end-1]) {
		end--
	}
	return end
}

func emitStandardASCIICoreSpan(text string, start, end int, position *uint32, tokens *[]Token) {
	if start >= end || !standardASCIISpanHasCore(text, start, end) {
		return
	}
	*tokens = append(*tokens, Token{Text: strings.Clone(text[start:end]), Offset: uint32(start), Position: *position})
	*position++
}

func emitStandardASCIITokenSpan(ctx context.Context, text string, start, end int, maxTokenLength uint32, position *uint32, tokens *[]Token) error {
	if uint64(end-start) <= uint64(maxTokenLength) {
		*tokens = append(*tokens, Token{Text: strings.Clone(text[start:end]), Offset: uint32(start), Position: *position})
		*position++
		return nil
	}
	tokenStart := start
	codepointCount := uint32(0)
	for index := start; index < end; {
		if err := checkStandardContext(ctx, index); err != nil {
			return err
		}
		if codepointCount >= maxTokenLength && standardASCIIIsLetterOrDigit(text[index]) {
			emitStandardASCIICoreSpan(text, tokenStart, trimStandardASCIINonCoreSuffix(text, tokenStart, index), position, tokens)
			tokenStart = index
			codepointCount = 0
			continue
		}
		codepointCount++
		index++
	}
	tokenEnd := end
	if uint64(end-tokenStart) > uint64(maxTokenLength) {
		tokenEnd = trimStandardASCIINonCoreSuffix(text, tokenStart, end)
	}
	emitStandardASCIICoreSpan(text, tokenStart, tokenEnd, position, tokens)
	return nil
}

func tokenizeStandardASCII(ctx context.Context, text string, maxTokenLength uint32) ([]Token, error) {
	tokens := make([]Token, 0)
	position := uint32(0)
	for index := 0; index < len(text); {
		if err := checkStandardContext(ctx, index); err != nil {
			return nil, err
		}
		value := text[index]
		if standardASCIIIsLetterOrDigit(value) {
			end, err := scanStandardASCIIWordToken(ctx, text, index)
			if err != nil {
				return nil, err
			}
			if err := emitStandardASCIITokenSpan(ctx, text, index, end, maxTokenLength, &position, &tokens); err != nil {
				return nil, err
			}
			index = end
			continue
		}
		if standardASCIIClass(uint32(value)) == wordBreakExtendNumLet {
			end, err := scanStandardASCIIWordToken(ctx, text, index)
			if err != nil {
				return nil, err
			}
			if standardASCIISpanHasCore(text, index, end) {
				if err := emitStandardASCIITokenSpan(ctx, text, index, end, maxTokenLength, &position, &tokens); err != nil {
					return nil, err
				}
			}
			index = end
			continue
		}
		index++
	}
	return tokens, nil
}

func checkStandardContext(ctx context.Context, progress int) error {
	if progress&4095 == 0 {
		return ctx.Err()
	}
	return nil
}

var _ Tokenizer = (*StandardTokenizer)(nil)
