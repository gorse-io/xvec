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
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

type ngramTokenizerFixture struct {
	BaselineCommit    string `json:"baseline_commit"`
	SourceSHA256      string `json:"source_sha256"`
	UnicodeProvider   string `json:"unicode_provider"`
	UnicodeDataSHA256 string `json:"unicode_data_sha256"`
	UnicodeVersion    string `json:"unicode_version"`
	Cases             []struct {
		Name       string          `json:"name"`
		Min        uint32          `json:"min"`
		Max        uint32          `json:"max"`
		TokenChars NGramTokenChars `json:"token_chars"`
		InputHex   string          `json:"input_hex"`
		Tokens     []struct {
			TextHex  string `json:"text_hex"`
			Offset   uint32 `json:"offset"`
			Position uint32 `json:"position"`
		} `json:"tokens"`
	} `json:"cases"`
}

func TestNGramTokenizerBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/ngram_tokenizer_58375ff.json")
	require.NoError(t, err)

	var fixture ngramTokenizerFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.SourceSHA256 == "106f58aa34dc0a3d718e1131656edffcadfca51d5700628287259be2986e7785")
	require.True(t, fixture.UnicodeProvider == "utf8proc 2.11.3 e5e799221b45bbb90f5fdc5c69b6b8dfbf017e78")
	require.True(t, fixture.UnicodeDataSHA256 == "950e549dbfc853c4304425f3af1875e72fa9fc9697c273c763400c2da4e380a7")
	require.True(t, fixture.UnicodeVersion == "17.0.0")

	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			input, err := hex.DecodeString(test.InputHex)
			require.NoError(t, err)

			tokenizer := mustNGramTokenizer(t, NGramTokenizerOptions{Min: test.Min, Max: test.Max, TokenChars: test.TokenChars})
			got, err := tokenizer.Tokenize(context.Background(), string(input))
			require.NoError(t, err)

			want := make([]Token, len(test.Tokens))
			for index, token := range test.Tokens {
				text, err := hex.DecodeString(token.TextHex)
				require.NoError(t, err)

				want[index] = Token{Text: string(text), Offset: token.Offset, Position: token.Position}
			}
			require.Equal(t, want, got)
		})
	}
}

func TestNGramTokenizerOptions(t *testing.T) {
	defaults := DefaultNGramTokenizerOptions()
	require.Equal(t, NGramTokenizerOptions{Min: 2, Max: 2}, defaults)

	valid := []NGramTokenizerOptions{
		{Min: 1, Max: 1},
		{Min: 2, Max: 3, TokenChars: NGramTokenCharLetter | NGramTokenCharDigit},
		{Min: math.MaxUint32 - 1, Max: math.MaxUint32},
	}
	for _, options := range valid {
		tokenizer, err := NewNGramTokenizer(options)
		require.NoError(t, err)
		require.True(t, tokenizer.Name() == "ngram")
		require.Equal(t, options.Min, tokenizer.Min())
		require.Equal(t, options.Max, tokenizer.Max())
		require.Equal(t, options.TokenChars, tokenizer.TokenChars())
	}
	invalid := []NGramTokenizerOptions{
		{},
		{Min: 1},
		{Max: 1},
		{Min: 3, Max: 2},
		{Min: 1, Max: 3},
		{Min: 2, Max: 2, TokenChars: 1 << 31},
	}
	for _, options := range invalid {
		_, err := NewNGramTokenizer(options)
		require.ErrorIs(t, err, ErrInvalidNGramTokenizerOptions)
	}
}

func TestNGramTokenizerPinnedBehavior(t *testing.T) {
	tests := []struct {
		name    string
		options NGramTokenizerOptions
		input   string
		want    []string
	}{
		{"default", DefaultNGramTokenizerOptions(), "hello", []string{"he", "el", "ll", "lo"}},
		{"mixed-length", NGramTokenizerOptions{Min: 2, Max: 3}, "hello", []string{"he", "hel", "el", "ell", "ll", "llo", "lo"}},
		{"chinese", DefaultNGramTokenizerOptions(), "中文分词", []string{"中文", "文分", "分词"}},
		{"all-valid", DefaultNGramTokenizerOptions(), "foobar未跟踪文件", []string{"fo", "oo", "ob", "ba", "ar", "r未", "未跟", "跟踪", "踪文", "文件"}},
		{"letter-digit", NGramTokenizerOptions{Min: 2, Max: 2, TokenChars: NGramTokenCharLetter | NGramTokenCharDigit}, "ab cd,中文!de", []string{"ab", "cd", "中文", "de"}},
		{"space-punctuation-symbol", NGramTokenizerOptions{Min: 2, Max: 2, TokenChars: NGramTokenCharWhitespace | NGramTokenCharPunctuation | NGramTokenCharSymbol}, "a !$b", []string{" !", "!$"}},
		{"short-segments", NGramTokenizerOptions{Min: 2, Max: 2, TokenChars: NGramTokenCharLetter}, "a b ! c", []string{}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tokenizer := mustNGramTokenizer(t, test.options)
			got, err := tokenizer.Tokenize(context.Background(), test.input)
			require.NoError(t, err)
			{
				texts := ngramTokenTexts(got)
				require.Equal(t, test.want, texts)
			}

			assertNGramTokenRanges(t, test.input, got, test.options)
		})
	}
}

func TestNGramTokenizerNULAndMalformedUTF8(t *testing.T) {
	tokenizer := mustNGramTokenizer(t, DefaultNGramTokenizerOptions())
	nulInput := "a\x00b"
	nulTokens, err := tokenizer.Tokenize(context.Background(), nulInput)
	require.NoError(t, err)
	{
		want := []string{"a\x00", "\x00b"}
		require.Equal(t, want, ngramTokenTexts(nulTokens))
	}

	malformed := string([]byte{'a', 'b', 0xff, 'c', 'd'})
	malformedTokens, err := tokenizer.Tokenize(context.Background(), malformed)
	require.NoError(t, err)
	{
		want := []Token{{Text: "ab", Offset: 0, Position: 0}, {Text: "cd", Offset: 3, Position: 1}}
		require.Equal(t, want, malformedTokens)
	}
}

func TestNGramTokenizerUnicode17Classes(t *testing.T) {
	whitespace := mustNGramTokenizer(t, NGramTokenizerOptions{Min: 1, Max: 1, TokenChars: NGramTokenCharWhitespace})
	input := "\t \n\u2028\u00a0\u2007\u202f"
	tokens, err := whitespace.Tokenize(context.Background(), input)
	require.NoError(t, err)
	{
		want := []string{"\t", " ", "\n", "\u2028"}
		require.Equal(t, want, ngramTokenTexts(tokens))
	}

	symbol := mustNGramTokenizer(t, NGramTokenizerOptions{Min: 1, Max: 1, TokenChars: NGramTokenCharSymbol})
	symbolInput := "$\u0301\x01√"
	tokens, err = symbol.Tokenize(context.Background(), symbolInput)
	require.NoError(t, err)
	{
		want := []string{"$", "√"}
		require.Equal(t, want, ngramTokenTexts(tokens))
	}

	// Unicode 17 additions prove that classification does not depend on Go's
	// older standard-library Unicode tables: U+1C89 is Lu and U+1C8A is Ll.
	letters := mustNGramTokenizer(t, NGramTokenizerOptions{Min: 1, Max: 1, TokenChars: NGramTokenCharLetter})
	tokens, err = letters.Tokenize(context.Background(), "\u1c89\u1c8a")
	require.NoError(t, err)
	{
		want := []string{"\u1c89", "\u1c8a"}
		require.Equal(t, want, ngramTokenTexts(tokens))
	}
}

func TestNGramTokenizerPositionsAcrossSegments(t *testing.T) {
	tokenizer := mustNGramTokenizer(t, NGramTokenizerOptions{Min: 2, Max: 2, TokenChars: NGramTokenCharLetter})
	input := "abcd 中文分"
	tokens, err := tokenizer.Tokenize(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, tokens, 5)

	assertNGramTokenRanges(t, input, tokens, NGramTokenizerOptions{Min: 2, Max: 2, TokenChars: NGramTokenCharLetter})
}

func TestNGramTokenizerUnicodeTableIsOrdered(t *testing.T) {
	require.Len(t, ngramUnicodeClassRanges, 1205)

	for index, item := range ngramUnicodeClassRanges {
		require.True(t, item.first <= item.last)
		require.True(t, item.last <= utf8.MaxRune)
		require.False(t, item.class == 0)
		require.True(t, item.class&^ngramTokenCharAll == 0)
		require.False(t, index > 0 && ngramUnicodeClassRanges[index-1].last >= item.first)
	}
}

func TestNGramTokenizerContextCancellation(t *testing.T) {
	tokenizer := mustNGramTokenizer(t, DefaultNGramTokenizerOptions())
	{
		_, err := tokenizer.Tokenize(nil, "text")
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := tokenizer.Tokenize(canceled, "text")
		require.ErrorIs(t, err, context.Canceled)
	}

	midway := newCancelAfterChecks(4)
	{
		_, err := tokenizer.Tokenize(midway, strings.Repeat("x", 64<<10))
		require.ErrorIs(t, err, context.Canceled)
	}
}

func FuzzNGramTokenizer(f *testing.F) {
	for _, seed := range []string{"", "hello", "全文检索", "a\x00b", string([]byte{'a', 0xff, 'b'}), "$\u2028é"} {
		f.Add(seed, uint8(2), uint8(0), uint8(0))
	}
	f.Fuzz(func(t *testing.T, input string, minimumSeed, rangeSeed, maskSeed uint8) {
		minimum := uint32(minimumSeed%8) + 1
		maximum := minimum + uint32(rangeSeed&1)
		options := NGramTokenizerOptions{Min: minimum, Max: maximum, TokenChars: NGramTokenChars(maskSeed & uint8(ngramTokenCharAll))}
		tokens, err := mustNGramTokenizer(t, options).Tokenize(context.Background(), input)
		require.NoError(t, err)

		assertNGramTokenRanges(t, input, tokens, options)
	})
}

func BenchmarkNGramTokenizer(b *testing.B) {
	text := strings.Repeat("vector 中文 search 3.14 👩🏽‍💻 ", 1024)
	tokenizer := mustNGramTokenizer(b, NGramTokenizerOptions{Min: 2, Max: 3, TokenChars: NGramTokenCharLetter | NGramTokenCharDigit})
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	for b.Loop() {
		{
			_, err := tokenizer.Tokenize(context.Background(), text)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func mustNGramTokenizer(tb testing.TB, options NGramTokenizerOptions) *NGramTokenizer {
	tb.Helper()
	tokenizer, err := NewNGramTokenizer(options)
	require.NoError(tb, err)

	return tokenizer
}

func ngramTokenTexts(tokens []Token) []string {
	texts := make([]string, len(tokens))
	for index := range tokens {
		texts[index] = tokens[index].Text
	}
	return texts
}

func assertNGramTokenRanges(t testing.TB, input string, tokens []Token, options NGramTokenizerOptions) {
	t.Helper()
	previousOffset := uint32(0)
	for position, token := range tokens {
		start := int(token.Offset)
		end := start + len(token.Text)
		require.Equal(t, uint32(position), token.Position)
		require.False(t, token.Text == "")
		require.True(t, utf8.ValidString(token.Text))
		require.False(t, position > 0 && token.Offset < previousOffset)
		require.True(t, end <= len(input))
		require.Equal(t, token.Text, input[start:end])

		length := uint32(utf8.RuneCountInString(token.Text))
		require.True(t, length >= options.Min)
		require.True(t, length <= options.Max)

		if options.TokenChars != 0 {
			for _, value := range token.Text {
				require.False(t, lookupNGramTokenChar(uint32(value))&options.TokenChars == 0)
			}
		}
		previousOffset = token.Offset
	}
}
