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
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

type standardTokenizerFixture struct {
	BaselineCommit     string `json:"baseline_commit"`
	SourceSHA256       string `json:"source_sha256"`
	UnicodeTableSHA256 string `json:"unicode_table_sha256"`
	UnicodeVersion     string `json:"unicode_version"`
	Cases              []struct {
		Name           string `json:"name"`
		MaxTokenLength uint32 `json:"max_token_length"`
		InputHex       string `json:"input_hex"`
		Tokens         []struct {
			TextHex  string `json:"text_hex"`
			Offset   uint32 `json:"offset"`
			Position uint32 `json:"position"`
		} `json:"tokens"`
	} `json:"cases"`
}

func TestStandardTokenizerBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/standard_tokenizer_58375ff.json")
	require.NoError(t, err)

	var fixture standardTokenizerFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.SourceSHA256 == "3f9a7ee811e9fac253bac363d5b208b9e7a03cc1bb51b771221e9417bc8864e9")
	require.True(t, fixture.UnicodeTableSHA256 == "3d666796d24191c9708fb3a183d7ce0f61962da0b34dacfdc73d03061f342722")
	require.True(t, fixture.UnicodeVersion == "17.0.0")

	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			input, err := hex.DecodeString(test.InputHex)
			require.NoError(t, err)

			tokenizer := mustStandardTokenizer(t, test.MaxTokenLength)
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

func TestStandardTokenizerOptions(t *testing.T) {
	defaults := DefaultStandardTokenizerOptions()
	require.True(t, defaults.MaxTokenLength == 255)

	for _, length := range []uint32{1, 255, MaxStandardMaxTokenLength} {
		tokenizer, err := NewStandardTokenizer(StandardTokenizerOptions{MaxTokenLength: length})
		require.NoError(t, err)
		require.True(t, tokenizer.Name() == "standard")
		require.Equal(t, length, tokenizer.MaxTokenLength())
	}
	for _, length := range []uint32{0, MaxStandardMaxTokenLength + 1} {
		_, err := NewStandardTokenizer(StandardTokenizerOptions{MaxTokenLength: length})
		require.ErrorIs(t, err, ErrInvalidStandardTokenizerOptions)
	}
}

func TestStandardTokenizerUnicodeTablesAreOrdered(t *testing.T) {
	for _, ranges := range map[string][]standardUnicodeClassRange{
		"word-break": standardWordBreakRanges[:],
		"script":     standardScriptClassRanges[:],
	} {
		for index, item := range ranges {
			require.True(t, item.first <= item.last)
			require.True(t, item.last <= utf8.MaxRune)
			require.False(t, index > 0 && ranges[index-1].last >= item.first)
		}
	}
	for _, ranges := range map[string][]standardUnicodeRange{
		"extended-pictographic": standardExtendedPictographicRanges[:],
		"emoji-modifier-base":   standardEmojiModifierBaseRanges[:],
		"emoji-modifier":        standardEmojiModifierRanges[:],
		"complex-context":       standardLineBreakComplexContextRanges[:],
	} {
		for index, item := range ranges {
			require.True(t, item.first <= item.last)
			require.True(t, item.last <= utf8.MaxRune)
			require.False(t, index > 0 && ranges[index-1].last >= item.first)
		}
	}
}

func TestStandardTokenizerASCIIAndPunctuation(t *testing.T) {
	tokenizer := mustStandardTokenizer(t, 255)
	tests := []struct {
		input string
		want  []string
	}{
		{"hello world", []string{"hello", "world"}},
		{"hello,world!test", []string{"hello", "world", "test"}},
		{"abc123 xyz", []string{"abc123", "xyz"}},
		{"  .,;!  ", []string{}},
		{"dog's 3.14 1,000 example.com hello,world host:port a:b host:9200", []string{
			"dog's", "3.14", "1,000", "example.com", "hello", "world", "host:port", "a:b", "host", "9200",
		}},
		{"foo_bar v1_2 _lead __123 _", []string{"foo_bar", "v1_2", "_lead", "__123"}},
	}
	for _, test := range tests {
		got, err := tokenizer.Tokenize(context.Background(), test.input)
		require.NoError(t, err)
		{
			texts := standardTokenTexts(got)
			require.Equal(t, test.want, texts)
		}

		assertStandardTokenRanges(t, test.input, got)
	}
	empty, err := tokenizer.Tokenize(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
}

func TestStandardTokenizerUnicodeScriptsAndMarks(t *testing.T) {
	tokenizer := mustStandardTokenizer(t, 255)
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{"accented", "café résumé", []string{"café", "résumé"}},
		{"marks", "e\u0301 \u0301 \ufe0f", []string{"e\u0301"}},
		{"cyrillic", "Москва Россия", []string{"Москва", "Россия"}},
		{"cjk", "全文检索", []string{"全", "文", "检", "索"}},
		{"unicode17-cjk", "\U0002ebf0\U00031350\U000323b0\U0002f800", []string{"\U0002ebf0", "\U00031350", "\U000323b0", "\U0002f800"}},
		{"cjk-mark", "中\ufe00文", []string{"中\ufe00", "文"}},
		{"mixed", "hello世界test", []string{"hello", "世", "界", "test"}},
		{"hiragana", "か\u3099な", []string{"か\u3099", "な"}},
		{"scripts", "ひらがな カタカナ 한국 ไทย မန", []string{"ひ", "ら", "が", "な", "カタカナ", "한국", "ไทย", "မန"}},
		{"thai-marks", "กั ั", []string{"กั"}},
		{"hangul-symbol", "㉠ 한국", []string{"한국"}},
		{"hebrew", "א' א\"ב", []string{"א'", "א\"ב"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := tokenizer.Tokenize(context.Background(), test.input)
			require.NoError(t, err)
			{
				texts := standardTokenTexts(got)
				require.Equal(t, test.want, texts)
			}

			assertStandardTokenRanges(t, test.input, got)
		})
	}
}

func TestStandardTokenizerEmojiSequences(t *testing.T) {
	tokenizer := mustStandardTokenizer(t, 255)
	tests := []struct {
		input string
		want  []string
	}{
		{"👩‍💻 ❤️", []string{"👩‍💻", "❤️"}},
		{"1️⃣ #⃣ *️⃣", []string{"1️⃣", "#⃣", "*️⃣"}},
		{"👍🏽 ☝️🏻 🏽", []string{"👍🏽", "☝️🏻", "🏽"}},
		{"👩🏽‍💻", []string{"👩🏽‍💻"}},
		{"🇺🇸🇨🇦🇯", []string{"🇺🇸", "🇨🇦", "🇯"}},
		{"🇦\u0308🇧 🇦‍🇧🇨", []string{"🇦\u0308🇧", "🇦‍🇧", "🇨"}},
		{"\u200d🛑 a\u200d🛑 \u200dⓂ", []string{"\u200d🛑", "a\u200d🛑", "\u200dⓂ"}},
	}
	for _, test := range tests {
		got, err := tokenizer.Tokenize(context.Background(), test.input)
		require.NoError(t, err)
		{
			texts := standardTokenTexts(got)
			require.Equal(t, test.want, texts)
		}

		assertStandardTokenRanges(t, test.input, got)
	}
}

func TestStandardTokenizerMalformedUTF8(t *testing.T) {
	input := string([]byte{'a', 'b', 0xff, 0xc0, 'c', 'd'})
	tokens, err := mustStandardTokenizer(t, 255).Tokenize(context.Background(), input)
	require.NoError(t, err)

	want := []Token{{Text: "ab", Offset: 0, Position: 0}, {Text: "cd", Offset: 4, Position: 1}}
	require.Equal(t, want, tokens)
}

func TestStandardTokenizerMaxTokenLength(t *testing.T) {
	tests := []struct {
		length uint32
		input  string
		want   []string
	}{
		{5, "abcdefgh", []string{"abcde", "fgh"}},
		{4, "café", []string{"café"}},
		{3, "café", []string{"caf", "é"}},
		{2, "ab\u0301c", []string{"ab\u0301", "c"}},
		{3, "dog's", []string{"dog", "s"}},
		{1, "_lead", []string{"l", "e", "a", "d"}},
		{1, "abc__def", []string{"a", "b", "c", "d", "e", "f"}},
		{1, "a中bc", []string{"a", "中", "b", "c"}},
	}
	for _, test := range tests {
		tokens, err := mustStandardTokenizer(t, test.length).Tokenize(context.Background(), test.input)
		require.NoError(t, err)
		{
			texts := standardTokenTexts(tokens)
			require.Equal(t, test.want, texts)
		}

		assertStandardTokenRanges(t, test.input, tokens)
	}
}

func TestStandardTokenizerContextCancellation(t *testing.T) {
	tokenizer := mustStandardTokenizer(t, 255)
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

	for _, input := range []string{strings.Repeat("x", 64<<10), strings.Repeat("é", 32<<10)} {
		midway := newCancelAfterChecks(4)
		{
			_, err := tokenizer.Tokenize(midway, input)
			require.ErrorIs(t, err, context.Canceled)
		}
	}
}

func FuzzStandardTokenizer(f *testing.F) {
	for _, seed := range []string{"", "hello,world", "全文 café", "👩🏽‍💻 🇺🇸", string([]byte{'a', 0xff, 'b'})} {
		f.Add(seed, uint16(255))
	}
	f.Fuzz(func(t *testing.T, input string, length uint16) {
		maxLength := uint32(length%64) + 1
		tokens, err := mustStandardTokenizer(t, maxLength).Tokenize(context.Background(), input)
		require.NoError(t, err)

		assertStandardTokenRanges(t, input, tokens)
		for _, token := range tokens {
			require.False(t, utf8.ValidString(token.Text) && uint32(utf8.RuneCountInString(token.Text)) < 1)
		}
	})
}

func BenchmarkStandardTokenizer(b *testing.B) {
	text := strings.Repeat("The quick brown fox's vector is 3.14; 中文检索 👩🏽‍💻. ", 1024)
	tokenizer := mustStandardTokenizer(b, 255)
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

func mustStandardTokenizer(tb testing.TB, maxTokenLength uint32) *StandardTokenizer {
	tb.Helper()
	tokenizer, err := NewStandardTokenizer(StandardTokenizerOptions{MaxTokenLength: maxTokenLength})
	require.NoError(tb, err)

	return tokenizer
}

func standardTokenTexts(tokens []Token) []string {
	texts := make([]string, len(tokens))
	for index := range tokens {
		texts[index] = tokens[index].Text
	}
	return texts
}

func assertStandardTokenRanges(t testing.TB, input string, tokens []Token) {
	t.Helper()
	lastEnd := 0
	for position, token := range tokens {
		start := int(token.Offset)
		end := start + len(token.Text)
		require.Equal(t, uint32(position), token.Position)
		require.False(t, token.Text == "")
		require.True(t, start >= lastEnd)
		require.True(t, end <= len(input))
		require.Equal(t, token.Text, input[start:end])

		lastEnd = end
	}
}
