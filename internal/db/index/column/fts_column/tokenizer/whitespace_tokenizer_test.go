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
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

type whitespaceTokenizerFixture struct {
	BaselineCommit string `json:"baseline_commit"`
	Cases          []struct {
		Name     string `json:"name"`
		InputHex string `json:"input_hex"`
		Tokens   []struct {
			TextHex  string `json:"text_hex"`
			Offset   uint32 `json:"offset"`
			Position uint32 `json:"position"`
		} `json:"tokens"`
	} `json:"cases"`
}

func TestWhitespaceTokenizerBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/whitespace_tokenizer_58375ff.json")
	require.NoError(t, err)

	var fixture whitespaceTokenizerFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")

	tokenizer := NewWhitespaceTokenizer()
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			input, err := hex.DecodeString(test.InputHex)
			require.NoError(t, err)

			want := make([]Token, len(test.Tokens))
			for index, token := range test.Tokens {
				text, err := hex.DecodeString(token.TextHex)
				require.NoError(t, err)

				want[index] = Token{Text: string(text), Offset: token.Offset, Position: token.Position}
			}
			got, err := tokenizer.Tokenize(context.Background(), string(input))
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
}

func TestWhitespaceTokenizerPinnedByteSemantics(t *testing.T) {
	tokenizer := NewWhitespaceTokenizer()
	require.True(t, tokenizer.Name() == "whitespace")

	input := " \talpha\nBETA\v中文\fC++\r\nlast "
	want := []Token{
		{Text: "alpha", Offset: 2, Position: 0},
		{Text: "BETA", Offset: 8, Position: 1},
		{Text: "中文", Offset: 13, Position: 2},
		{Text: "C++", Offset: 20, Position: 3},
		{Text: "last", Offset: 25, Position: 4},
	}
	got, err := tokenizer.Tokenize(context.Background(), input)
	require.NoError(t, err)
	require.Equal(t, want, got)

	for _, input := range []string{"", " ", "\t\n\v\f\r"} {
		got, err := tokenizer.Tokenize(context.Background(), input)
		require.NoError(t, err)
		require.NotNil(t, got)
		require.Len(t, got, 0)
	}
}

func TestWhitespaceTokenizerPreservesUnicodeSpacesPunctuationAndInvalidUTF8(t *testing.T) {
	tokenizer := NewWhitespaceTokenizer()
	input := "a\u00a0b\u2003c +plus foo-bar \xffx"
	want := []Token{
		{Text: "a\u00a0b\u2003c", Offset: 0, Position: 0},
		{Text: "+plus", Offset: 9, Position: 1},
		{Text: "foo-bar", Offset: 15, Position: 2},
		{Text: "\xffx", Offset: 23, Position: 3},
	}
	got, err := tokenizer.Tokenize(context.Background(), input)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

func TestWhitespaceTokenizerContextCancellation(t *testing.T) {
	tokenizer := NewWhitespaceTokenizer()
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

	midway := newCancelAfterChecks(3)
	{
		_, err := tokenizer.Tokenize(midway, strings.Repeat("x", 32<<10))
		require.ErrorIs(t, err, context.Canceled)
	}
}

func FuzzWhitespaceTokenizer(f *testing.F) {
	for _, seed := range []string{
		"", "hello world", "\talpha\n beta\r\n", "中文\u00a0空格", string([]byte{'a', 0xff, ' ', 'b'}),
	} {
		f.Add(seed)
	}
	tokenizer := NewWhitespaceTokenizer()
	f.Fuzz(func(t *testing.T, input string) {
		tokens, err := tokenizer.Tokenize(context.Background(), input)
		require.NoError(t, err)

		cursor := 0
		for position, token := range tokens {
			require.Equal(t, uint32(position), token.Position)
			require.False(t, token.Text == "")

			start := int(token.Offset)
			end := start + len(token.Text)
			require.True(t, start >= cursor)
			require.True(t, end <= len(input))
			require.Equal(t, token.Text, input[start:end])

			for _, value := range []byte(token.Text) {
				require.False(t, asciiWhitespace(value))
			}
			for ; cursor < start; cursor++ {
				require.True(t, asciiWhitespace(input[cursor]))
			}
			cursor = end
		}
		for ; cursor < len(input); cursor++ {
			require.True(t, asciiWhitespace(input[cursor]))
		}
	})
}

func BenchmarkWhitespaceTokenizer(b *testing.B) {
	text := strings.Repeat("The quick\tbrown fox jumps over 中文 and C++.\n", 1024)
	tokenizer := NewWhitespaceTokenizer()
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
