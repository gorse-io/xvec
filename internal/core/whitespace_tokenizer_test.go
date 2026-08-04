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
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
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
	if err != nil {
		t.Fatal(err)
	}
	var fixture whitespaceTokenizerFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" {
		t.Fatalf("fixture baseline = %q", fixture.BaselineCommit)
	}
	tokenizer := NewWhitespaceTokenizer()
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			input, err := hex.DecodeString(test.InputHex)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]Token, len(test.Tokens))
			for index, token := range test.Tokens {
				text, err := hex.DecodeString(token.TextHex)
				if err != nil {
					t.Fatal(err)
				}
				want[index] = Token{Text: string(text), Offset: token.Offset, Position: token.Position}
			}
			got, err := tokenizer.Tokenize(context.Background(), string(input))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("tokens = %#v, want %#v", got, want)
			}
		})
	}
}

func TestWhitespaceTokenizerPinnedByteSemantics(t *testing.T) {
	tokenizer := NewWhitespaceTokenizer()
	if tokenizer.Name() != "whitespace" {
		t.Fatalf("name = %q", tokenizer.Name())
	}
	input := " \talpha\nBETA\v中文\fC++\r\nlast "
	want := []Token{
		{Text: "alpha", Offset: 2, Position: 0},
		{Text: "BETA", Offset: 8, Position: 1},
		{Text: "中文", Offset: 13, Position: 2},
		{Text: "C++", Offset: 20, Position: 3},
		{Text: "last", Offset: 25, Position: 4},
	}
	got, err := tokenizer.Tokenize(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %#v, want %#v", got, want)
	}

	for _, input := range []string{"", " ", "\t\n\v\f\r"} {
		got, err := tokenizer.Tokenize(context.Background(), input)
		if err != nil {
			t.Fatal(err)
		}
		if got == nil || len(got) != 0 {
			t.Fatalf("whitespace-only %q = %#v", input, got)
		}
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
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %#v, want %#v", got, want)
	}
}

func TestWhitespaceTokenizerContextCancellation(t *testing.T) {
	tokenizer := NewWhitespaceTokenizer()
	if _, err := tokenizer.Tokenize(nil, "text"); err == nil {
		t.Fatal("nil context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tokenizer.Tokenize(canceled, "text"); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled error = %v", err)
	}
	midway := newCancelAfterChecks(3)
	if _, err := tokenizer.Tokenize(midway, strings.Repeat("x", 32<<10)); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-token cancellation error = %v", err)
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
		if err != nil {
			t.Fatal(err)
		}
		cursor := 0
		for position, token := range tokens {
			if token.Position != uint32(position) || token.Text == "" {
				t.Fatalf("token %d = %#v", position, token)
			}
			start := int(token.Offset)
			end := start + len(token.Text)
			if start < cursor || end > len(input) || input[start:end] != token.Text {
				t.Fatalf("token %d has invalid source range: %#v", position, token)
			}
			for _, value := range []byte(token.Text) {
				if asciiWhitespace(value) {
					t.Fatalf("token %d contains ASCII whitespace: %#v", position, token)
				}
			}
			for ; cursor < start; cursor++ {
				if !asciiWhitespace(input[cursor]) {
					t.Fatalf("un-tokenized non-whitespace byte at %d", cursor)
				}
			}
			cursor = end
		}
		for ; cursor < len(input); cursor++ {
			if !asciiWhitespace(input[cursor]) {
				t.Fatalf("trailing non-whitespace byte at %d", cursor)
			}
		}
	})
}

func BenchmarkWhitespaceTokenizer(b *testing.B) {
	text := strings.Repeat("The quick\tbrown fox jumps over 中文 and C++.\n", 1024)
	tokenizer := NewWhitespaceTokenizer()
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := tokenizer.Tokenize(context.Background(), text); err != nil {
			b.Fatal(err)
		}
	}
}
