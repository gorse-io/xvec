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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type jiebaTokenizerFixture struct {
	BaselineCommit       string `json:"baseline_commit"`
	SourceSHA256         string `json:"source_sha256"`
	CppJiebaCommit       string `json:"cppjieba_commit"`
	DictionarySHA256     string `json:"dictionary_sha256"`
	ModelSHA256          string `json:"model_sha256"`
	UserDictionarySHA256 string `json:"user_dictionary_sha256"`
	Cases                []struct {
		Name           string       `json:"name"`
		Mode           JiebaCutMode `json:"mode"`
		UserDictionary bool         `json:"user_dictionary"`
		InputHex       string       `json:"input_hex"`
		Tokens         []struct {
			TextHex  string `json:"text_hex"`
			Offset   uint32 `json:"offset"`
			Position uint32 `json:"position"`
		} `json:"tokens"`
	} `json:"cases"`
}

func TestJiebaTokenizerBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/jieba_tokenizer_58375ff.json")
	require.NoError(t, err)

	var fixture jiebaTokenizerFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.SourceSHA256 == "b5092a63528a3bd4faf6e31c156217d5dddccb678cb829569b87cb63e6218edd")
	require.True(t, fixture.CppJiebaCommit == "b3602bef7d1f67521a61788a74fb5801a0e62cd3")
	require.True(t, fixture.DictionarySHA256 == "7c66ea73c84bc8699905422312fbe11045879a270a5b8e0a93cf97e48efed412")
	require.True(t, fixture.ModelSHA256 == "3479bc295e7dd885bcb04098a63731876519fc1c92dbedc1928ec47e1a8bcfc0")
	require.True(t, fixture.UserDictionarySHA256 == "bde317009e581fd0b615e7e8c6a4aad7bcd4be86ee8350f83f87ca75fddc1542")

	for path, want := range map[string]string{
		filepath.Join(jiebaTestDictDir(), jiebaDictionaryFile): "7c66ea73c84bc8699905422312fbe11045879a270a5b8e0a93cf97e48efed412",
		filepath.Join(jiebaTestDictDir(), jiebaHMMModelFile):   "3479bc295e7dd885bcb04098a63731876519fc1c92dbedc1928ec47e1a8bcfc0",
		filepath.Join(jiebaTestDictDir(), "user.dict.utf8"):    "bde317009e581fd0b615e7e8c6a4aad7bcd4be86ee8350f83f87ca75fddc1542",
	} {
		data, err := os.ReadFile(path)
		require.NoError(t, err)
		{
			got := fmt.Sprintf("%x", sha256.Sum256(data))
			require.Equal(t, want, got)
		}
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			options := JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: test.Mode}
			if test.UserDictionary {
				options.UserDictPath = filepath.Join(jiebaTestDictDir(), "user.dict.utf8")
			}
			tokenizer := mustJiebaTokenizer(t, options)
			input, err := hex.DecodeString(test.InputHex)
			require.NoError(t, err)

			got, err := tokenizer.Tokenize(context.Background(), string(input))
			require.NoError(t, err)

			want := make([]Token, len(test.Tokens))
			for index, token := range test.Tokens {
				text, err := hex.DecodeString(token.TextHex)
				require.NoError(t, err)

				want[index] = Token{Text: string(text), Offset: token.Offset, Position: token.Position}
			}
			require.Equal(t, want, got)

			assertJiebaTokenRanges(t, string(input), got)
		})
	}
}

func TestJiebaTokenizerOptionsAndResolution(t *testing.T) {
	savedDefault := DefaultJiebaDictDir()
	t.Cleanup(func() { SetDefaultJiebaDictDir(savedDefault) })
	t.Setenv("ZVEC_JIEBA_DICT_DIR", "")
	SetDefaultJiebaDictDir("")

	defaults := DefaultJiebaTokenizerOptions()
	require.Equal(t, JiebaCutModeSearch, defaults.CutMode)
	{
		_, err := NewJiebaTokenizer(context.Background(), defaults)
		require.ErrorIs(t, err, ErrInvalidJiebaTokenizerOptions)
	}

	for _, mode := range []JiebaCutMode{JiebaCutModeSearch, JiebaCutModeMix, JiebaCutModeFull, JiebaCutModeHMM} {
		tokenizer := mustJiebaTokenizer(t, JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: mode})
		require.True(t, tokenizer.Name() == "jieba")
		require.Equal(t, mode, tokenizer.CutMode())
		require.Equal(t, jiebaTestDictDir(), tokenizer.DictDir())
	}
	{
		_, err := NewJiebaTokenizer(context.Background(), JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: "unknown"})
		require.ErrorIs(t, err, ErrInvalidJiebaTokenizerOptions)
	}

	SetDefaultJiebaDictDir(jiebaTestDictDir())
	tokenizer := mustJiebaTokenizer(t, DefaultJiebaTokenizerOptions())
	require.Equal(t, jiebaTestDictDir(), tokenizer.DictDir())

	SetDefaultJiebaDictDir("missing-global")
	t.Setenv("ZVEC_JIEBA_DICT_DIR", jiebaTestDictDir())
	tokenizer = mustJiebaTokenizer(t, DefaultJiebaTokenizerOptions())
	require.Equal(t, jiebaTestDictDir(), tokenizer.DictDir())

	t.Setenv("ZVEC_JIEBA_DICT_DIR", "missing-environment")
	tokenizer = mustJiebaTokenizer(t, JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: JiebaCutModeSearch})
	require.Equal(t, jiebaTestDictDir(), tokenizer.DictDir())
}

func TestJiebaTokenizerModeResourceRequirements(t *testing.T) {
	dictionaryOnly := t.TempDir()
	copyJiebaTestFile(t, filepath.Join(jiebaTestDictDir(), jiebaDictionaryFile), filepath.Join(dictionaryOnly, jiebaDictionaryFile))
	mustJiebaTokenizer(t, JiebaTokenizerOptions{DictDir: dictionaryOnly, CutMode: JiebaCutModeFull})
	for _, mode := range []JiebaCutMode{JiebaCutModeSearch, JiebaCutModeMix, JiebaCutModeHMM} {
		_, err := NewJiebaTokenizer(context.Background(), JiebaTokenizerOptions{DictDir: dictionaryOnly, CutMode: mode})
		require.ErrorIs(t, err, ErrInvalidJiebaTokenizerOptions)
	}

	modelOnly := t.TempDir()
	copyJiebaTestFile(t, filepath.Join(jiebaTestDictDir(), jiebaHMMModelFile), filepath.Join(modelOnly, jiebaHMMModelFile))
	mustJiebaTokenizer(t, JiebaTokenizerOptions{DictDir: modelOnly, CutMode: JiebaCutModeHMM})
	for _, mode := range []JiebaCutMode{JiebaCutModeSearch, JiebaCutModeMix, JiebaCutModeFull} {
		_, err := NewJiebaTokenizer(context.Background(), JiebaTokenizerOptions{DictDir: modelOnly, CutMode: mode})
		require.ErrorIs(t, err, ErrInvalidJiebaTokenizerOptions)
	}
}

func TestJiebaTokenizerUserDictionaryIsolation(t *testing.T) {
	withUser := mustJiebaTokenizer(t, JiebaTokenizerOptions{
		DictDir: jiebaTestDictDir(), UserDictPath: filepath.Join(jiebaTestDictDir(), "user.dict.utf8"), CutMode: JiebaCutModeMix,
	})
	withoutUser := mustJiebaTokenizer(t, JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: JiebaCutModeMix})
	withTokens, err := withUser.Tokenize(context.Background(), "甲乙")
	require.NoError(t, err)

	withoutTokens, err := withoutUser.Tokenize(context.Background(), "甲乙")
	require.NoError(t, err)
	{
		want := []string{"甲", "乙"}
		require.Equal(t, want, jiebaTokenTexts(withTokens))
	}
	{
		want := []string{"甲乙"}
		require.Equal(t, want, jiebaTokenTexts(withoutTokens))
	}
}

func TestJiebaTokenizerPinnedDecoderAndASCII(t *testing.T) {
	tokenizer := mustJiebaTokenizer(t, JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: JiebaCutModeHMM})
	tests := []struct {
		input string
		want  []string
	}{
		{"abc1.2 3.14 A9!", []string{"abc1.2", " ", "3.14", " ", "A9", "!"}},
		{string([]byte{0xc0, 0x80}), []string{string([]byte{0xc0, 0x80})}},
		{string([]byte{0x80, 'A'}), []string{string([]byte{0x80, 'A'})}},
	}
	for _, test := range tests {
		tokens, err := tokenizer.Tokenize(context.Background(), test.input)
		require.NoError(t, err)
		{
			texts := jiebaTokenTexts(tokens)
			require.Equal(t, test.want, texts)
		}
	}
	for _, input := range []string{string([]byte{'a', 0xc2}), string([]byte{0xf8, 'a'})} {
		{
			_, err := tokenizer.Tokenize(context.Background(), input)
			require.ErrorIs(t, err, ErrInvalidJiebaUTF8)
		}
	}
	empty, err := tokenizer.Tokenize(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
}

func TestJiebaTokenizerInvalidResources(t *testing.T) {
	badDictionary := t.TempDir()
	{
		err := os.WriteFile(filepath.Join(badDictionary, jiebaDictionaryFile), []byte("bad line\n"), 0o600)
		require.NoError(t, err)
	}

	_, err := NewJiebaTokenizer(context.Background(), JiebaTokenizerOptions{DictDir: badDictionary, CutMode: JiebaCutModeFull})
	require.ErrorIs(t, err, ErrInvalidJiebaTokenizerOptions)

	badModel := t.TempDir()
	{
		err := os.WriteFile(filepath.Join(badModel, jiebaHMMModelFile), []byte("0 0\n"), 0o600)
		require.NoError(t, err)
	}

	_, err = NewJiebaTokenizer(context.Background(), JiebaTokenizerOptions{DictDir: badModel, CutMode: JiebaCutModeHMM})
	require.ErrorIs(t, err, ErrInvalidJiebaTokenizerOptions)

	badUser := filepath.Join(t.TempDir(), "user.dict")
	{
		err := os.WriteFile(badUser, []byte("word not-a-frequency tag\n"), 0o600)
		require.NoError(t, err)
	}

	_, err = NewJiebaTokenizer(context.Background(), JiebaTokenizerOptions{
		DictDir: jiebaTestDictDir(), UserDictPath: badUser, CutMode: JiebaCutModeFull,
	})
	require.ErrorIs(t, err, ErrInvalidJiebaTokenizerOptions)
}

func TestJiebaTokenizerContextCancellation(t *testing.T) {
	options := JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: JiebaCutModeSearch}
	{
		_, err := NewJiebaTokenizer(nil, options)
		require.Error(t, err,
			"nil construction context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := NewJiebaTokenizer(canceled, options)
		require.ErrorIs(t, err, context.Canceled)
	}

	tokenizer := mustJiebaTokenizer(t, options)
	{
		_, err := tokenizer.Tokenize(nil, "中文")
		require.Error(t, err,
			"nil tokenization context succeeded")
	}
	{
		_, err := tokenizer.Tokenize(canceled, "中文")
		require.ErrorIs(t, err, context.Canceled)
	}

	largeDir := t.TempDir()
	var dictionary strings.Builder
	for index := 0; index < 5000; index++ {
		fmt.Fprintf(&dictionary, "词%d %d n\n", index, index+1)
	}
	{
		err := os.WriteFile(filepath.Join(largeDir, jiebaDictionaryFile), []byte(dictionary.String()), 0o600)
		require.NoError(t, err)
	}

	midLoad := newCancelAfterChecks(3)
	_, err := NewJiebaTokenizer(midLoad, JiebaTokenizerOptions{DictDir: largeDir, CutMode: JiebaCutModeFull})
	require.ErrorIs(t, err, context.Canceled)

	midTokenize := newCancelAfterChecks(4)
	_, err = tokenizer.Tokenize(midTokenize, strings.Repeat("中华人民共和国自然语言处理", 5000))
	require.ErrorIs(t, err, context.Canceled)
}

func TestJiebaTokenizerConcurrentUse(t *testing.T) {
	tokenizer := mustJiebaTokenizer(t, JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: JiebaCutModeSearch})
	want, err := tokenizer.Tokenize(context.Background(), "中华人民共和国成立 abc1.2")
	require.NoError(t, err)

	var wait sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for worker := 0; worker < 32; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 20; iteration++ {
				got, err := tokenizer.Tokenize(context.Background(), "中华人民共和国成立 abc1.2")
				if err != nil {
					errorsChannel <- err
					return
				}
				if !assert.Equal(t, want, got) {
					errorsChannel <- errors.New("concurrent result differs")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		require.NoError(t, err)
	}
}

func FuzzJiebaTokenizer(f *testing.F) {
	for _, seed := range []string{"", "中华人民共和国", "abc1.2 中文", "杭研", string([]byte{0xc0, 0x80}), string([]byte{'a', 0xf8})} {
		f.Add(seed)
	}
	tokenizer := mustJiebaTokenizer(f, JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: JiebaCutModeSearch})
	f.Fuzz(func(t *testing.T, input string) {
		tokens, err := tokenizer.Tokenize(context.Background(), input)
		if err != nil {
			require.ErrorIs(t, err, ErrInvalidJiebaUTF8)

			return
		}
		assertJiebaTokenRanges(t, input, tokens)
	})
}

func BenchmarkJiebaTokenizer(b *testing.B) {
	tokenizer := mustJiebaTokenizer(b, JiebaTokenizerOptions{DictDir: jiebaTestDictDir(), CutMode: JiebaCutModeSearch})
	text := strings.Repeat("中华人民共和国自然语言处理技术 abc1.2。", 1024)
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

func jiebaTestDictDir() string { return filepath.Join("testdata", "jieba") }

func mustJiebaTokenizer(tb testing.TB, options JiebaTokenizerOptions) *JiebaTokenizer {
	tb.Helper()
	tokenizer, err := NewJiebaTokenizer(context.Background(), options)
	require.NoError(tb, err)

	return tokenizer
}

func copyJiebaTestFile(t testing.TB, source, target string) {
	t.Helper()
	data, err := os.ReadFile(source)
	require.NoError(t, err)
	{
		err := os.WriteFile(target, data, 0o600)
		require.NoError(t, err)
	}
}

func jiebaTokenTexts(tokens []Token) []string {
	texts := make([]string, len(tokens))
	for index := range tokens {
		texts[index] = tokens[index].Text
	}
	return texts
}

func assertJiebaTokenRanges(t testing.TB, input string, tokens []Token) {
	t.Helper()
	for position, token := range tokens {
		start := int(token.Offset)
		end := start + len(token.Text)
		require.Equal(t, uint32(position), token.Position)
		require.False(t, token.Text == "")
		require.True(t, start >= 0)
		require.True(t, end <= len(input))
		require.Equal(t, token.Text, input[start:end])
	}
}
