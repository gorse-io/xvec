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
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStemmerTokenFilterOptions(t *testing.T) {
	defaults := DefaultStemmerTokenFilterOptions()
	require.True(t, defaults.Language == "english")
	require.NoError(t, defaults.Validate())

	for _, options := range []StemmerTokenFilterOptions{{}, {Language: "english"}, {Language: "en"}, {Language: "porter"}} {
		filter, err := NewStemmerTokenFilter(options)
		require.NoError(t, err)
		require.True(t, filter.Name() == "stemmer")

		wantLanguage := options.Language
		if wantLanguage == "" {
			wantLanguage = "english"
		}
		require.Equal(t, wantLanguage, filter.Language())
	}
	for _, language := range []string{"English", "EN", "nonexistent_lang", " english", "english "} {
		options := StemmerTokenFilterOptions{Language: language}
		{
			err := options.Validate()
			require.ErrorIs(t, err, ErrInvalidStemmerOptions)
		}
		{
			filter, err := NewStemmerTokenFilter(options)
			require.Nil(t, filter)
			require.ErrorIs(t, err, ErrInvalidStemmerOptions)
		}
	}
}

func TestSupportedStemmerLanguages(t *testing.T) {
	languages := SupportedStemmerLanguages()
	require.Len(t, languages, 115)

	hash := sha256.Sum256([]byte(strings.Join(languages, "\n") + "\n"))
	{
		got := fmt.Sprintf("%x", hash)
		require.True(t, got == "f682fb56f2f7c4a6b7952967057c07c0d63a4ac0f57fcdbb414c59a6acefe7b1")
	}

	for index := 1; index < len(languages); index++ {
		require.True(t, languages[index-1] < languages[index])
	}
	languages[0] = "modified"
	require.True(t, SupportedStemmerLanguages()[0] == "ar",
		"caller mutated language registry")
}

func TestStemmerTokenFilterBaselineBehavior(t *testing.T) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	require.NoError(t, err)

	input := []Token{
		{Text: "", Offset: 1, Position: 2},
		{Text: "running", Offset: 3, Position: 4},
		{Text: "cats", Offset: 11, Position: 5},
		{Text: "easily", Offset: 16, Position: 6},
		{Text: "connection", Offset: 23, Position: 7},
	}
	got, err := filter.Filter(context.Background(), input)
	require.NoError(t, err)

	want := []Token{
		{Text: "", Offset: 1, Position: 2},
		{Text: "run", Offset: 3, Position: 4},
		{Text: "cat", Offset: 11, Position: 5},
		{Text: "easili", Offset: 16, Position: 6},
		{Text: "connect", Offset: 23, Position: 7},
	}
	require.Equal(t, want, got)

	got[1].Offset = 99
	require.Equal(t, Token{Text: "running", Offset: 3, Position: 4}, input[1])

	for _, empty := range [][]Token{nil, {}} {
		result, err := filter.Filter(context.Background(), empty)
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result, 0)
	}
}

func TestStemmerTokenFilterLowercaseChain(t *testing.T) {
	lowercase := NewLowercaseTokenFilter()
	stemmer, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	require.NoError(t, err)

	tokens, err := lowercase.Filter(context.Background(), []Token{{Text: "Running"}, {Text: "Cats"}, {Text: "EASILY"}})
	require.NoError(t, err)

	tokens, err = stemmer.Filter(context.Background(), tokens)
	require.NoError(t, err)
	{
		want := []Token{{Text: "run"}, {Text: "cat"}, {Text: "easili"}}
		require.Equal(t, want, tokens)
	}
}

func TestStemmerTokenFilterContextCancellation(t *testing.T) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	require.NoError(t, err)
	{
		_, err := filter.Filter(nil, nil)
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := filter.Filter(canceled, []Token{{Text: "running"}})
		require.ErrorIs(t, err, context.Canceled)
	}

	midway := newCancelAfterChecks(4)
	{
		_, err := filter.Filter(midway, []Token{{Text: strings.Repeat("running", 16<<10)}})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestStemmerTokenFilterConcurrentUse(t *testing.T) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{Language: "english"})
	require.NoError(t, err)

	input := []Token{{Text: "running", Offset: 7, Position: 9}, {Text: "connections", Offset: 15, Position: 10}}
	want, err := filter.Filter(context.Background(), input)
	require.NoError(t, err)

	var wait sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for worker := 0; worker < 32; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				got, err := filter.Filter(context.Background(), input)
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

func FuzzStemmerTokenFilter(f *testing.F) {
	for _, seed := range []string{"", "running", "easily", "connection", "τρέχοντας", string([]byte{0xc0, 0x80, 'r', 'u', 'n', 'n', 'i', 'n', 'g'}), string([]byte{0xed, 0xa0, 0x80})} {
		f.Add(seed)
	}
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	require.NoError(f, err)

	f.Fuzz(func(t *testing.T, input string) {
		tokens := []Token{{Text: input, Offset: 123, Position: 456}}
		got, err := filter.Filter(context.Background(), tokens)
		require.NoError(t, err)
		require.Len(t, got, 1)
		require.True(t, got[0].Offset == 123)
		require.True(t, got[0].Position == 456)
		require.Equal(t, input, tokens[0].Text)
	})
}

func BenchmarkStemmerTokenFilter(b *testing.B) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	if err != nil {
		require.NoError(b, err)
	}

	tokens := make([]Token, 1024)
	for index := range tokens {
		tokens[index] = Token{Text: "running", Offset: uint32(index * 8), Position: uint32(index)}
	}
	b.SetBytes(int64(len(tokens) * len("running")))
	b.ReportAllocs()
	for b.Loop() {
		{
			_, err := filter.Filter(context.Background(), tokens)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}
