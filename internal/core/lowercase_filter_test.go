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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type lowercaseFilterFixture struct {
	BaselineCommit     string `json:"baseline_commit"`
	SourceSHA256       string `json:"source_sha256"`
	UTF8ProcCommit     string `json:"utf8proc_commit"`
	UTF8ProcDataSHA256 string `json:"utf8proc_data_sha256"`
	MappingPairsSHA256 string `json:"mapping_pairs_sha256"`
	MappingCount       int    `json:"mapping_count"`
	Cases              []struct {
		Name      string `json:"name"`
		InputHex  string `json:"input_hex"`
		OutputHex string `json:"output_hex"`
	} `json:"cases"`
}

func TestLowercaseTokenFilterBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/lowercase_filter_58375ff.json")
	require.NoError(t, err)

	var fixture lowercaseFilterFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	require.True(t, fixture.SourceSHA256 == "52a9c44a460818cf4d4857f980eea8bad944466ca8ca93e754107d4aa47ea7eb")
	require.True(t, fixture.UTF8ProcCommit == "e5e799221b45bbb90f5fdc5c69b6b8dfbf017e78")
	require.True(t, fixture.UTF8ProcDataSHA256 == "950e549dbfc853c4304425f3af1875e72fa9fc9697c273c763400c2da4e380a7")
	require.True(t, fixture.MappingPairsSHA256 == "3687b35be3fc408c2a4044074d2e673f7cc4818d15e3911ba161aabf252d6068")
	require.True(t, fixture.MappingCount == 1488)

	filter := NewLowercaseTokenFilter()
	require.True(t, filter.Name() == "lowercase")

	for index, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			input, err := hex.DecodeString(test.InputHex)
			require.NoError(t, err)

			want, err := hex.DecodeString(test.OutputHex)
			require.NoError(t, err)

			tokens := []Token{{Text: string(input), Offset: uint32(index + 7), Position: uint32(index + 11)}}
			got, err := filter.Filter(context.Background(), tokens)
			require.NoError(t, err)
			{
				expected := []Token{{Text: string(want), Offset: tokens[0].Offset, Position: tokens[0].Position}}
				require.Equal(t, expected, got)
			}
			require.Equal(t, string(input), tokens[0].Text,
				"filter modified its input")
		})
	}
}

func TestLowercaseUnicode17TableIdentity(t *testing.T) {
	hash := sha256.New()
	count := 0
	for value := rune(0); value <= 0x10ffff; value++ {
		lower := lowercaseUnicode17(value)
		if lower != value {
			_, _ = fmt.Fprintf(hash, "%X %X\n", value, lower)
			count++
		}
	}
	require.True(t, count == 1488)
	{
		got := fmt.Sprintf("%x", hash.Sum(nil))
		require.True(t, got == "3687b35be3fc408c2a4044074d2e673f7cc4818d15e3911ba161aabf252d6068")
	}

	for index, mapping := range lowercaseUnicode17Ranges {
		require.True(t, mapping.first <= mapping.last)
		require.True(t, mapping.step >= 1)
		require.False(t, index > 0 && lowercaseUnicode17Ranges[index-1].last >= mapping.first)
	}
}

func TestLowercaseTokenFilterEmptyAndOwnership(t *testing.T) {
	filter := NewLowercaseTokenFilter()
	for _, input := range [][]Token{nil, {}} {
		got, err := filter.Filter(context.Background(), input)
		require.NoError(t, err)
		require.NotNil(t, got)
		require.Len(t, got, 0)
	}
	input := []Token{{Text: "ABC", Offset: 5, Position: 3}}
	got, err := filter.Filter(context.Background(), input)
	require.NoError(t, err)

	got[0].Offset = 99
	require.Equal(t, Token{Text: "ABC", Offset: 5, Position: 3}, input[0])
}

func TestLowercaseTokenFilterContextCancellation(t *testing.T) {
	filter := NewLowercaseTokenFilter()
	{
		_, err := filter.Filter(nil, nil)
		require.Error(t, err,
			"nil context succeeded")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := filter.Filter(canceled, []Token{{Text: "ABC"}})
		require.ErrorIs(t, err, context.Canceled)
	}

	midway := newCancelAfterChecks(4)
	{
		_, err := filter.Filter(midway, []Token{{Text: strings.Repeat("A", 64<<10)}})
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestLowercaseTokenFilterConcurrentUse(t *testing.T) {
	filter := NewLowercaseTokenFilter()
	input := []Token{{Text: "ÜBER МОСКВА ΔΕΛΤΑ \U00010D50", Offset: 7, Position: 9}}
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

func FuzzLowercaseTokenFilter(f *testing.F) {
	for _, seed := range []string{"", "Hello", "ÜBER МОСКВА", "İKẞ", "\U00010D50", string([]byte{0xc0, 0x80, 'A'})} {
		f.Add(seed)
	}
	filter := NewLowercaseTokenFilter()
	f.Fuzz(func(t *testing.T, input string) {
		tokens := []Token{{Text: input, Offset: 123, Position: 456}}
		got, err := filter.Filter(context.Background(), tokens)
		require.NoError(t, err)
		require.Len(t, got, 1)
		require.True(t, got[0].Offset == 123)
		require.True(t, got[0].Position == 456)
		require.Equal(t, input, tokens[0].Text)

		twice, err := filter.Filter(context.Background(), got)
		require.NoError(t, err)
		require.Equal(t, got, twice)
	})
}

func BenchmarkLowercaseTokenFilter(b *testing.B) {
	filter := NewLowercaseTokenFilter()
	tokens := []Token{{Text: strings.Repeat("Hello ÜBER МОСКВА ΔΕΛΤΑ \U00010D50 中文 ", 1024), Offset: 7, Position: 9}}
	b.SetBytes(int64(len(tokens[0].Text)))
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
