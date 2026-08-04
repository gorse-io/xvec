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
	"reflect"
	"strings"
	"sync"
	"testing"
)

type asciiFoldingFixture struct {
	BaselineCommit        string `json:"baseline_commit"`
	SourceSHA256          string `json:"source_sha256"`
	HeaderSHA256          string `json:"header_sha256"`
	UTF8ProcCommit        string `json:"utf8proc_commit"`
	UTF8ProcDataSHA256    string `json:"utf8proc_data_sha256"`
	NFKDPairsSHA256       string `json:"nfkd_pairs_sha256"`
	ExtraFoldsSHA256      string `json:"extra_folds_sha256"`
	EffectivePairsSHA256  string `json:"effective_pairs_sha256"`
	EffectiveMappingCount int    `json:"effective_mapping_count"`
	Cases                 []struct {
		Name      string `json:"name"`
		InputHex  string `json:"input_hex"`
		OutputHex string `json:"output_hex"`
		Removed   bool   `json:"removed"`
	} `json:"cases"`
}

func TestASCIIFoldingTokenFilterBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/ascii_folding_filter_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture asciiFoldingFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.SourceSHA256 != "1b0962633ddcd1d703d7d86cd5257566b4ebafee8a8c546ba0b4eed4cb151307" ||
		fixture.HeaderSHA256 != "d1b9d6cd964d2d7bceb538424ce022c3e676784ff47c02400942e00367fc9454" ||
		fixture.UTF8ProcCommit != "e5e799221b45bbb90f5fdc5c69b6b8dfbf017e78" ||
		fixture.UTF8ProcDataSHA256 != "950e549dbfc853c4304425f3af1875e72fa9fc9697c273c763400c2da4e380a7" ||
		fixture.NFKDPairsSHA256 != "049564e35ef35becd34f82abc578733fc2cacfcaa84960659c741173cc7f5907" ||
		fixture.ExtraFoldsSHA256 != "b30b3076dea67303cc3e5d2fbdd16097c7deeba255200b63fd41b335346947fd" ||
		fixture.EffectivePairsSHA256 != "d4255c7dfe10844f0d464b8a82a3adc556f7bb42a5c0c39775e8b3516f3a4636" ||
		fixture.EffectiveMappingCount != 2120 {
		t.Fatalf("unexpected fixture identity: %#v", fixture)
	}
	filter := NewASCIIFoldingTokenFilter()
	if filter.Name() != "ascii_folding" {
		t.Fatalf("name = %q", filter.Name())
	}
	for index, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			input, err := hex.DecodeString(test.InputHex)
			if err != nil {
				t.Fatal(err)
			}
			want, err := hex.DecodeString(test.OutputHex)
			if err != nil {
				t.Fatal(err)
			}
			tokens := []Token{{Text: string(input), Offset: uint32(index + 7), Position: uint32(index + 11)}}
			got, err := filter.Filter(context.Background(), tokens)
			if err != nil {
				t.Fatal(err)
			}
			var expected []Token
			if test.Removed {
				expected = []Token{}
			} else {
				expected = []Token{{Text: string(want), Offset: tokens[0].Offset, Position: tokens[0].Position}}
			}
			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("Filter(%x) = %#v, want %#v", input, got, expected)
			}
			if tokens[0].Text != string(input) {
				t.Fatal("filter modified its input")
			}
		})
	}
}

func TestASCIIFoldingTablesIdentity(t *testing.T) {
	if len(asciiNFKDCodepoints) != 1986 || len(asciiNFKDReplacements) != len(asciiNFKDCodepoints) {
		t.Fatalf("NFKD table lengths = %d, %d", len(asciiNFKDCodepoints), len(asciiNFKDReplacements))
	}
	nfkdHash := sha256.New()
	for index, codepoint := range asciiNFKDCodepoints {
		if index > 0 && asciiNFKDCodepoints[index-1] >= codepoint || asciiNFKDReplacements[index] == 0 {
			t.Fatalf("invalid NFKD entry %d", index)
		}
		fmt.Fprintf(nfkdHash, "%X %X\n", codepoint, asciiNFKDReplacements[index])
	}
	if got := fmt.Sprintf("%x", nfkdHash.Sum(nil)); got != "049564e35ef35becd34f82abc578733fc2cacfcaa84960659c741173cc7f5907" {
		t.Fatalf("NFKD SHA-256 = %s", got)
	}

	extraHash := sha256.New()
	for index, fold := range asciiExtraFolds {
		if index > 0 && asciiExtraFolds[index-1].codepoint >= fold.codepoint || fold.replacement == "" {
			t.Fatalf("invalid extra fold %d", index)
		}
		fmt.Fprintf(extraHash, "\t{0x%04X, %q},\n", fold.codepoint, fold.replacement)
	}
	if got := fmt.Sprintf("%x", extraHash.Sum(nil)); got != "b30b3076dea67303cc3e5d2fbdd16097c7deeba255200b63fd41b335346947fd" {
		t.Fatalf("extra-fold SHA-256 = %s", got)
	}

	effectiveHash := sha256.New()
	count := 0
	for value := rune(0x80); value <= 0x10ffff; value++ {
		if replacement, found := lookupASCIIExtraFold(value); found {
			fmt.Fprintf(effectiveHash, "%X %X\n", value, []byte(replacement))
			count++
		} else if replacement, found := lookupASCIINFKDFold(value); found {
			fmt.Fprintf(effectiveHash, "%X %X\n", value, replacement)
			count++
		}
	}
	if count != 2120 {
		t.Fatalf("effective mapping count = %d", count)
	}
	if got := fmt.Sprintf("%x", effectiveHash.Sum(nil)); got != "d4255c7dfe10844f0d464b8a82a3adc556f7bb42a5c0c39775e8b3516f3a4636" {
		t.Fatalf("effective SHA-256 = %s", got)
	}
}

func TestASCIIFoldingTokenFilterRemovalAndOwnership(t *testing.T) {
	filter := NewASCIIFoldingTokenFilter()
	input := []Token{
		{Text: "", Offset: 1, Position: 2},
		{Text: "café", Offset: 3, Position: 4},
		{Text: "", Offset: 5, Position: 6},
		{Text: "中文", Offset: 7, Position: 8},
	}
	got, err := filter.Filter(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	want := []Token{{Text: "cafe", Offset: 3, Position: 4}, {Text: "中文", Offset: 7, Position: 8}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %#v, want %#v", got, want)
	}
	got[0].Offset = 99
	if input[1] != (Token{Text: "café", Offset: 3, Position: 4}) {
		t.Fatalf("input changed through result alias: %#v", input)
	}
	for _, empty := range [][]Token{nil, {}} {
		result, err := filter.Filter(context.Background(), empty)
		if err != nil || result == nil || len(result) != 0 {
			t.Fatalf("Filter(%#v) = %#v, %v", empty, result, err)
		}
	}
}

func TestASCIIFoldingTokenFilterContextCancellation(t *testing.T) {
	filter := NewASCIIFoldingTokenFilter()
	if _, err := filter.Filter(nil, nil); err == nil {
		t.Fatal("nil context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := filter.Filter(canceled, []Token{{Text: "café"}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled error = %v", err)
	}
	midway := newCancelAfterChecks(4)
	if _, err := filter.Filter(midway, []Token{{Text: strings.Repeat("A", 64<<10)}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-filter error = %v", err)
	}
}

func TestASCIIFoldingTokenFilterConcurrentUse(t *testing.T) {
	filter := NewASCIIFoldingTokenFilter()
	input := []Token{{Text: "café Æß ﬁ Ａ０ 中文 ←→", Offset: 7, Position: 9}}
	want, err := filter.Filter(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
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
				if !reflect.DeepEqual(got, want) {
					errorsChannel <- errors.New("concurrent result differs")
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		t.Fatal(err)
	}
}

func FuzzASCIIFoldingTokenFilter(f *testing.F) {
	for _, seed := range []string{"", "ASCII", "café Æß", "ﬁﬂＡ０", "中文ά café", "←→", string([]byte{0xc0, 0x80, 'A'})} {
		f.Add(seed)
	}
	filter := NewASCIIFoldingTokenFilter()
	f.Fuzz(func(t *testing.T, input string) {
		tokens := []Token{{Text: input, Offset: 123, Position: 456}}
		got, err := filter.Filter(context.Background(), tokens)
		if err != nil {
			t.Fatal(err)
		}
		if tokens[0].Text != input {
			t.Fatal("input changed")
		}
		if input == "" {
			if len(got) != 0 {
				t.Fatalf("empty token survived: %#v", got)
			}
			return
		}
		if len(got) != 1 || got[0].Offset != 123 || got[0].Position != 456 || got[0].Text == "" {
			t.Fatalf("metadata or output invalid: %#v", got)
		}
		twice, err := filter.Filter(context.Background(), got)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(twice, got) {
			t.Fatalf("filter is not idempotent: %x -> %x -> %x", input, got[0].Text, twice[0].Text)
		}
	})
}

func BenchmarkASCIIFoldingTokenFilter(b *testing.B) {
	filter := NewASCIIFoldingTokenFilter()
	tokens := []Token{{Text: strings.Repeat("café Æß ﬁﬂ Ａ０ 中文 ←→ ", 1024), Offset: 7, Position: 9}}
	b.SetBytes(int64(len(tokens[0].Text)))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := filter.Filter(context.Background(), tokens); err != nil {
			b.Fatal(err)
		}
	}
}
