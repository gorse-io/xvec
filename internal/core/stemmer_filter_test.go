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

type stemmerFilterFixture struct {
	BaselineCommit   string `json:"baseline_commit"`
	SourceSHA256     string `json:"source_sha256"`
	HeaderSHA256     string `json:"header_sha256"`
	SnowballVersion  string `json:"snowball_version"`
	SnowballCommit   string `json:"snowball_commit"`
	ModulesSHA256    string `json:"modules_sha256"`
	AlgorithmsSHA256 string `json:"algorithms_sha256"`
	AliasesSHA256    string `json:"aliases_sha256"`
	AlgorithmCount   int    `json:"algorithm_count"`
	AliasCount       int    `json:"alias_count"`
	Cases            []struct {
		Language  string `json:"language"`
		InputHex  string `json:"input_hex"`
		OutputHex string `json:"output_hex"`
	} `json:"cases"`
}

func loadStemmerFilterFixture(t testing.TB) stemmerFilterFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/stemmer_filter_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture stemmerFilterFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func TestStemmerTokenFilterBaselineFixture(t *testing.T) {
	fixture := loadStemmerFilterFixture(t)
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.SourceSHA256 != "8958f54f93148d162b4c4a2efbe36c5e6e5a22bd3bb24ec15a7568397ac498ec" ||
		fixture.HeaderSHA256 != "db25ee73a8c6b92a5367edde11ce9667b398486e58b12a0e7719c73b186765e7" ||
		fixture.SnowballVersion != "3.1.1" ||
		fixture.SnowballCommit != "cd195b51e948a902a4312f023f4a14392516a543" ||
		fixture.ModulesSHA256 != "a4f1a2fde0231ca137b2926de4da1a5c4e532c5a6e51248699e6e97af8170ad7" ||
		fixture.AlgorithmsSHA256 != "282b20fb1b8b31743af035ac1e13c24ebf5f42954a46b09686044ca547c6ae1f" ||
		fixture.AliasesSHA256 != "f682fb56f2f7c4a6b7952967057c07c0d63a4ac0f57fcdbb414c59a6acefe7b1" ||
		fixture.AlgorithmCount != 36 || fixture.AliasCount != 115 {
		t.Fatalf("unexpected fixture identity: %#v", fixture)
	}
	for index, test := range fixture.Cases {
		t.Run(fmt.Sprintf("%s/%d", test.Language, index), func(t *testing.T) {
			input, err := hex.DecodeString(test.InputHex)
			if err != nil {
				t.Fatal(err)
			}
			want, err := hex.DecodeString(test.OutputHex)
			if err != nil {
				t.Fatal(err)
			}
			filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{Language: test.Language})
			if err != nil {
				t.Fatal(err)
			}
			tokens := []Token{{Text: string(input), Offset: uint32(index + 7), Position: uint32(index + 11)}}
			got, err := filter.Filter(context.Background(), tokens)
			if err != nil {
				t.Fatal(err)
			}
			expected := []Token{{Text: string(want), Offset: tokens[0].Offset, Position: tokens[0].Position}}
			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("Filter(%x) = %#v, want %#v", input, got, expected)
			}
			if tokens[0].Text != string(input) {
				t.Fatal("filter modified its input")
			}
		})
	}
}

func TestStemmerTokenFilterOptions(t *testing.T) {
	defaults := DefaultStemmerTokenFilterOptions()
	if defaults.Language != "english" || defaults.Validate() != nil {
		t.Fatalf("defaults = %#v", defaults)
	}
	for _, options := range []StemmerTokenFilterOptions{{}, {Language: "english"}, {Language: "en"}, {Language: "porter"}} {
		filter, err := NewStemmerTokenFilter(options)
		if err != nil {
			t.Fatalf("NewStemmerTokenFilter(%#v): %v", options, err)
		}
		if filter.Name() != "stemmer" {
			t.Fatalf("Name() = %q", filter.Name())
		}
		wantLanguage := options.Language
		if wantLanguage == "" {
			wantLanguage = "english"
		}
		if filter.Language() != wantLanguage {
			t.Fatalf("Language() = %q, want %q", filter.Language(), wantLanguage)
		}
	}
	for _, language := range []string{"English", "EN", "nonexistent_lang", " english", "english "} {
		options := StemmerTokenFilterOptions{Language: language}
		if err := options.Validate(); !errors.Is(err, ErrInvalidStemmerOptions) {
			t.Fatalf("Validate(%q) = %v", language, err)
		}
		if filter, err := NewStemmerTokenFilter(options); filter != nil || !errors.Is(err, ErrInvalidStemmerOptions) {
			t.Fatalf("NewStemmerTokenFilter(%q) = %#v, %v", language, filter, err)
		}
	}
}

func TestSupportedStemmerLanguages(t *testing.T) {
	languages := SupportedStemmerLanguages()
	if len(languages) != 115 {
		t.Fatalf("language count = %d", len(languages))
	}
	hash := sha256.Sum256([]byte(strings.Join(languages, "\n") + "\n"))
	if got := fmt.Sprintf("%x", hash); got != "f682fb56f2f7c4a6b7952967057c07c0d63a4ac0f57fcdbb414c59a6acefe7b1" {
		t.Fatalf("language SHA-256 = %s", got)
	}
	for index := 1; index < len(languages); index++ {
		if languages[index-1] >= languages[index] {
			t.Fatalf("languages are not strictly sorted at %d", index)
		}
	}
	languages[0] = "modified"
	if SupportedStemmerLanguages()[0] != "ar" {
		t.Fatal("caller mutated language registry")
	}
}

func TestStemmerTokenFilterAliases(t *testing.T) {
	fixture := loadStemmerFilterFixture(t)
	caseByLanguage := make(map[string]struct{ input, output string }, fixture.AlgorithmCount)
	for _, test := range fixture.Cases[:fixture.AlgorithmCount] {
		input, err := hex.DecodeString(test.InputHex)
		if err != nil {
			t.Fatal(err)
		}
		output, err := hex.DecodeString(test.OutputHex)
		if err != nil {
			t.Fatal(err)
		}
		caseByLanguage[test.Language] = struct{ input, output string }{string(input), string(output)}
	}
	aliases := map[string][]string{
		"arabic":       {"arabic", "ar", "ara"},
		"armenian":     {"armenian", "hy", "hye", "arm"},
		"basque":       {"basque", "eu", "eus", "baq"},
		"catalan":      {"catalan", "ca", "cat"},
		"czech":        {"czech", "cs", "ces", "cze"},
		"danish":       {"danish", "da", "dan"},
		"dutch":        {"dutch", "nl", "dut", "nld", "kraaij_pohlmann"},
		"english":      {"english", "en", "eng"},
		"esperanto":    {"esperanto", "eo", "epo"},
		"estonian":     {"estonian", "et", "est"},
		"finnish":      {"finnish", "fi", "fin"},
		"french":       {"french", "fr", "fre", "fra"},
		"german":       {"german", "de", "ger", "deu"},
		"greek":        {"greek", "el", "gre", "ell"},
		"hindi":        {"hindi", "hi", "hin"},
		"hungarian":    {"hungarian", "hu", "hun"},
		"indonesian":   {"indonesian", "id", "ind"},
		"irish":        {"irish", "ga", "gle"},
		"italian":      {"italian", "it", "ita"},
		"lithuanian":   {"lithuanian", "lt", "lit"},
		"nepali":       {"nepali", "ne", "nep"},
		"norwegian":    {"norwegian", "no", "nor"},
		"persian":      {"persian", "fa", "fas", "pers"},
		"polish":       {"polish", "pl", "pol"},
		"portuguese":   {"portuguese", "pt", "por"},
		"romanian":     {"romanian", "ro", "rum", "ron"},
		"russian":      {"russian", "ru", "rus"},
		"serbian":      {"serbian", "sr", "srp"},
		"sesotho":      {"sesotho", "st", "sot"},
		"spanish":      {"spanish", "es", "esl", "spa"},
		"swedish":      {"swedish", "sv", "swe"},
		"tamil":        {"tamil", "ta", "tam"},
		"turkish":      {"turkish", "tr", "tur"},
		"yiddish":      {"yiddish", "yi", "yid"},
		"porter":       {"porter"},
		"dutch_porter": {"dutch_porter"},
	}
	seen := make(map[string]struct{}, fixture.AliasCount)
	for canonical, names := range aliases {
		test, found := caseByLanguage[canonical]
		if !found {
			t.Fatalf("no fixture for %q", canonical)
		}
		for _, name := range names {
			seen[name] = struct{}{}
			filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{Language: name})
			if err != nil {
				t.Fatalf("alias %q: %v", name, err)
			}
			got, err := filter.Filter(context.Background(), []Token{{Text: test.input}})
			if err != nil {
				t.Fatalf("alias %q: %v", name, err)
			}
			if len(got) != 1 || got[0].Text != test.output {
				t.Fatalf("alias %q produced %x, want %x", name, got[0].Text, test.output)
			}
		}
	}
	if len(aliases) != fixture.AlgorithmCount || len(seen) != fixture.AliasCount {
		t.Fatalf("covered %d algorithms and %d aliases", len(aliases), len(seen))
	}
}

func TestStemmerTokenFilterBaselineBehavior(t *testing.T) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	if err != nil {
		t.Fatal(err)
	}
	input := []Token{
		{Text: "", Offset: 1, Position: 2},
		{Text: "running", Offset: 3, Position: 4},
		{Text: "cats", Offset: 11, Position: 5},
		{Text: "easily", Offset: 16, Position: 6},
		{Text: "connection", Offset: 23, Position: 7},
	}
	got, err := filter.Filter(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	want := []Token{
		{Text: "", Offset: 1, Position: 2},
		{Text: "run", Offset: 3, Position: 4},
		{Text: "cat", Offset: 11, Position: 5},
		{Text: "easili", Offset: 16, Position: 6},
		{Text: "connect", Offset: 23, Position: 7},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %#v, want %#v", got, want)
	}
	got[1].Offset = 99
	if input[1] != (Token{Text: "running", Offset: 3, Position: 4}) {
		t.Fatalf("input changed through result alias: %#v", input)
	}
	for _, empty := range [][]Token{nil, {}} {
		result, err := filter.Filter(context.Background(), empty)
		if err != nil || result == nil || len(result) != 0 {
			t.Fatalf("Filter(%#v) = %#v, %v", empty, result, err)
		}
	}
}

func TestStemmerTokenFilterLowercaseChain(t *testing.T) {
	lowercase := NewLowercaseTokenFilter()
	stemmer, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := lowercase.Filter(context.Background(), []Token{{Text: "Running"}, {Text: "Cats"}, {Text: "EASILY"}})
	if err != nil {
		t.Fatal(err)
	}
	tokens, err = stemmer.Filter(context.Background(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	if want := []Token{{Text: "run"}, {Text: "cat"}, {Text: "easili"}}; !reflect.DeepEqual(tokens, want) {
		t.Fatalf("tokens = %#v, want %#v", tokens, want)
	}
}

func TestStemmerTokenFilterContextCancellation(t *testing.T) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := filter.Filter(nil, nil); err == nil {
		t.Fatal("nil context succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := filter.Filter(canceled, []Token{{Text: "running"}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled error = %v", err)
	}
	midway := newCancelAfterChecks(4)
	if _, err := filter.Filter(midway, []Token{{Text: strings.Repeat("running", 16<<10)}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-filter error = %v", err)
	}
}

func TestStemmerTokenFilterConcurrentUse(t *testing.T) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{Language: "english"})
	if err != nil {
		t.Fatal(err)
	}
	input := []Token{{Text: "running", Offset: 7, Position: 9}, {Text: "connections", Offset: 15, Position: 10}}
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

func FuzzStemmerTokenFilter(f *testing.F) {
	for _, seed := range []string{"", "running", "easily", "connection", "τρέχοντας", string([]byte{0xc0, 0x80, 'r', 'u', 'n', 'n', 'i', 'n', 'g'}), string([]byte{0xed, 0xa0, 0x80})} {
		f.Add(seed)
	}
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	if err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, input string) {
		tokens := []Token{{Text: input, Offset: 123, Position: 456}}
		got, err := filter.Filter(context.Background(), tokens)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].Offset != 123 || got[0].Position != 456 || tokens[0].Text != input {
			t.Fatalf("metadata or input changed: %#v, %#v", tokens, got)
		}
	})
}

func BenchmarkStemmerTokenFilter(b *testing.B) {
	filter, err := NewStemmerTokenFilter(StemmerTokenFilterOptions{})
	if err != nil {
		b.Fatal(err)
	}
	tokens := make([]Token, 1024)
	for index := range tokens {
		tokens[index] = Token{Text: "running", Offset: uint32(index * 8), Position: uint32(index)}
	}
	b.SetBytes(int64(len(tokens) * len("running")))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := filter.Filter(context.Background(), tokens); err != nil {
			b.Fatal(err)
		}
	}
}
