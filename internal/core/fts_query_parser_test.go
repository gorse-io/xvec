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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
)

type ftsQueryParserFixture struct {
	BaselineCommit     string `json:"baseline_commit"`
	LexerSHA256        string `json:"lexer_sha256"`
	ParserSHA256       string `json:"parser_sha256"`
	ASTSHA256          string `json:"ast_sha256"`
	ParserSourceHash   string `json:"parser_source_sha256"`
	ParserTestsHash    string `json:"parser_tests_sha256"`
	RewriterHash       string `json:"rewriter_source_sha256"`
	TermIteratorHash   string `json:"term_iterator_source_sha256"`
	AndIteratorHash    string `json:"and_iterator_source_sha256"`
	OrIteratorHash     string `json:"or_iterator_source_sha256"`
	PhraseIteratorHash string `json:"phrase_iterator_source_sha256"`
	Cases              []struct {
		Name     string `json:"name"`
		Query    string `json:"query"`
		Analyzer string `json:"analyzer"`
		Default  string `json:"default_operator"`
		Want     string `json:"want"`
		Error    string `json:"error"`
	} `json:"cases"`
}

func TestFTSQueryParserBaselineFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/fts_query_parser_58375ff.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture ftsQueryParserFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.BaselineCommit != "58375ff7b8fdd0d6fc7d234e47567b179777883b" ||
		fixture.LexerSHA256 != "73d93e4311af4a74a76f8db964441ceefb8ac20f2ac2fb6470b0c2163a3b8d8d" ||
		fixture.ParserSHA256 != "9d4bca5dba6040da755c6d4af25e54c963df34542c0d0ff7e4b4b7fef7760bf5" ||
		fixture.ASTSHA256 != "6f0e03452b3d1df98d557d239dc2969cec80a5de45eae2ade7a25ed200481d85" ||
		fixture.ParserSourceHash != "3e63b894375a0c58283accd1b642d765b2cbf97020b69687c3ad478aa1ef825a" ||
		fixture.ParserTestsHash != "45a6a5af0bb2b38abb81554bf7f857e931c9cee3f29b4975280ce6febcc24b09" ||
		fixture.RewriterHash != "178d7c625c755f4de250f832569a1354544c034f760d7da18555a864ad6411f7" ||
		fixture.TermIteratorHash != "842080ba9bce4bacdb102705da18e9bc36a029138d4b8f269caeab4681f0f2c8" ||
		fixture.AndIteratorHash != "a524cd701608563f13fda44c2838772cb979b8f6af7195b19bfe527956cef59c" ||
		fixture.OrIteratorHash != "6645478f1ae200077f4686d076214aa8a70e81ddbee1bf9397403fcac011684d" ||
		fixture.PhraseIteratorHash != "6316f87dab229ba02fbb588f0047f23a9b988aab584d0d87fb9987896dd565f7" {
		t.Fatal("baseline identity drift")
	}
	standard := newFTSStandardTestPipeline(t)
	whitespace, err := NewFTSTokenizerPipeline(NewWhitespaceTokenizer())
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			analyzer := FTSAnalyzer(standard)
			if test.Analyzer == "whitespace" {
				analyzer = whitespace
			}
			defaultOperator, err := ParseFTSDefaultOperator(test.Default)
			if err != nil {
				t.Fatal(err)
			}
			node, err := ParseFTSQuery(context.Background(), test.Query, analyzer, defaultOperator)
			if test.Error != "" {
				if node != nil || err == nil || !strings.Contains(err.Error(), test.Error) {
					t.Fatalf("Parse = %#v, %v, want error containing %q", node, err, test.Error)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := node.String(); got != test.Want {
				t.Fatalf("AST = %q, want %q", got, test.Want)
			}
		})
	}
}

func TestLexFTSQueryLongestMatchAndLocations(t *testing.T) {
	query := "or AND Not ORbit 12 1.5 1. full-text C\\+\\+\n\"a \\\"b\\\"\" 中文 😀"
	tokens, err := LexFTSQuery(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	wantTypes := []FTSQueryTokenType{
		FTSQueryTokenOR, FTSQueryTokenAND, FTSQueryTokenNOT,
		FTSQueryTokenRegularID, FTSQueryTokenNumber, FTSQueryTokenNumber,
		FTSQueryTokenTerm, FTSQueryTokenRegularID, FTSQueryTokenTerm,
		FTSQueryTokenPhrase, FTSQueryTokenTerm, FTSQueryTokenDefault,
		FTSQueryTokenEOF,
	}
	wantTexts := []string{
		"or", "AND", "Not", "ORbit", "12", "1.5", "1.", "full-text",
		`C\+\+`, `"a \"b\""`, "中文", "😀", "",
	}
	if len(tokens) != len(wantTypes) {
		t.Fatalf("token count = %d: %#v", len(tokens), tokens)
	}
	for index, token := range tokens {
		if token.Type != wantTypes[index] || token.Text != wantTexts[index] {
			t.Errorf("token %d = %s %q, want %s %q", index, token.Type, token.Text, wantTypes[index], wantTexts[index])
		}
		if token.End < token.Offset || index > 0 && token.Offset < tokens[index-1].End {
			t.Errorf("token %d has invalid offsets [%d,%d)", index, token.Offset, token.End)
		}
	}
	phrase := tokens[9]
	if phrase.Line != 2 || phrase.Column != 0 {
		t.Fatalf("phrase location = %d:%d", phrase.Line, phrase.Column)
	}
	if eof := tokens[len(tokens)-1]; eof.Offset != uint32(len(query)) || eof.End != uint32(len(query)) {
		t.Fatalf("EOF = %#v", eof)
	}
}

func TestLexFTSQueryGrammarEdges(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"AND1 123abc abc.123 _id-2 a@b", []string{"REGULAR_ID:AND1", "TERM:123abc", "TERM:abc.123", "REGULAR_ID:_id-2", "TERM:a@b"}},
		{"\"unclosed phrase", []string{`DEFAULT:"`, "REGULAR_ID:unclosed", "REGULAR_ID:phrase"}},
		{"! !", []string{"DEFAULT:!", "DEFAULT:!"}},
		{"\v\f", []string{"DEFAULT:\v", "DEFAULT:\f"}},
		{"a\\-b path\\\\dir", []string{`TERM:a\-b`, `TERM:path\\dir`}},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%x", test.input), func(t *testing.T) {
			tokens, err := LexFTSQuery(context.Background(), test.input)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]string, 0, len(tokens)-1)
			for _, token := range tokens[:len(tokens)-1] {
				got = append(got, token.Type.String()+":"+token.Text)
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("tokens = %#v, want %#v", got, test.want)
			}
		})
	}
	for _, character := range []byte(`-+=&|!(){}[]^"~*?:\/`) {
		query := "x\\" + string(character)
		tokens, err := LexFTSQuery(context.Background(), query)
		if err != nil {
			t.Fatal(err)
		}
		if len(tokens) != 2 || tokens[0].Type != FTSQueryTokenTerm || tokens[0].Text != query {
			t.Errorf("escaped %q tokens = %#v", character, tokens)
		}
	}
}

func TestFTSTokenizerPipeline(t *testing.T) {
	tokenizer, err := NewStandardTokenizer(DefaultStandardTokenizerOptions())
	if err != nil {
		t.Fatal(err)
	}
	filters := []TokenFilter{NewLowercaseTokenFilter(), NewASCIIFoldingTokenFilter()}
	pipeline, err := NewFTSTokenizerPipeline(tokenizer, filters...)
	if err != nil {
		t.Fatal(err)
	}
	filters[0] = nil
	if pipeline.TokenizerName() != "standard" || !reflect.DeepEqual(pipeline.FilterNames(), []string{"lowercase", "ascii_folding"}) {
		t.Fatalf("pipeline metadata = %q %#v", pipeline.TokenizerName(), pipeline.FilterNames())
	}
	names := pipeline.FilterNames()
	names[0] = "changed"
	if pipeline.FilterNames()[0] != "lowercase" {
		t.Fatal("FilterNames aliases pipeline state")
	}
	tokens, err := pipeline.Analyze(context.Background(), "CAFÉ")
	if err != nil {
		t.Fatal(err)
	}
	if want := []Token{{Text: "cafe", Offset: 0, Position: 0}}; !reflect.DeepEqual(tokens, want) {
		t.Fatalf("Analyze = %#v, want %#v", tokens, want)
	}
}

type nilFTSTestTokenizer struct{}

func (*nilFTSTestTokenizer) Name() string { return "nil" }
func (*nilFTSTestTokenizer) Tokenize(context.Context, string) ([]Token, error) {
	return nil, nil
}

type nilFTSTestFilter struct{}

func (*nilFTSTestFilter) Name() string { return "nil" }
func (*nilFTSTestFilter) Filter(context.Context, []Token) ([]Token, error) {
	return nil, nil
}

func TestFTSTokenizerPipelineInvalidAndCancellation(t *testing.T) {
	var tokenizer *nilFTSTestTokenizer
	if pipeline, err := NewFTSTokenizerPipeline(tokenizer); pipeline != nil || !errors.Is(err, ErrInvalidFTSAnalyzer) {
		t.Fatalf("typed nil tokenizer = %#v, %v", pipeline, err)
	}
	var filter *nilFTSTestFilter
	if pipeline, err := NewFTSTokenizerPipeline(NewWhitespaceTokenizer(), filter); pipeline != nil || !errors.Is(err, ErrInvalidFTSAnalyzer) {
		t.Fatalf("typed nil filter = %#v, %v", pipeline, err)
	}
	pipeline, err := NewFTSTokenizerPipeline(NewWhitespaceTokenizer())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pipeline.Analyze(nil, "text"); !errors.Is(err, ErrInvalidFTSAnalyzer) {
		t.Fatalf("nil context = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := pipeline.Analyze(canceled, "text"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled = %v", err)
	}
	if _, err := (FTSAnalyzerFunc)(nil).Analyze(context.Background(), "text"); !errors.Is(err, ErrInvalidFTSAnalyzer) {
		t.Fatalf("nil analyzer function = %v", err)
	}
}

func TestParseFTSQueryASTAndModifiers(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	tests := []struct {
		query string
		op    FTSDefaultOperator
		want  string
		type_ FTSQueryNodeType
	}{
		{"vector", FTSDefaultOperatorOR, "vector", FTSQueryNodeTerm},
		{"full-text", FTSDefaultOperatorOR, "OR(full text)", FTSQueryNodeOr},
		{"full-text", FTSDefaultOperatorAND, "AND(full text)", FTSQueryNodeAnd},
		{"+full-text", FTSDefaultOperatorOR, "+OR(full text)", FTSQueryNodeOr},
		{"cat OR dog", FTSDefaultOperatorOR, "OR(cat dog)", FTSQueryNodeOr},
		{"cat AND dog", FTSDefaultOperatorOR, "AND(cat dog)", FTSQueryNodeAnd},
		{"a OR b AND c", FTSDefaultOperatorOR, "OR(a AND(b c))", FTSQueryNodeOr},
		{"a AND b OR c AND d", FTSDefaultOperatorOR, "OR(AND(a b) AND(c d))", FTSQueryNodeOr},
		{"a NOT (b OR c)", FTSDefaultOperatorOR, "AND(a -OR(b c))", FTSQueryNodeAnd},
		{"a AND NOT b", FTSDefaultOperatorOR, "AND(a -b)", FTSQueryNodeAnd},
		{"+a b -c", FTSDefaultOperatorAND, "AND(+a b -c)", FTSQueryNodeAnd},
		{"+(a OR b) c", FTSDefaultOperatorOR, "OR(+OR(a b) c)", FTSQueryNodeOr},
		{"(a b) c", FTSDefaultOperatorAND, "AND(AND(a b) c)", FTSQueryNodeAnd},
		{"a AND b c", FTSDefaultOperatorOR, "AND(a OR(b c))", FTSQueryNodeAnd},
		{"!!! ??? ...", FTSDefaultOperatorOR, "<empty>", FTSQueryNodeEmpty},
		{"\"!!! ???\"", FTSDefaultOperatorOR, `""`, FTSQueryNodePhrase},
	}
	for _, test := range tests {
		t.Run(test.query+"/"+test.op.String(), func(t *testing.T) {
			node, err := ParseFTSQuery(context.Background(), test.query, pipeline, test.op)
			if err != nil {
				t.Fatal(err)
			}
			if node.Type() != test.type_ || node.String() != test.want {
				t.Fatalf("AST = %s %q, want %s %q", node.Type(), node.String(), test.type_, test.want)
			}
		})
	}

	node, err := ParseFTSQuery(context.Background(), `+"exact phrase"`, pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	phrase, ok := node.(*FTSPhraseQueryNode)
	if !ok || !phrase.Flags.Must || phrase.Flags.MustNot || !reflect.DeepEqual(phrase.Terms, []string{"exact", "phrase"}) {
		t.Fatalf("phrase = %#v", node)
	}
}

func TestAnalyzeFTSMatchQuery(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	for _, test := range []struct {
		op   FTSDefaultOperator
		want string
	}{
		{FTSDefaultOperatorOR, "OR(apple and banana)"},
		{FTSDefaultOperatorAND, "AND(apple and banana)"},
	} {
		node, err := AnalyzeFTSMatchQuery(context.Background(), "Apple AND banana", pipeline, test.op)
		if err != nil || node.String() != test.want {
			t.Errorf("Analyze(%s) = %#v, %v", test.op, node, err)
		}
	}
	if node, err := AnalyzeFTSMatchQuery(context.Background(), "!!!", pipeline, FTSDefaultOperatorOR); err != nil || node.Type() != FTSQueryNodeEmpty {
		t.Fatalf("empty match = %#v, %v", node, err)
	}
	if node, err := AnalyzeFTSMatchQuery(nil, "apple", pipeline, FTSDefaultOperatorOR); node != nil || !errors.Is(err, ErrInvalidFTSQuery) {
		t.Fatalf("nil context = %#v, %v", node, err)
	}
	if node, err := AnalyzeFTSMatchQuery(context.Background(), "apple", nil, FTSDefaultOperatorOR); node != nil || !errors.Is(err, ErrInvalidFTSQuery) {
		t.Fatalf("nil analyzer = %#v, %v", node, err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := AnalyzeFTSMatchQuery(canceled, "apple", pipeline, FTSDefaultOperatorOR); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled = %v", err)
	}
	sentinel := errors.New("match analysis failed")
	failed := FTSAnalyzerFunc(func(context.Context, string) ([]Token, error) { return nil, sentinel })
	if _, err := AnalyzeFTSMatchQuery(context.Background(), "apple", failed, FTSDefaultOperatorOR); !errors.Is(err, sentinel) {
		t.Fatalf("analysis error = %v", err)
	}
	oversized := make([]Token, MaxFTSQueryTokens+1)
	tooMany := FTSAnalyzerFunc(func(context.Context, string) ([]Token, error) { return oversized, nil })
	if _, err := AnalyzeFTSMatchQuery(context.Background(), "apple", tooMany, FTSDefaultOperatorOR); !errors.Is(err, ErrFTSQueryTooComplex) {
		t.Fatalf("oversized analysis = %v", err)
	}
}

func TestFTSQueryASTDebugText(t *testing.T) {
	term := &FTSTermQueryNode{
		Flags: FTSQueryModifier{Must: true, MustNot: true, Should: true, Boost: 2},
		Term:  "vector",
	}
	if got := term.String(); got != "+vector^2.000000" {
		t.Fatalf("term text = %q", got)
	}
	phrase := &FTSPhraseQueryNode{
		Flags: FTSQueryModifier{MustNot: true, Boost: 0.5},
		Terms: []string{"exact", "phrase"},
	}
	if got := phrase.String(); got != `-"exact phrase"^0.500000` {
		t.Fatalf("phrase text = %q", got)
	}
	composite := &FTSAndQueryNode{
		Flags:    FTSQueryModifier{Should: true, Boost: 3},
		Children: []FTSQueryNode{term, phrase},
	}
	if got := composite.String(); got != `?AND(+vector^2.000000 -"exact phrase"^0.500000)` {
		t.Fatalf("composite text = %q", got)
	}
	if FTSQueryNodeType(99).String() != "UNKNOWN(99)" || FTSQueryTokenType(99).String() != "UNKNOWN(99)" || FTSDefaultOperator(99).String() != "UNKNOWN(99)" {
		t.Fatal("unknown enum text differs")
	}
}

func TestParseFTSQueryAnalysisAndEscapes(t *testing.T) {
	standard := newFTSStandardTestPipeline(t)
	tests := map[string]string{
		`"machine, LEARNING!"`: `"machine learning"`,
		`"host:port"`:          `"host:port"`,
		`'hello world'`:        "OR(hello world)",
		`"unclosed phrase`:     "OR(unclosed phrase)",
	}
	for query, want := range tests {
		node, err := ParseFTSQuery(context.Background(), query, standard, FTSDefaultOperatorOR)
		if err != nil {
			t.Fatalf("Parse(%q): %v", query, err)
		}
		if node.String() != want {
			t.Errorf("Parse(%q) = %q, want %q", query, node.String(), want)
		}
	}

	whitespace, err := NewFTSTokenizerPipeline(NewWhitespaceTokenizer())
	if err != nil {
		t.Fatal(err)
	}
	escaped := map[string]string{
		`C\+\+`:             "C++",
		`a\-b`:              "a-b",
		`path\\dir`:         `path\dir`,
		`"hello \"world\""`: `"hello "world""`,
		`"a\\b"`:            `"a\b"`,
	}
	for query, want := range escaped {
		node, err := ParseFTSQuery(context.Background(), query, whitespace, FTSDefaultOperatorOR)
		if err != nil {
			t.Fatalf("Parse(%q): %v", query, err)
		}
		if node.String() != want {
			t.Errorf("Parse(%q) = %q, want %q", query, node.String(), want)
		}
	}
}

func TestParseFTSQueryErrorsAndLocations(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	tests := []struct {
		query   string
		kind    error
		message string
		offset  uint32
	}{
		{"", ErrFTSQuerySyntax, "expected a term", 0},
		{"()", ErrFTSQuerySyntax, "expected a term", 1},
		{"NOT a", ErrFTSQuerySyntax, "expected a term", 0},
		{"(a OR b", ErrFTSQuerySyntax, "expected ')'", 7},
		{"a OR", ErrFTSQuerySyntax, "expected a term", 4},
		{"a AND NOT", ErrFTSQuerySyntax, "expected a term", 9},
		{"+", ErrFTSQuerySyntax, "expected an atom", 1},
		{"host:port", ErrUnsupportedFTSQuery, "field-prefixed", 0},
		{"term^2", ErrUnsupportedFTSQuery, "boost", 4},
		{"term^x", ErrFTSQuerySyntax, "expected a number", 5},
	}
	for _, test := range tests {
		t.Run(test.query, func(t *testing.T) {
			node, err := ParseFTSQuery(context.Background(), test.query, pipeline, FTSDefaultOperatorOR)
			if node != nil || !errors.Is(err, test.kind) || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("Parse = %#v, %v", node, err)
			}
			var parseError *FTSQueryParseError
			if !errors.As(err, &parseError) || parseError.Offset != test.offset || parseError.Line != 1 {
				t.Fatalf("parse error = %#v", parseError)
			}
		})
	}
	node, err := ParseFTSQuery(context.Background(), "中文 OR\n)", pipeline, FTSDefaultOperatorOR)
	var parseError *FTSQueryParseError
	if node != nil || !errors.As(err, &parseError) || parseError.Offset != 10 || parseError.Line != 2 || parseError.Column != 0 {
		t.Fatalf("multiline error = %#v, %v", parseError, err)
	}
}

func TestParseFTSQueryInvalidConfigurationCancellationAndDepth(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	if node, err := ParseFTSQuery(nil, "a", pipeline, FTSDefaultOperatorOR); node != nil || !errors.Is(err, ErrInvalidFTSQuery) {
		t.Fatalf("nil context = %#v, %v", node, err)
	}
	var analyzer *FTSTokenizerPipeline
	if node, err := ParseFTSQuery(context.Background(), "a", analyzer, FTSDefaultOperatorOR); node != nil || !errors.Is(err, ErrInvalidFTSQuery) {
		t.Fatalf("nil analyzer = %#v, %v", node, err)
	}
	if node, err := ParseFTSQuery(context.Background(), "a", FTSAnalyzerFunc(nil), FTSDefaultOperatorOR); node != nil || !errors.Is(err, ErrInvalidFTSQuery) {
		t.Fatalf("nil analyzer function = %#v, %v", node, err)
	}
	if node, err := ParseFTSQuery(context.Background(), "a", pipeline, 99); node != nil || !errors.Is(err, ErrInvalidFTSQuery) {
		t.Fatalf("invalid operator = %#v, %v", node, err)
	}
	for input, want := range map[string]FTSDefaultOperator{"": FTSDefaultOperatorOR, "or": FTSDefaultOperatorOR, "AnD": FTSDefaultOperatorAND} {
		if got, err := ParseFTSDefaultOperator(input); err != nil || got != want {
			t.Errorf("ParseFTSDefaultOperator(%q) = %v, %v", input, got, err)
		}
	}
	if _, err := ParseFTSDefaultOperator("XOR"); !errors.Is(err, ErrInvalidFTSQuery) {
		t.Fatalf("invalid default operator = %v", err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := LexFTSQuery(canceled, "a"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled lexer = %v", err)
	}
	if _, err := ParseFTSQuery(canceled, "a", pipeline, FTSDefaultOperatorOR); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled parser = %v", err)
	}
	midway := newCancelAfterChecks(3)
	if _, err := ParseFTSQuery(midway, strings.Repeat("word ", 5000), pipeline, FTSDefaultOperatorOR); !errors.Is(err, context.Canceled) {
		t.Fatalf("midway cancellation = %v", err)
	}
	longTerm := newCancelAfterChecks(4)
	if _, err := LexFTSQuery(longTerm, strings.Repeat("a", 1<<20)); !errors.Is(err, context.Canceled) {
		t.Fatalf("long-term cancellation = %v", err)
	}
	longPhrase := newCancelAfterChecks(4)
	if _, err := LexFTSQuery(longPhrase, `"`+strings.Repeat("a", 1<<20)+`"`); !errors.Is(err, context.Canceled) {
		t.Fatalf("long-phrase cancellation = %v", err)
	}

	deep := strings.Repeat("(", MaxFTSQueryDepth+1) + "a" + strings.Repeat(")", MaxFTSQueryDepth+1)
	if _, err := ParseFTSQuery(context.Background(), deep, pipeline, FTSDefaultOperatorOR); !errors.Is(err, ErrFTSQueryTooComplex) {
		t.Fatalf("deep query = %v", err)
	}
	maximumDepth := strings.Repeat("(", MaxFTSQueryDepth) + "a" + strings.Repeat(")", MaxFTSQueryDepth)
	if node, err := ParseFTSQuery(context.Background(), maximumDepth, pipeline, FTSDefaultOperatorOR); err != nil || node.String() != "a" {
		t.Fatalf("maximum-depth query = %#v, %v", node, err)
	}
	tooManyTokens := strings.Repeat("!", MaxFTSQueryTokens+1)
	if _, err := LexFTSQuery(context.Background(), tooManyTokens); !errors.Is(err, ErrFTSQueryTooComplex) {
		t.Fatalf("large token stream = %v", err)
	}
	tooManyAnalyzed := make([]Token, MaxFTSQueryTokens+1)
	oversizedAnalyzer := FTSAnalyzerFunc(func(context.Context, string) ([]Token, error) {
		return tooManyAnalyzed, nil
	})
	if _, err := ParseFTSQuery(context.Background(), "a", oversizedAnalyzer, FTSDefaultOperatorOR); !errors.Is(err, ErrFTSQueryTooComplex) {
		t.Fatalf("large analyzed stream = %v", err)
	}
	cancellableTokens := make([]Token, 5000)
	for index := range cancellableTokens {
		cancellableTokens[index] = Token{Text: "a"}
	}
	cancellableAnalyzer := FTSAnalyzerFunc(func(context.Context, string) ([]Token, error) {
		return cancellableTokens, nil
	})
	if _, err := AnalyzeFTSMatchQuery(newCancelAfterChecks(3), "a", cancellableAnalyzer, FTSDefaultOperatorOR); !errors.Is(err, context.Canceled) {
		t.Fatalf("analyzed AST cancellation = %v", err)
	}
}

func TestParseFTSQueryAnalyzerFailureAndOwnership(t *testing.T) {
	sentinel := errors.New("analysis failed")
	failed := FTSAnalyzerFunc(func(context.Context, string) ([]Token, error) {
		return nil, sentinel
	})
	if node, err := ParseFTSQuery(context.Background(), "term AND", failed, FTSDefaultOperatorOR); node != nil || !errors.Is(err, ErrFTSQuerySyntax) {
		t.Fatalf("syntax must precede analysis = %#v, %v", node, err)
	}
	analyzed := 0
	counting := FTSAnalyzerFunc(func(context.Context, string) ([]Token, error) {
		analyzed++
		return []Token{{Text: "unexpected"}}, nil
	})
	for _, query := range []string{"host:port", "term^2"} {
		if node, err := ParseFTSQuery(context.Background(), query, counting, FTSDefaultOperatorOR); node != nil || !errors.Is(err, ErrUnsupportedFTSQuery) {
			t.Fatalf("unsupported %q = %#v, %v", query, node, err)
		}
	}
	if analyzed != 0 {
		t.Fatalf("unsupported atoms invoked analyzer %d times", analyzed)
	}
	if node, err := ParseFTSQuery(context.Background(), "term", failed, FTSDefaultOperatorOR); node != nil || !errors.Is(err, sentinel) {
		t.Fatalf("analyzer failure = %#v, %v", node, err)
	}
	shared := []Token{{Text: "owned"}, {Text: "terms"}}
	analyzer := FTSAnalyzerFunc(func(context.Context, string) ([]Token, error) { return shared, nil })
	node, err := ParseFTSQuery(context.Background(), `"source"`, analyzer, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	shared[0].Text = "changed"
	phrase := node.(*FTSPhraseQueryNode)
	if !reflect.DeepEqual(phrase.Terms, []string{"owned", "terms"}) {
		t.Fatalf("parser aliases analyzer output: %#v", phrase.Terms)
	}
}

func TestParseFTSQueryConcurrentUse(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	const query = `+Vector -slow "Machine Learning" OR database`
	wantNode, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
	if err != nil {
		t.Fatal(err)
	}
	want := wantNode.String()
	var wait sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for worker := 0; worker < 32; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 100; iteration++ {
				node, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
				if err != nil || node.String() != want {
					errorsChannel <- fmt.Errorf("Parse = %#v, %v", node, err)
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

func FuzzLexFTSQuery(f *testing.F) {
	for _, seed := range []string{"", "a AND b", `+foo -bar "exact phrase"`, "中文 OR 😀", "(a NOT b)"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, query string) {
		tokens, err := LexFTSQuery(context.Background(), query)
		if err != nil {
			return
		}
		if len(tokens) == 0 || tokens[len(tokens)-1].Type != FTSQueryTokenEOF {
			t.Fatal("missing EOF")
		}
		previousEnd := uint32(0)
		for index, token := range tokens {
			if token.Offset < previousEnd || token.End < token.Offset || uint64(token.End) > uint64(len(query)) {
				t.Fatalf("token %d has invalid range %#v", index, token)
			}
			previousEnd = token.End
		}
	})
}

func FuzzParseFTSQuery(f *testing.F) {
	for _, seed := range []string{"a", "a AND b", `+foo -bar "exact phrase"`, "!!!", "(a OR b) NOT c"} {
		f.Add(seed)
	}
	pipeline := newFTSStandardTestPipeline(f)
	f.Fuzz(func(t *testing.T, query string) {
		node, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
		if err != nil {
			return
		}
		if node == nil || node.String() == "" {
			t.Fatal("successful parse returned an empty AST")
		}
		assertValidFTSQueryAST(t, node, 0)
	})
}

func BenchmarkParseFTSQuery(b *testing.B) {
	pipeline := newFTSStandardTestPipeline(b)
	query := `+(vector database OR "nearest neighbor") AND retrieval NOT "slow scan"`
	b.ReportAllocs()
	b.SetBytes(int64(len(query)))
	for b.Loop() {
		if _, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR); err != nil {
			b.Fatal(err)
		}
	}
}

func newFTSStandardTestPipeline(t testing.TB) *FTSTokenizerPipeline {
	t.Helper()
	tokenizer, err := NewStandardTokenizer(DefaultStandardTokenizerOptions())
	if err != nil {
		t.Fatal(err)
	}
	pipeline, err := NewFTSTokenizerPipeline(tokenizer, NewLowercaseTokenFilter())
	if err != nil {
		t.Fatal(err)
	}
	return pipeline
}

func assertValidFTSQueryAST(t testing.TB, node FTSQueryNode, depth int) {
	t.Helper()
	if node == nil || depth > MaxFTSQueryDepth+2 {
		t.Fatalf("invalid AST at depth %d", depth)
	}
	modifier := node.Modifier()
	if modifier.Boost != 1 {
		t.Fatalf("parser-produced boost = %v", modifier.Boost)
	}
	switch typed := node.(type) {
	case *FTSTermQueryNode:
	case *FTSPhraseQueryNode:
		if typed.Terms == nil {
			t.Fatal("phrase terms are nil")
		}
	case *FTSAndQueryNode:
		for _, child := range typed.Children {
			assertValidFTSQueryAST(t, child, depth+1)
		}
	case *FTSOrQueryNode:
		for _, child := range typed.Children {
			assertValidFTSQueryAST(t, child, depth+1)
		}
	case *FTSEmptyQueryNode:
	default:
		t.Fatalf("unknown AST node %T", node)
	}
}
