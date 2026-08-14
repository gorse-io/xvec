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

package ftscolumn

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.NoError(t, err)

	var fixture ftsQueryParserFixture
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.BaselineCommit == "58375ff7b8fdd0d6fc7d234e47567b179777883b",

		"baseline identity drift")
	require.True(t, fixture.LexerSHA256 == "73d93e4311af4a74a76f8db964441ceefb8ac20f2ac2fb6470b0c2163a3b8d8d",

		"baseline identity drift")
	require.True(t, fixture.ParserSHA256 == "9d4bca5dba6040da755c6d4af25e54c963df34542c0d0ff7e4b4b7fef7760bf5",

		"baseline identity drift")
	require.True(t, fixture.ASTSHA256 == "6f0e03452b3d1df98d557d239dc2969cec80a5de45eae2ade7a25ed200481d85",

		"baseline identity drift")
	require.True(t, fixture.ParserSourceHash == "3e63b894375a0c58283accd1b642d765b2cbf97020b69687c3ad478aa1ef825a",

		"baseline identity drift")
	require.True(t, fixture.ParserTestsHash == "45a6a5af0bb2b38abb81554bf7f857e931c9cee3f29b4975280ce6febcc24b09",

		"baseline identity drift")
	require.True(t, fixture.RewriterHash == "178d7c625c755f4de250f832569a1354544c034f760d7da18555a864ad6411f7",

		"baseline identity drift")
	require.True(t, fixture.TermIteratorHash == "842080ba9bce4bacdb102705da18e9bc36a029138d4b8f269caeab4681f0f2c8",

		"baseline identity drift")
	require.True(t, fixture.AndIteratorHash == "a524cd701608563f13fda44c2838772cb979b8f6af7195b19bfe527956cef59c",

		"baseline identity drift")
	require.True(t, fixture.OrIteratorHash == "6645478f1ae200077f4686d076214aa8a70e81ddbee1bf9397403fcac011684d",

		"baseline identity drift")
	require.True(t, fixture.PhraseIteratorHash == "6316f87dab229ba02fbb588f0047f23a9b988aab584d0d87fb9987896dd565f7",
		"baseline identity drift")

	standard := newFTSStandardTestPipeline(t)
	whitespace, err := tokenizer.NewFTSTokenizerPipeline(tokenizer.NewWhitespaceTokenizer())
	require.NoError(t, err)

	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			analyzer := tokenizer.FTSAnalyzer(standard)
			if test.Analyzer == "whitespace" {
				analyzer = whitespace
			}
			defaultOperator, err := ParseFTSDefaultOperator(test.Default)
			require.NoError(t, err)

			node, err := ParseFTSQuery(context.Background(), test.Query, analyzer, defaultOperator)
			if test.Error != "" {
				require.Nil(t, node)
				require.Error(t, err)
				require.Contains(t, err.Error(), test.Error)

				return
			}
			require.NoError(t, err)
			{
				got := node.String()
				require.Equal(t, test.Want, got)
			}
		})
	}
}

func TestLexFTSQueryLongestMatchAndLocations(t *testing.T) {
	query := "or AND Not ORbit 12 1.5 1. full-text C\\+\\+\n\"a \\\"b\\\"\" 中文 😀"
	tokens, err := LexFTSQuery(context.Background(), query)
	require.NoError(t, err)

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
	require.Len(t, tokens, len(wantTypes))

	for index, token := range tokens {
		assert.False(t, token.Type != wantTypes[index] || token.Text != wantTexts[index])
		assert.False(t, token.End < token.Offset || index > 0 && token.Offset < tokens[index-1].End)
	}
	phrase := tokens[9]
	require.True(t, phrase.Line == 2)
	require.True(t, phrase.Column == 0)
	{
		eof := tokens[len(tokens)-1]
		require.Equal(t, uint32(len(query)), eof.Offset)
		require.Equal(t, uint32(len(query)), eof.End)
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
			require.NoError(t, err)

			got := make([]string, 0, len(tokens)-1)
			for _, token := range tokens[:len(tokens)-1] {
				got = append(got, token.Type.String()+":"+token.Text)
			}
			require.Equal(t, test.want, got)
		})
	}
	for _, character := range []byte(`-+=&|!(){}[]^"~*?:\/`) {
		query := "x\\" + string(character)
		tokens, err := LexFTSQuery(context.Background(), query)
		require.NoError(t, err)
		assert.False(t, len(tokens) != 2 || tokens[0].Type != FTSQueryTokenTerm || tokens[0].Text != query)
	}
}

func TestFTSTokenizerPipeline(t *testing.T) {
	standardTokenizer, err := tokenizer.NewStandardTokenizer(tokenizer.DefaultStandardTokenizerOptions())
	require.NoError(t, err)

	filters := []tokenizer.TokenFilter{tokenizer.NewLowercaseTokenFilter(), tokenizer.NewASCIIFoldingTokenFilter()}
	pipeline, err := tokenizer.NewFTSTokenizerPipeline(standardTokenizer, filters...)
	require.NoError(t, err)

	filters[0] = nil
	require.True(t, pipeline.TokenizerName() == "standard")
	require.Equal(t, []string{"lowercase", "ascii_folding"}, pipeline.FilterNames())

	names := pipeline.FilterNames()
	names[0] = "changed"
	require.True(t, pipeline.FilterNames()[0] == "lowercase",
		"FilterNames aliases pipeline state")

	tokens, err := pipeline.Analyze(context.Background(), "CAFÉ")
	require.NoError(t, err)
	{
		want := []tokenizer.Token{{Text: "cafe", Offset: 0, Position: 0}}
		require.Equal(t, want, tokens)
	}
}

type nilFTSTestTokenizer struct{}

func (*nilFTSTestTokenizer) Name() string { return "nil" }
func (*nilFTSTestTokenizer) Tokenize(context.Context, string) ([]tokenizer.Token, error) {
	return nil, nil
}

type nilFTSTestFilter struct{}

func (*nilFTSTestFilter) Name() string { return "nil" }
func (*nilFTSTestFilter) Filter(context.Context, []tokenizer.Token) ([]tokenizer.Token, error) {
	return nil, nil
}

func TestFTSTokenizerPipelineInvalidAndCancellation(t *testing.T) {
	var typedTokenizer *nilFTSTestTokenizer
	{
		pipeline, err := tokenizer.NewFTSTokenizerPipeline(typedTokenizer)
		require.Nil(t, pipeline)
		require.ErrorIs(t, err, tokenizer.ErrInvalidFTSAnalyzer)
	}

	var filter *nilFTSTestFilter
	{
		pipeline, err := tokenizer.NewFTSTokenizerPipeline(tokenizer.NewWhitespaceTokenizer(), filter)
		require.Nil(t, pipeline)
		require.ErrorIs(t, err, tokenizer.ErrInvalidFTSAnalyzer)
	}

	pipeline, err := tokenizer.NewFTSTokenizerPipeline(tokenizer.NewWhitespaceTokenizer())
	require.NoError(t, err)
	{
		_, err := pipeline.Analyze(nil, "text")
		require.ErrorIs(t, err, tokenizer.ErrInvalidFTSAnalyzer)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := pipeline.Analyze(canceled, "text")
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := (tokenizer.FTSAnalyzerFunc)(nil).Analyze(context.Background(), "text")
		require.ErrorIs(t, err, tokenizer.ErrInvalidFTSAnalyzer)
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
			require.NoError(t, err)
			require.Equal(t, test.type_, node.Type())
			require.Equal(t, test.want, node.String())
		})
	}

	node, err := ParseFTSQuery(context.Background(), `+"exact phrase"`, pipeline, FTSDefaultOperatorOR)
	require.NoError(t, err)

	phrase, ok := node.(*FTSPhraseQueryNode)
	require.True(t, ok)
	require.True(t, phrase.Flags.Must)
	require.False(t, phrase.Flags.MustNot)
	require.Equal(t, []string{"exact", "phrase"}, phrase.Terms)
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
		assert.False(t, err != nil || node.String() != test.want)
	}
	{
		node, err := AnalyzeFTSMatchQuery(context.Background(), "!!!", pipeline, FTSDefaultOperatorOR)
		require.NoError(t, err)
		require.Equal(t, FTSQueryNodeEmpty, node.Type())
	}
	{
		node, err := AnalyzeFTSMatchQuery(nil, "apple", pipeline, FTSDefaultOperatorOR)
		require.Nil(t, node)
		require.ErrorIs(t, err, ErrInvalidFTSQuery)
	}
	{
		node, err := AnalyzeFTSMatchQuery(context.Background(), "apple", nil, FTSDefaultOperatorOR)
		require.Nil(t, node)
		require.ErrorIs(t, err, ErrInvalidFTSQuery)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := AnalyzeFTSMatchQuery(canceled, "apple", pipeline, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, context.Canceled)
	}

	sentinel := errors.New("match analysis failed")
	failed := tokenizer.FTSAnalyzerFunc(func(context.Context, string) ([]tokenizer.Token, error) { return nil, sentinel })
	{
		_, err := AnalyzeFTSMatchQuery(context.Background(), "apple", failed, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, sentinel)
	}

	oversized := make([]tokenizer.Token, MaxFTSQueryTokens+1)
	tooMany := tokenizer.FTSAnalyzerFunc(func(context.Context, string) ([]tokenizer.Token, error) { return oversized, nil })
	{
		_, err := AnalyzeFTSMatchQuery(context.Background(), "apple", tooMany, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, ErrFTSQueryTooComplex)
	}
}

func TestFTSQueryASTDebugText(t *testing.T) {
	term := &FTSTermQueryNode{
		Flags: FTSQueryModifier{Must: true, MustNot: true, Should: true, Boost: 2},
		Term:  "vector",
	}
	{
		got := term.String()
		require.True(t, got == "+vector^2.000000")
	}

	phrase := &FTSPhraseQueryNode{
		Flags: FTSQueryModifier{MustNot: true, Boost: 0.5},
		Terms: []string{"exact", "phrase"},
	}
	{
		got := phrase.String()
		require.True(t, got == `-"exact phrase"^0.500000`)
	}

	composite := &FTSAndQueryNode{
		Flags:    FTSQueryModifier{Should: true, Boost: 3},
		Children: []FTSQueryNode{term, phrase},
	}
	{
		got := composite.String()
		require.True(t, got == `?AND(+vector^2.000000 -"exact phrase"^0.500000)`)
	}
	require.True(t, FTSQueryNodeType(99).String() == "UNKNOWN(99)",
		"unknown enum text differs")
	require.True(t, FTSQueryTokenType(99).String() == "UNKNOWN(99)",
		"unknown enum text differs")
	require.True(t, FTSDefaultOperator(99).String() == "UNKNOWN(99)",
		"unknown enum text differs")
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
		require.NoError(t, err)
		assert.Equal(t, want, node.String())
	}

	whitespace, err := tokenizer.NewFTSTokenizerPipeline(tokenizer.NewWhitespaceTokenizer())
	require.NoError(t, err)

	escaped := map[string]string{
		`C\+\+`:             "C++",
		`a\-b`:              "a-b",
		`path\\dir`:         `path\dir`,
		`"hello \"world\""`: `"hello "world""`,
		`"a\\b"`:            `"a\b"`,
	}
	for query, want := range escaped {
		node, err := ParseFTSQuery(context.Background(), query, whitespace, FTSDefaultOperatorOR)
		require.NoError(t, err)
		assert.Equal(t, want, node.String())
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
			require.Nil(t, node)
			require.ErrorIs(t, err, test.kind)
			require.Contains(t, err.Error(), test.message)

			var parseError *FTSQueryParseError
			require.ErrorAs(t, err, &parseError)
			require.Equal(t, test.offset, parseError.Offset)
			require.True(t, parseError.Line == 1)
		})
	}
	node, err := ParseFTSQuery(context.Background(), "中文 OR\n)", pipeline, FTSDefaultOperatorOR)
	var parseError *FTSQueryParseError
	require.Nil(t, node)
	require.ErrorAs(t, err, &parseError)
	require.True(t, parseError.Offset == 10)
	require.True(t, parseError.Line == 2)
	require.True(t, parseError.Column == 0)
}

func TestParseFTSQueryInvalidConfigurationCancellationAndDepth(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	{
		node, err := ParseFTSQuery(nil, "a", pipeline, FTSDefaultOperatorOR)
		require.Nil(t, node)
		require.ErrorIs(t, err, ErrInvalidFTSQuery)
	}

	var analyzer *tokenizer.FTSTokenizerPipeline
	{
		node, err := ParseFTSQuery(context.Background(), "a", analyzer, FTSDefaultOperatorOR)
		require.Nil(t, node)
		require.ErrorIs(t, err, ErrInvalidFTSQuery)
	}
	{
		node, err := ParseFTSQuery(context.Background(), "a", tokenizer.FTSAnalyzerFunc(nil), FTSDefaultOperatorOR)
		require.Nil(t, node)
		require.ErrorIs(t, err, ErrInvalidFTSQuery)
	}
	{
		node, err := ParseFTSQuery(context.Background(), "a", pipeline, 99)
		require.Nil(t, node)
		require.ErrorIs(t, err, ErrInvalidFTSQuery)
	}

	for input, want := range map[string]FTSDefaultOperator{"": FTSDefaultOperatorOR, "or": FTSDefaultOperatorOR, "AnD": FTSDefaultOperatorAND} {
		{
			got, err := ParseFTSDefaultOperator(input)
			assert.False(t, err != nil || got != want)
		}
	}
	{
		_, err := ParseFTSDefaultOperator("XOR")
		require.ErrorIs(t, err, ErrInvalidFTSQuery)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	{
		_, err := LexFTSQuery(canceled, "a")
		require.ErrorIs(t, err, context.Canceled)
	}
	{
		_, err := ParseFTSQuery(canceled, "a", pipeline, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, context.Canceled)
	}

	midway := newCancelAfterChecks(3)
	{
		_, err := ParseFTSQuery(midway, strings.Repeat("word ", 5000), pipeline, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, context.Canceled)
	}

	longTerm := newCancelAfterChecks(4)
	{
		_, err := LexFTSQuery(longTerm, strings.Repeat("a", 1<<20))
		require.ErrorIs(t, err, context.Canceled)
	}

	longPhrase := newCancelAfterChecks(4)
	{
		_, err := LexFTSQuery(longPhrase, `"`+strings.Repeat("a", 1<<20)+`"`)
		require.ErrorIs(t, err, context.Canceled)
	}

	deep := strings.Repeat("(", MaxFTSQueryDepth+1) + "a" + strings.Repeat(")", MaxFTSQueryDepth+1)
	{
		_, err := ParseFTSQuery(context.Background(), deep, pipeline, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, ErrFTSQueryTooComplex)
	}

	maximumDepth := strings.Repeat("(", MaxFTSQueryDepth) + "a" + strings.Repeat(")", MaxFTSQueryDepth)
	{
		node, err := ParseFTSQuery(context.Background(), maximumDepth, pipeline, FTSDefaultOperatorOR)
		require.NoError(t, err)
		require.True(t, node.String() == "a")
	}

	tooManyTokens := strings.Repeat("!", MaxFTSQueryTokens+1)
	{
		_, err := LexFTSQuery(context.Background(), tooManyTokens)
		require.ErrorIs(t, err, ErrFTSQueryTooComplex)
	}

	tooManyAnalyzed := make([]tokenizer.Token, MaxFTSQueryTokens+1)
	oversizedAnalyzer := tokenizer.FTSAnalyzerFunc(func(context.Context, string) ([]tokenizer.Token, error) {
		return tooManyAnalyzed, nil
	})
	{
		_, err := ParseFTSQuery(context.Background(), "a", oversizedAnalyzer, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, ErrFTSQueryTooComplex)
	}

	cancellableTokens := make([]tokenizer.Token, 5000)
	for index := range cancellableTokens {
		cancellableTokens[index] = tokenizer.Token{Text: "a"}
	}
	cancellableAnalyzer := tokenizer.FTSAnalyzerFunc(func(context.Context, string) ([]tokenizer.Token, error) {
		return cancellableTokens, nil
	})
	{
		_, err := AnalyzeFTSMatchQuery(newCancelAfterChecks(3), "a", cancellableAnalyzer, FTSDefaultOperatorOR)
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestParseFTSQueryAnalyzerFailureAndOwnership(t *testing.T) {
	sentinel := errors.New("analysis failed")
	failed := tokenizer.FTSAnalyzerFunc(func(context.Context, string) ([]tokenizer.Token, error) {
		return nil, sentinel
	})
	{
		node, err := ParseFTSQuery(context.Background(), "term AND", failed, FTSDefaultOperatorOR)
		require.Nil(t, node)
		require.ErrorIs(t, err, ErrFTSQuerySyntax)
	}

	analyzed := 0
	counting := tokenizer.FTSAnalyzerFunc(func(context.Context, string) ([]tokenizer.Token, error) {
		analyzed++
		return []tokenizer.Token{{Text: "unexpected"}}, nil
	})
	for _, query := range []string{"host:port", "term^2"} {
		{
			node, err := ParseFTSQuery(context.Background(), query, counting, FTSDefaultOperatorOR)
			require.Nil(t, node)
			require.ErrorIs(t, err, ErrUnsupportedFTSQuery)
		}
	}
	require.True(t, analyzed == 0)
	{
		node, err := ParseFTSQuery(context.Background(), "term", failed, FTSDefaultOperatorOR)
		require.Nil(t, node)
		require.ErrorIs(t, err, sentinel)
	}

	shared := []tokenizer.Token{{Text: "owned"}, {Text: "terms"}}
	analyzer := tokenizer.FTSAnalyzerFunc(func(context.Context, string) ([]tokenizer.Token, error) { return shared, nil })
	node, err := ParseFTSQuery(context.Background(), `"source"`, analyzer, FTSDefaultOperatorOR)
	require.NoError(t, err)

	shared[0].Text = "changed"
	phrase := node.(*FTSPhraseQueryNode)
	require.Equal(t, []string{"owned", "terms"}, phrase.Terms)
}

func TestParseFTSQueryConcurrentUse(t *testing.T) {
	pipeline := newFTSStandardTestPipeline(t)
	const query = `+Vector -slow "Machine Learning" OR database`
	wantNode, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
	require.NoError(t, err)

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
		require.NoError(t, err)
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
		require.False(t, len(tokens) == 0,
			"missing EOF")
		require.Equal(t, FTSQueryTokenEOF, tokens[len(tokens)-1].Type,
			"missing EOF")

		previousEnd := uint32(0)
		for _, token := range tokens {
			require.True(t, token.Offset >= previousEnd)
			require.True(t, token.End >= token.Offset)
			require.True(t, uint64(token.End) <= uint64(len(query)))

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
		require.NotNil(t, node,
			"successful parse returned an empty AST")
		require.False(t, node.String() == "",
			"successful parse returned an empty AST")

		assertValidFTSQueryAST(t, node, 0)
	})
}

func BenchmarkParseFTSQuery(b *testing.B) {
	pipeline := newFTSStandardTestPipeline(b)
	query := `+(vector database OR "nearest neighbor") AND retrieval NOT "slow scan"`
	b.ReportAllocs()
	b.SetBytes(int64(len(query)))
	for b.Loop() {
		{
			_, err := ParseFTSQuery(context.Background(), query, pipeline, FTSDefaultOperatorOR)
			if err != nil {
				require.NoError(b, err)
			}
		}
	}
}

func newFTSStandardTestPipeline(t testing.TB) *tokenizer.FTSTokenizerPipeline {
	t.Helper()
	standardTokenizer, err := tokenizer.NewStandardTokenizer(tokenizer.DefaultStandardTokenizerOptions())
	require.NoError(t, err)

	pipeline, err := tokenizer.NewFTSTokenizerPipeline(standardTokenizer, tokenizer.NewLowercaseTokenFilter())
	require.NoError(t, err)

	return pipeline
}

func assertValidFTSQueryAST(t testing.TB, node FTSQueryNode, depth int) {
	t.Helper()
	require.NotNil(t, node)
	require.True(t, depth <= MaxFTSQueryDepth+2)

	modifier := node.Modifier()
	require.True(t, modifier.Boost == 1)

	switch typed := node.(type) {
	case *FTSTermQueryNode:
	case *FTSPhraseQueryNode:
		require.NotNil(t, typed.Terms,
			"phrase terms are nil")

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
		require.FailNowf(t, "unknown AST node", "%T", node)
	}
}
