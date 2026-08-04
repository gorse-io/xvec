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

package sql

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLexFilterKeywordsLiteralsCommentsAndPositions(t *testing.T) {
	input := "Name >= -1.5e+2 and tag='A\\'b'\nOR score < = 3 -- ignored\n/* block */ active = TRUE"
	tokens, err := Lex(input)
	require.NoError(t, err)

	wantKinds := []TokenKind{
		TokenIdentifier, TokenGreaterEqual, TokenFloat, TokenAnd,
		TokenIdentifier, TokenEqual, TokenString, TokenOr,
		TokenIdentifier, TokenLess, TokenEqual, TokenInteger,
		TokenIdentifier, TokenEqual, TokenTrue, TokenEOF,
	}
	gotKinds := make([]TokenKind, len(tokens))
	for index := range tokens {
		gotKinds[index] = tokens[index].Kind
	}
	require.Equal(t, wantKinds, gotKinds)
	require.True(t, tokens[0].Text == "Name")
	require.True(t, tokens[3].Text == "and")
	require.True(t, tokens[6].Text == `'A\'b'`)
	{
		position := tokens[7].Span.Start
		require.True(t, position.Line == 2)
		require.True(t, position.Column == 1)
		require.True(t, position.Offset == 31)
	}
	{
		position := tokens[12].Span.Start
		require.True(t, position.Line == 3)
		require.True(t, position.Column == 13)
	}
}

func TestLexFilterLongestTokenRules(t *testing.T) {
	tokens, err := Lex("1-dash_score_field=1 123abc=2 -name=3 n=1F e=1E+3 f=.5")
	require.NoError(t, err)

	want := []struct {
		kind TokenKind
		text string
	}{
		{TokenIdentifier, "1-dash_score_field"}, {TokenEqual, "="}, {TokenInteger, "1"},
		{TokenIdentifier, "123abc"}, {TokenEqual, "="}, {TokenInteger, "2"},
		{TokenIdentifier, "-name"}, {TokenEqual, "="}, {TokenInteger, "3"},
		{TokenIdentifier, "n"}, {TokenEqual, "="}, {TokenFloat, "1F"},
		{TokenIdentifier, "e"}, {TokenEqual, "="}, {TokenFloat, "1E+3"},
		{TokenIdentifier, "f"}, {TokenEqual, "="}, {TokenFloat, ".5"},
		{TokenEOF, ""},
	}
	require.Len(t, tokens, len(want))

	for index := range want {
		require.Equal(t, want[index].kind, tokens[index].Kind)
		require.Equal(t, want[index].text, tokens[index].Text)
	}
}

func TestLexFilterErrorsCarryExactPosition(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		line   int
		column int
		offset int
	}{
		{name: "character", input: "a=1\n&", line: 2, column: 1, offset: 4},
		{name: "quote", input: "a = 'open", line: 1, column: 5, offset: 4},
		{name: "comment", input: "a=1 /* open", line: 1, column: 5, offset: 4},
		{name: "UTF-8", input: string([]byte{'a', '=', 0xff}), line: 1, column: 3, offset: 2},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := Lex(testCase.input)
			var parseErr *ParseError
			require.ErrorAs(t, err, &parseErr)

			position := parseErr.Position
			require.Equal(t, testCase.line, position.Line)
			require.Equal(t, testCase.column, position.Column)
			require.Equal(t, testCase.offset, position.Offset)
		})
	}
	{
		_, err := Lex(strings.Repeat("a", MaxFilterBytes+1))
		require.Error(t, err,
			"oversized filter succeeded")
	}
}
