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

package sql

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLikePatternBaselineCasesAndClassification(t *testing.T) {
	tests := []struct {
		pattern string
		mode    LikeMode
		literal string
		matches []string
		misses  []string
	}{
		{pattern: "user-22", mode: LikeExact, literal: "user-22", matches: []string{"user-22"}, misses: []string{"xuser-22", "user-220"}},
		{pattern: "%", mode: LikeAny, matches: []string{"", "anything", "line\nbreak"}},
		{pattern: "user-22%", mode: LikePrefix, literal: "user-22", matches: []string{"user-22", "user-220"}, misses: []string{"xuser-22"}},
		{pattern: "%ser-22", mode: LikeSuffix, literal: "ser-22", matches: []string{"user-22", "ser-22"}, misses: []string{"user-220"}},
		{pattern: "%ser%", mode: LikeContains, literal: "ser", matches: []string{"user-22", "ser"}, misses: []string{"use-22"}},
		{pattern: "user%2", mode: LikeGeneral, matches: []string{"user2", "user-22", "user---2"}, misses: []string{"xuser2", "user20"}},
		{pattern: "user-_2", mode: LikeGeneral, matches: []string{"user-22", "user-_2", "user-界2"}, misses: []string{"user-2", "user-222"}},
		{pattern: `user-\%%`, mode: LikePrefix, literal: "user-%", matches: []string{"user-%22"}, misses: []string{"user-22"}},
		{pattern: `user-\_%`, mode: LikePrefix, literal: "user-_", matches: []string{"user-_22"}, misses: []string{"user-22"}},
		{pattern: `a\\b`, mode: LikeExact, literal: `a\b`, matches: []string{`a\b`}, misses: []string{"ab"}},
	}
	for _, testCase := range tests {
		t.Run(testCase.pattern, func(t *testing.T) {
			pattern, err := CompileLike(testCase.pattern)
			require.NoError(t, err)
			require.Equal(t, testCase.pattern, pattern.Raw())
			require.Equal(t, testCase.mode, pattern.Mode())
			require.Equal(t, testCase.literal, pattern.Literal())

			for _, value := range testCase.matches {
				assert.True(t, pattern.Match(value))
			}
			for _, value := range testCase.misses {
				assert.False(t, pattern.Match(value))
			}
		})
	}
}

func TestLikePatternCollapsesPercentsAndRejectsInvalidUTF8(t *testing.T) {
	pattern, err := CompileLike("%%needle%%")
	require.NoError(t, err)
	require.Equal(t, LikeContains, pattern.Mode())
	require.True(t, pattern.Literal() == "needle")
	require.True(t, pattern.Match("a needle here"))
	{
		_, err := CompileLike(string([]byte{0xff}))
		require.Error(t, err,
			"invalid UTF-8 pattern succeeded")
	}
	require.False(t, (*LikePattern)(nil).Match("anything"),
		"nil pattern matched")
}

func FuzzLikePattern(f *testing.F) {
	for _, seed := range []struct{ pattern, value string }{
		{"%", "anything"}, {"user-_2", "user-22"}, {`user-\%%`, "user-%22"},
		{"%界_", "边界a"}, {"trailing\\", "trailing\\"},
	} {
		f.Add(seed.pattern, seed.value)
	}
	f.Fuzz(func(t *testing.T, pattern, value string) {
		compiled, err := CompileLike(pattern)
		if err != nil {
			return
		}
		first := compiled.Match(value)
		require.Equal(t, first, compiled.Match(value),
			"LIKE result is not deterministic")
	})
}
