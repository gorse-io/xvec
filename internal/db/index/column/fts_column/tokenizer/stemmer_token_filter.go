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
	"errors"
	"fmt"
	"strings"

	snowball "github.com/gorse-io/xvec/thirdparty/snowball"
	"github.com/gorse-io/xvec/thirdparty/snowball/stemmers"
)

// ErrInvalidStemmerOptions identifies an unknown Snowball language or alias.
var ErrInvalidStemmerOptions = errors.New("core: invalid stemmer options")

// StemmerTokenFilterOptions configures a Snowball language. An empty language
// selects the baseline default, english.
type StemmerTokenFilterOptions struct {
	Language string
}

// DefaultStemmerTokenFilterOptions returns the baseline English settings.
func DefaultStemmerTokenFilterOptions() StemmerTokenFilterOptions {
	return StemmerTokenFilterOptions{Language: "english"}
}

// Validate checks the case-sensitive libstemmer language and alias table.
func (o StemmerTokenFilterOptions) Validate() error {
	language := o.Language
	if language == "" {
		language = "english"
	}
	if _, found := stemmers.Lookup(language); !found {
		return fmt.Errorf("%w: unknown Language %q", ErrInvalidStemmerOptions, o.Language)
	}
	return nil
}

// SupportedStemmerLanguages returns all 115 case-sensitive libstemmer names
// and aliases in lexical order.
func SupportedStemmerLanguages() []string { return stemmers.Languages() }

// StemmerTokenFilter applies one generated Snowball 3.1.1 algorithm. It is
// immutable and safe for concurrent use.
type StemmerTokenFilter struct {
	language string
	stem     stemmers.Func
}

// NewStemmerTokenFilter constructs a validated stemmer filter.
func NewStemmerTokenFilter(options StemmerTokenFilterOptions) (*StemmerTokenFilter, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}
	language := options.Language
	if language == "" {
		language = "english"
	}
	stem, _ := stemmers.Lookup(language)
	return &StemmerTokenFilter{language: language, stem: stem}, nil
}

func (*StemmerTokenFilter) Name() string { return "stemmer" }

// Language returns the configured canonical name or alias.
func (f *StemmerTokenFilter) Language() string { return f.language }

// Filter returns an owned token slice with stemmed text and unchanged order,
// offsets, and positions. Empty tokens are retained, matching libstemmer.
func (f *StemmerTokenFilter) Filter(ctx context.Context, tokens []Token) ([]Token, error) {
	if ctx == nil {
		return nil, errors.New("core: nil token filter context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([]Token, len(tokens))
	environment := snowball.NewEnv("")
	for index, token := range tokens {
		for offset := 0; offset < len(token.Text); offset += 4096 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		environment.SetCurrent(token.Text)
		f.stem(environment)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result[index] = token
		result[index].Text = strings.Clone(environment.Current())
	}
	return result, nil
}

var _ TokenFilter = (*StemmerTokenFilter)(nil)
