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
	"errors"
	"fmt"
	"reflect"
)

// ErrInvalidFTSAnalyzer identifies a nil or incomplete query/index analysis
// pipeline.
var ErrInvalidFTSAnalyzer = errors.New("core: invalid FTS analyzer")

// FTSAnalyzer applies the same tokenization and filtering rules to indexed
// text and query terms. Implementations must be safe for concurrent calls if
// they are shared between collections.
type FTSAnalyzer interface {
	Analyze(ctx context.Context, text string) ([]Token, error)
}

// FTSAnalyzerFunc adapts a function to FTSAnalyzer.
type FTSAnalyzerFunc func(context.Context, string) ([]Token, error)

// Analyze invokes f.
func (f FTSAnalyzerFunc) Analyze(ctx context.Context, text string) ([]Token, error) {
	if f == nil {
		return nil, fmt.Errorf("%w: nil analyzer function", ErrInvalidFTSAnalyzer)
	}
	return f(ctx, text)
}

// FTSTokenizerPipeline applies one tokenizer followed by zero or more filters.
// The pipeline snapshots the filter list and is safe for concurrent use when
// its tokenizer and filters are safe for concurrent use.
type FTSTokenizerPipeline struct {
	tokenizer Tokenizer
	filters   []TokenFilter
}

// NewFTSTokenizerPipeline constructs a validated analysis pipeline.
func NewFTSTokenizerPipeline(tokenizer Tokenizer, filters ...TokenFilter) (*FTSTokenizerPipeline, error) {
	if ftsNilInterface(tokenizer) {
		return nil, fmt.Errorf("%w: tokenizer is required", ErrInvalidFTSAnalyzer)
	}
	ownedFilters := append([]TokenFilter(nil), filters...)
	for index, filter := range ownedFilters {
		if ftsNilInterface(filter) {
			return nil, fmt.Errorf("%w: filter %d is nil", ErrInvalidFTSAnalyzer, index)
		}
	}
	return &FTSTokenizerPipeline{tokenizer: tokenizer, filters: ownedFilters}, nil
}

// TokenizerName returns the configured tokenizer name.
func (p *FTSTokenizerPipeline) TokenizerName() string {
	if p == nil || ftsNilInterface(p.tokenizer) {
		return ""
	}
	return p.tokenizer.Name()
}

// FilterNames returns an owned ordered filter-name slice.
func (p *FTSTokenizerPipeline) FilterNames() []string {
	if p == nil {
		return []string{}
	}
	names := make([]string, len(p.filters))
	for index, filter := range p.filters {
		names[index] = filter.Name()
	}
	return names
}

// Analyze tokenizes text and applies each filter in declaration order.
func (p *FTSTokenizerPipeline) Analyze(ctx context.Context, text string) ([]Token, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSAnalyzer)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if p == nil || ftsNilInterface(p.tokenizer) {
		return nil, fmt.Errorf("%w: tokenizer is required", ErrInvalidFTSAnalyzer)
	}
	tokens, err := p.tokenizer.Tokenize(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("fts tokenizer %q: %w", p.tokenizer.Name(), err)
	}
	for _, filter := range p.filters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tokens, err = filter.Filter(ctx, tokens)
		if err != nil {
			return nil, fmt.Errorf("fts token filter %q: %w", filter.Name(), err)
		}
	}
	return tokens, nil
}

func ftsNilInterface(value any) bool {
	if value == nil {
		return true
	}
	reflection := reflect.ValueOf(value)
	switch reflection.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflection.IsNil()
	default:
		return false
	}
}

var (
	_ FTSAnalyzer = FTSAnalyzerFunc(nil)
	_ FTSAnalyzer = (*FTSTokenizerPipeline)(nil)
)
