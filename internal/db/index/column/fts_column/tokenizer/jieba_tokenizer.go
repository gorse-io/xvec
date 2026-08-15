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
	"math"
	"os"
	"path/filepath"
	"sync/atomic"

	"github.com/gorse-io/xvec/thirdparty/cppjieba"
)

const (
	jiebaDictionaryFile = "jieba.dict.utf8"
	jiebaHMMModelFile   = "hmm_model.utf8"
)

// JiebaCutMode selects the baseline cppjieba segmentation algorithm.
type JiebaCutMode = cppjieba.CutMode

const (
	JiebaCutModeSearch = cppjieba.CutModeSearch
	JiebaCutModeMix    = cppjieba.CutModeMix
	JiebaCutModeFull   = cppjieba.CutModeFull
	JiebaCutModeHMM    = cppjieba.CutModeHMM
)

var (
	// ErrInvalidJiebaTokenizerOptions identifies invalid construction options.
	ErrInvalidJiebaTokenizerOptions = errors.New("core: invalid jieba tokenizer options")
	// ErrInvalidJiebaUTF8 identifies input that cppjieba's pinned decoder
	// cannot decode as one complete sequence.
	ErrInvalidJiebaUTF8 = cppjieba.ErrInvalidUTF8

	defaultJiebaDictionaryDirectory atomic.Pointer[string]
)

// JiebaTokenizerOptions configures dictionary resolution and cut mode. An
// empty CutMode means search. An empty DictDir resolves from
// ZVEC_JIEBA_DICT_DIR and then DefaultJiebaDictDir.
type JiebaTokenizerOptions struct {
	DictDir      string
	UserDictPath string
	CutMode      JiebaCutMode
}

// DefaultJiebaTokenizerOptions returns the baseline search-mode defaults.
func DefaultJiebaTokenizerOptions() JiebaTokenizerOptions {
	return JiebaTokenizerOptions{CutMode: JiebaCutModeSearch}
}

// SetDefaultJiebaDictDir sets the process-wide fallback dictionary directory.
// Explicit options and ZVEC_JIEBA_DICT_DIR have higher priority.
func SetDefaultJiebaDictDir(path string) {
	value := path
	defaultJiebaDictionaryDirectory.Store(&value)
}

// DefaultJiebaDictDir returns the process-wide fallback directory.
func DefaultJiebaDictDir() string {
	value := defaultJiebaDictionaryDirectory.Load()
	if value == nil {
		return ""
	}
	return *value
}

// Validate checks mode and the availability of required resource paths. File
// contents are validated by NewJiebaTokenizer.
func (o JiebaTokenizerOptions) Validate() error {
	mode := o.CutMode
	if mode == "" {
		mode = JiebaCutModeSearch
	}
	switch mode {
	case JiebaCutModeSearch, JiebaCutModeMix, JiebaCutModeFull, JiebaCutModeHMM:
	default:
		return fmt.Errorf("%w: unknown CutMode %q", ErrInvalidJiebaTokenizerOptions, o.CutMode)
	}
	if resolveJiebaDictDir(o.DictDir) == "" {
		return fmt.Errorf("%w: dictionary directory is not configured", ErrInvalidJiebaTokenizerOptions)
	}
	return nil
}

// JiebaTokenizer adapts the vendored cppjieba implementation to Tokenizer.
type JiebaTokenizer struct {
	mode      JiebaCutMode
	dictDir   string
	segmenter *cppjieba.Segmenter
}

// NewJiebaTokenizer loads and validates the resources required by the chosen
// mode. Search and mix require both files, full requires only the dictionary,
// and HMM requires only the model.
func NewJiebaTokenizer(ctx context.Context, options JiebaTokenizerOptions) (*JiebaTokenizer, error) {
	if ctx == nil {
		return nil, errors.New("core: nil tokenizer context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	mode := options.CutMode
	if mode == "" {
		mode = JiebaCutModeSearch
	}
	directory := resolveJiebaDictDir(options.DictDir)
	segmenter, err := cppjieba.New(ctx, cppjieba.Options{
		DictionaryPath: filepath.Join(directory, jiebaDictionaryFile),
		HMMModelPath:   filepath.Join(directory, jiebaHMMModelFile),
		UserDictPath:   options.UserDictPath,
		CutMode:        mode,
	})
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidJiebaTokenizerOptions, err)
	}
	return &JiebaTokenizer{mode: mode, dictDir: directory, segmenter: segmenter}, nil
}

func (*JiebaTokenizer) Name() string { return "jieba" }

func (t *JiebaTokenizer) CutMode() JiebaCutMode { return t.mode }

func (t *JiebaTokenizer) DictDir() string { return t.dictDir }

func resolveJiebaDictDir(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if environment := os.Getenv("ZVEC_JIEBA_DICT_DIR"); environment != "" {
		return environment
	}
	return DefaultJiebaDictDir()
}

// Tokenize converts cppjieba words to FTS tokens with contiguous positions.
func (t *JiebaTokenizer) Tokenize(ctx context.Context, text string) ([]Token, error) {
	if ctx == nil {
		return nil, errors.New("core: nil tokenizer context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(len(text)) > math.MaxUint32 {
		return nil, ErrTokenizerInputTooLarge
	}
	words, err := t.segmenter.Cut(ctx, text)
	if err != nil {
		return nil, err
	}
	tokens := make([]Token, len(words))
	for index, word := range words {
		tokens[index] = Token{Text: word.Text, Offset: word.Offset, Position: uint32(index)}
	}
	return tokens, nil
}

var _ Tokenizer = (*JiebaTokenizer)(nil)
