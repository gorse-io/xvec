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
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

const (
	jiebaDictionaryFile = "jieba.dict.utf8"
	jiebaHMMModelFile   = "hmm_model.utf8"
	jiebaMaxWordLength  = 512
	jiebaMinDouble      = -3.14e100
)

// JiebaCutMode selects the baseline cppjieba segmentation algorithm.
type JiebaCutMode string

const (
	JiebaCutModeSearch JiebaCutMode = "search"
	JiebaCutModeMix    JiebaCutMode = "mix"
	JiebaCutModeFull   JiebaCutMode = "full"
	JiebaCutModeHMM    JiebaCutMode = "hmm"
)

var (
	// ErrInvalidJiebaTokenizerOptions identifies invalid construction options.
	ErrInvalidJiebaTokenizerOptions = errors.New("core: invalid jieba tokenizer options")
	// ErrInvalidJiebaUTF8 identifies input that cppjieba's pinned decoder
	// cannot decode as one complete sequence.
	ErrInvalidJiebaUTF8 = errors.New("core: invalid UTF-8 for jieba tokenizer")

	defaultJiebaDictionaryDirectory atomic.Pointer[string]
	jiebaDictionaryCache            = struct {
		sync.Mutex
		items map[string]*jiebaDictionary
	}{items: make(map[string]*jiebaDictionary)}
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
	if !mode.valid() {
		return fmt.Errorf("%w: unknown CutMode %q", ErrInvalidJiebaTokenizerOptions, o.CutMode)
	}
	if resolveJiebaDictDir(o.DictDir) == "" {
		return fmt.Errorf("%w: dictionary directory is not configured", ErrInvalidJiebaTokenizerOptions)
	}
	return nil
}

func (m JiebaCutMode) valid() bool {
	switch m {
	case JiebaCutModeSearch, JiebaCutModeMix, JiebaCutModeFull, JiebaCutModeHMM:
		return true
	default:
		return false
	}
}

// JiebaTokenizer is an immutable, concurrency-safe pure-Go tokenizer using
// the baseline cppjieba dictionary and HMM model formats.
type JiebaTokenizer struct {
	mode       JiebaCutMode
	dictDir    string
	dictionary *jiebaDictionary
	model      *jiebaHMMModel
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
	tokenizer := &JiebaTokenizer{mode: mode, dictDir: directory}

	if mode != JiebaCutModeHMM {
		dictionaryPath := filepath.Join(directory, jiebaDictionaryFile)
		dictionary, err := loadCachedJiebaDictionary(ctx, dictionaryPath)
		if err != nil {
			return nil, fmt.Errorf("%w: load %s: %w", ErrInvalidJiebaTokenizerOptions, dictionaryPath, err)
		}
		if options.UserDictPath != "" {
			dictionary = dictionary.clone()
			if err := dictionary.loadUserDictionaries(ctx, options.UserDictPath); err != nil {
				return nil, fmt.Errorf("%w: load user dictionary: %w", ErrInvalidJiebaTokenizerOptions, err)
			}
		}
		tokenizer.dictionary = dictionary
	}
	if mode != JiebaCutModeFull {
		modelPath := filepath.Join(directory, jiebaHMMModelFile)
		model, err := loadJiebaHMMModel(ctx, modelPath)
		if err != nil {
			return nil, fmt.Errorf("%w: load %s: %w", ErrInvalidJiebaTokenizerOptions, modelPath, err)
		}
		tokenizer.model = model
	}
	return tokenizer, nil
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

type jiebaDictionaryEntry struct {
	weight float64
	length int
}

type jiebaDictionary struct {
	entries       map[string]jiebaDictionaryEntry
	prefixes      map[string]struct{}
	minWeight     float64
	medianWeight  float64
	maxWordLength int
	userSingles   map[uint32]struct{}
	frequencySum  float64
}

func loadCachedJiebaDictionary(ctx context.Context, path string) (*jiebaDictionary, error) {
	jiebaDictionaryCache.Lock()
	if dictionary := jiebaDictionaryCache.items[path]; dictionary != nil {
		jiebaDictionaryCache.Unlock()
		return dictionary, nil
	}
	jiebaDictionaryCache.Unlock()
	dictionary, err := loadJiebaDictionary(ctx, path)
	if err != nil {
		return nil, err
	}
	jiebaDictionaryCache.Lock()
	defer jiebaDictionaryCache.Unlock()
	if existing := jiebaDictionaryCache.items[path]; existing != nil {
		return existing, nil
	}
	jiebaDictionaryCache.items[path] = dictionary
	return dictionary, nil
}

func loadJiebaDictionary(ctx context.Context, path string) (*jiebaDictionary, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	type sourceEntry struct {
		key       string
		frequency float64
		length    int
	}
	sources := make([]sourceEntry, 0, 64<<10)
	frequencySum := 0.0
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), 2<<20)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		if lineNumber&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		fields := strings.Fields(scanner.Text())
		if len(fields) != 3 {
			return nil, fmt.Errorf("line %d: expected word, frequency, and tag", lineNumber)
		}
		codepoints, ok := decodeJiebaDictionaryWord(fields[0])
		if !ok || len(codepoints) == 0 {
			return nil, fmt.Errorf("line %d: invalid dictionary word", lineNumber)
		}
		frequency, err := strconv.ParseFloat(fields[1], 64)
		if err != nil || frequency <= 0 || math.IsNaN(frequency) || math.IsInf(frequency, 0) {
			return nil, fmt.Errorf("line %d: invalid positive frequency %q", lineNumber, fields[1])
		}
		sources = append(sources, sourceEntry{key: jiebaCodepointKey(codepoints), frequency: frequency, length: len(codepoints)})
		frequencySum += frequency
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(sources) == 0 || frequencySum <= 0 || math.IsInf(frequencySum, 0) {
		return nil, errors.New("dictionary is empty or has an invalid frequency sum")
	}

	entries := make(map[string]jiebaDictionaryEntry, len(sources))
	prefixes := make(map[string]struct{}, len(sources))
	weights := make([]float64, len(sources))
	maxWordLength := 0
	for index, source := range sources {
		weight := math.Log(source.frequency / frequencySum)
		weights[index] = weight
		entries[source.key] = jiebaDictionaryEntry{weight: weight, length: source.length}
		addJiebaPrefixes(prefixes, source.key, source.length)
		if source.length > maxWordLength {
			maxWordLength = source.length
		}
	}
	sort.Float64s(weights)
	if maxWordLength > jiebaMaxWordLength {
		maxWordLength = jiebaMaxWordLength
	}
	return &jiebaDictionary{
		entries:       entries,
		prefixes:      prefixes,
		minWeight:     weights[0],
		medianWeight:  weights[len(weights)/2],
		maxWordLength: maxWordLength,
		userSingles:   make(map[uint32]struct{}),
		frequencySum:  frequencySum,
	}, nil
}

func (d *jiebaDictionary) clone() *jiebaDictionary {
	entries := make(map[string]jiebaDictionaryEntry, len(d.entries))
	for key, entry := range d.entries {
		entries[key] = entry
	}
	prefixes := make(map[string]struct{}, len(d.prefixes))
	for key := range d.prefixes {
		prefixes[key] = struct{}{}
	}
	return &jiebaDictionary{
		entries: entries, prefixes: prefixes, minWeight: d.minWeight, medianWeight: d.medianWeight,
		maxWordLength: d.maxWordLength, userSingles: make(map[uint32]struct{}),
		frequencySum: d.frequencySum,
	}
}

func (d *jiebaDictionary) loadUserDictionaries(ctx context.Context, paths string) error {
	items := strings.FieldsFunc(paths, func(value rune) bool { return value == '|' || value == ';' })
	if len(items) == 0 {
		return errors.New("user dictionary path is empty")
	}
	for _, path := range items {
		if err := d.loadUserDictionary(ctx, path); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	return nil
}

func (d *jiebaDictionary) loadUserDictionary(ctx context.Context, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), 2<<20)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		if lineNumber&1023 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		if len(fields) > 3 {
			return fmt.Errorf("line %d: expected word, optional tag, or frequency and tag", lineNumber)
		}
		codepoints, ok := decodeJiebaDictionaryWord(fields[0])
		if !ok || len(codepoints) == 0 {
			return fmt.Errorf("line %d: invalid dictionary word", lineNumber)
		}
		weight := d.medianWeight
		if len(fields) == 3 {
			frequency, err := strconv.ParseInt(fields[1], 10, 32)
			if err != nil || frequency < 0 {
				return fmt.Errorf("line %d: invalid frequency %q", lineNumber, fields[1])
			}
			if frequency != 0 {
				weight = math.Log(float64(frequency) / d.frequencySum)
			}
		}
		key := jiebaCodepointKey(codepoints)
		d.entries[key] = jiebaDictionaryEntry{weight: weight, length: len(codepoints)}
		addJiebaPrefixes(d.prefixes, key, len(codepoints))
		if len(codepoints) == 1 {
			d.userSingles[codepoints[0]] = struct{}{}
		}
		if len(codepoints) > d.maxWordLength {
			d.maxWordLength = len(codepoints)
			if d.maxWordLength > jiebaMaxWordLength {
				d.maxWordLength = jiebaMaxWordLength
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return ctx.Err()
}

func (d *jiebaDictionary) find(runes []jiebaRune, start, end int) (jiebaDictionaryEntry, bool) {
	entry, ok := d.entries[jiebaRuneKey(runes, start, end)]
	return entry, ok
}

func (d *jiebaDictionary) lookup(runes []jiebaRune, start, end int) (jiebaDictionaryEntry, bool, bool) {
	key := jiebaRuneKey(runes, start, end)
	entry, found := d.entries[key]
	_, prefix := d.prefixes[key]
	return entry, found, prefix
}

func addJiebaPrefixes(prefixes map[string]struct{}, key string, length int) {
	limit := length
	if limit > jiebaMaxWordLength {
		limit = jiebaMaxWordLength
	}
	for prefixLength := 1; prefixLength < limit; prefixLength++ {
		prefixes[key[:prefixLength*4]] = struct{}{}
	}
}

func (d *jiebaDictionary) isUserSingle(value uint32) bool {
	_, exists := d.userSingles[value]
	return exists
}

type jiebaHMMModel struct {
	start      [4]float64
	transition [4][4]float64
	emission   [4]map[uint32]float64
}

func loadJiebaHMMModel(ctx context.Context, path string) (*jiebaHMMModel, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), 2<<20)
	nextDataLine := func() (string, error) {
		for scanner.Scan() {
			if err := ctx.Err(); err != nil {
				return "", err
			}
			line := strings.TrimSpace(scanner.Text())
			if line != "" && !strings.HasPrefix(line, "#") {
				return line, nil
			}
		}
		if err := scanner.Err(); err != nil {
			return "", err
		}
		return "", fmt.Errorf("unexpected end of HMM model: %w", io.ErrUnexpectedEOF)
	}
	model := &jiebaHMMModel{}
	line, err := nextDataLine()
	if err != nil {
		return nil, err
	}
	if err := parseJiebaProbabilityRow(line, model.start[:]); err != nil {
		return nil, fmt.Errorf("start probabilities: %w", err)
	}
	for state := 0; state < 4; state++ {
		line, err = nextDataLine()
		if err != nil {
			return nil, err
		}
		if err := parseJiebaProbabilityRow(line, model.transition[state][:]); err != nil {
			return nil, fmt.Errorf("transition row %d: %w", state, err)
		}
	}
	for state := 0; state < 4; state++ {
		line, err = nextDataLine()
		if err != nil {
			return nil, err
		}
		emissions, err := parseJiebaEmissions(line)
		if err != nil {
			return nil, fmt.Errorf("emission row %d: %w", state, err)
		}
		model.emission[state] = emissions
	}
	return model, nil
}

func parseJiebaProbabilityRow(line string, target []float64) error {
	fields := strings.Fields(line)
	if len(fields) != len(target) {
		return fmt.Errorf("got %d values, want %d", len(fields), len(target))
	}
	for index, field := range fields {
		value, err := strconv.ParseFloat(field, 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("invalid probability %q", field)
		}
		target[index] = value
	}
	return nil
}

func parseJiebaEmissions(line string) (map[uint32]float64, error) {
	items := strings.Split(line, ",")
	emissions := make(map[uint32]float64, len(items))
	for _, item := range items {
		parts := strings.Split(item, ":")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid emission %q", item)
		}
		codepoints, ok := decodeJiebaDictionaryWord(parts[0])
		if !ok || len(codepoints) != 1 {
			return nil, fmt.Errorf("invalid emission codepoint %q", parts[0])
		}
		probability, err := strconv.ParseFloat(parts[1], 64)
		if err != nil || math.IsNaN(probability) || math.IsInf(probability, 0) {
			return nil, fmt.Errorf("invalid emission probability %q", parts[1])
		}
		emissions[codepoints[0]] = probability
	}
	return emissions, nil
}

func (m *jiebaHMMModel) emit(state int, value uint32) float64 {
	if probability, exists := m.emission[state][value]; exists {
		return probability
	}
	return jiebaMinDouble
}

func decodeJiebaDictionaryWord(word string) ([]uint32, bool) {
	runes, ok := decodeJiebaUTF8(word)
	if !ok {
		return nil, false
	}
	values := make([]uint32, len(runes))
	for index := range runes {
		values[index] = runes[index].value
	}
	return values, true
}

func jiebaCodepointKey(values []uint32) string {
	buffer := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], value)
	}
	return string(buffer)
}

func jiebaRuneKey(runes []jiebaRune, start, end int) string {
	buffer := make([]byte, (end-start)*4)
	for index := start; index < end; index++ {
		binary.LittleEndian.PutUint32(buffer[(index-start)*4:], runes[index].value)
	}
	return string(buffer)
}

var _ Tokenizer = (*JiebaTokenizer)(nil)

type jiebaRune struct {
	value uint32
	start uint32
	end   uint32
}

type jiebaWordRange struct {
	start int
	end   int // inclusive
}

type jiebaInputRange struct {
	start int
	end   int // exclusive
}

// Tokenize segments text with the immutable resources loaded at construction.
// Offsets are original byte offsets and positions are output sequence numbers,
// including overlapping search-mode subwords.
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
	if text == "" {
		return []Token{}, nil
	}
	runes, valid, err := decodeJiebaUTF8Context(ctx, text)
	if err != nil {
		return nil, err
	}
	if !valid {
		return nil, ErrInvalidJiebaUTF8
	}
	ranges, err := splitJiebaInputRanges(ctx, runes)
	if err != nil {
		return nil, err
	}
	words := make([]jiebaWordRange, 0, len(runes))
	for _, inputRange := range ranges {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var cut []jiebaWordRange
		var err error
		switch t.mode {
		case JiebaCutModeSearch:
			cut, err = t.cutJiebaSearch(ctx, runes, inputRange.start, inputRange.end)
		case JiebaCutModeMix:
			cut, err = t.cutJiebaMix(ctx, runes, inputRange.start, inputRange.end)
		case JiebaCutModeFull:
			cut, err = t.cutJiebaFull(ctx, runes, inputRange.start, inputRange.end)
		case JiebaCutModeHMM:
			cut, err = t.cutJiebaHMM(ctx, runes, inputRange.start, inputRange.end)
		}
		if err != nil {
			return nil, err
		}
		words = append(words, cut...)
	}
	tokens := make([]Token, 0, len(words))
	for index, word := range words {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if word.start < 0 || word.end < word.start || word.end >= len(runes) {
			return nil, errors.New("core: jieba produced an invalid word range")
		}
		start := runes[word.start].start
		end := runes[word.end].end
		if start == end {
			continue
		}
		tokens = append(tokens, Token{
			Text: strings.Clone(text[start:end]), Offset: start, Position: uint32(len(tokens)),
		})
	}
	return tokens, nil
}

func decodeJiebaUTF8(text string) ([]jiebaRune, bool) {
	runes, valid, _ := decodeJiebaUTF8Context(context.Background(), text)
	return runes, valid
}

func decodeJiebaUTF8Context(ctx context.Context, text string) ([]jiebaRune, bool, error) {
	runes := make([]jiebaRune, 0, len(text)/2)
	for index := 0; index < len(text); {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, false, err
			}
		}
		lead := text[index]
		value := uint32(0)
		size := 0
		switch {
		case lead&0x80 == 0:
			value = uint32(lead & 0x7f)
			size = 1
		case lead <= 0xdf && index+1 < len(text):
			value = uint32(lead&0x1f)<<6 | uint32(text[index+1]&0x3f)
			size = 2
		case lead <= 0xef && index+2 < len(text):
			value = uint32(lead&0x0f)<<12 | uint32(text[index+1]&0x3f)<<6 | uint32(text[index+2]&0x3f)
			size = 3
		case lead <= 0xf7 && index+3 < len(text):
			value = uint32(lead&0x07)<<18 | uint32(text[index+1]&0x3f)<<12 |
				uint32(text[index+2]&0x3f)<<6 | uint32(text[index+3]&0x3f)
			size = 4
		default:
			return nil, false, nil
		}
		runes = append(runes, jiebaRune{value: value, start: uint32(index), end: uint32(index + size)})
		index += size
	}
	return runes, true, nil
}

func splitJiebaInputRanges(ctx context.Context, runes []jiebaRune) ([]jiebaInputRange, error) {
	ranges := make([]jiebaInputRange, 0)
	start := 0
	for index, item := range runes {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if !jiebaSpecialSeparator(item.value) {
			continue
		}
		if start < index {
			ranges = append(ranges, jiebaInputRange{start: start, end: index})
		}
		ranges = append(ranges, jiebaInputRange{start: index, end: index + 1})
		start = index + 1
	}
	if start < len(runes) {
		ranges = append(ranges, jiebaInputRange{start: start, end: len(runes)})
	}
	return ranges, nil
}

func jiebaSpecialSeparator(value uint32) bool {
	switch value {
	case ' ', '\t', '\n', 0xff0c, 0x3002:
		return true
	default:
		return false
	}
}

type jiebaDAGCandidate struct {
	end   int // absolute and inclusive
	entry jiebaDictionaryEntry
	found bool
}

func (d *jiebaDictionary) buildDAG(ctx context.Context, runes []jiebaRune, start, end int) ([][]jiebaDAGCandidate, error) {
	dag := make([][]jiebaDAGCandidate, end-start)
	work := 0
	for absolute := start; absolute < end; absolute++ {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		entry, found, canExtend := d.lookup(runes, absolute, absolute+1)
		work++
		dag[absolute-start] = append(dag[absolute-start], jiebaDAGCandidate{end: absolute, entry: entry, found: found})
		limit := end - absolute
		if limit > d.maxWordLength {
			limit = d.maxWordLength
		}
		for length := 2; length <= limit && canExtend; length++ {
			if work&4095 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			entry, found, canExtend = d.lookup(runes, absolute, absolute+length)
			work++
			if found {
				dag[absolute-start] = append(dag[absolute-start], jiebaDAGCandidate{end: absolute + length - 1, entry: entry, found: true})
			}
		}
	}
	return dag, nil
}

func (t *JiebaTokenizer) cutJiebaMP(ctx context.Context, runes []jiebaRune, start, end int) ([]jiebaWordRange, error) {
	dag, err := t.dictionary.buildDAG(ctx, runes, start, end)
	if err != nil {
		return nil, err
	}
	weights := make([]float64, len(dag))
	chosen := make([]jiebaDAGCandidate, len(dag))
	for local := len(dag) - 1; local >= 0; local-- {
		if local&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		best := jiebaMinDouble
		for _, candidate := range dag[local] {
			value := 0.0
			nextLocal := candidate.end - start + 1
			if nextLocal < len(dag) {
				value += weights[nextLocal]
			}
			if candidate.found {
				value += candidate.entry.weight
			} else {
				value += t.dictionary.minWeight
			}
			if value > best {
				best = value
				chosen[local] = candidate
			}
		}
		weights[local] = best
	}
	words := make([]jiebaWordRange, 0, len(dag))
	for local := 0; local < len(dag); {
		candidate := chosen[local]
		absolute := start + local
		wordEnd := absolute
		if candidate.found {
			wordEnd = absolute + candidate.entry.length - 1
		}
		words = append(words, jiebaWordRange{start: absolute, end: wordEnd})
		local = wordEnd - start + 1
	}
	return words, nil
}

func (t *JiebaTokenizer) cutJiebaFull(ctx context.Context, runes []jiebaRune, start, end int) ([]jiebaWordRange, error) {
	dag, err := t.dictionary.buildDAG(ctx, runes, start, end)
	if err != nil {
		return nil, err
	}
	words := make([]jiebaWordRange, 0, len(dag))
	maxIndex := 0
	for local, candidates := range dag {
		if local&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for _, candidate := range candidates {
			wordLength := 1
			if candidate.found {
				wordLength = candidate.entry.length
				if wordLength >= 2 || len(candidates) == 1 && maxIndex <= local {
					words = append(words, jiebaWordRange{start: start + local, end: candidate.end})
				}
			} else if len(candidates) == 1 && maxIndex <= local {
				words = append(words, jiebaWordRange{start: start + local, end: candidate.end})
			}
			if local+wordLength > maxIndex {
				maxIndex = local + wordLength
			}
		}
	}
	return words, nil
}

func (t *JiebaTokenizer) cutJiebaMix(ctx context.Context, runes []jiebaRune, start, end int) ([]jiebaWordRange, error) {
	mpWords, err := t.cutJiebaMP(ctx, runes, start, end)
	if err != nil {
		return nil, err
	}
	words := make([]jiebaWordRange, 0, len(mpWords))
	for index := 0; index < len(mpWords); index++ {
		word := mpWords[index]
		if word.start != word.end || t.dictionary.isUserSingle(runes[word.start].value) {
			words = append(words, word)
			continue
		}
		next := index
		for next < len(mpWords) && mpWords[next].start == mpWords[next].end && !t.dictionary.isUserSingle(runes[mpWords[next].start].value) {
			next++
		}
		hmmWords, err := t.cutJiebaHMM(ctx, runes, word.start, mpWords[next-1].end+1)
		if err != nil {
			return nil, err
		}
		words = append(words, hmmWords...)
		index = next - 1
	}
	return words, nil
}

func (t *JiebaTokenizer) cutJiebaSearch(ctx context.Context, runes []jiebaRune, start, end int) ([]jiebaWordRange, error) {
	mixed, err := t.cutJiebaMix(ctx, runes, start, end)
	if err != nil {
		return nil, err
	}
	words := make([]jiebaWordRange, 0, len(mixed))
	work := 0
	for _, word := range mixed {
		if work&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		length := word.end - word.start + 1
		if length > 2 {
			for offset := 0; offset+1 < length; offset++ {
				work++
				if work&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				if _, found := t.dictionary.find(runes, word.start+offset, word.start+offset+2); found {
					words = append(words, jiebaWordRange{start: word.start + offset, end: word.start + offset + 1})
				}
			}
		}
		if length > 3 {
			for offset := 0; offset+2 < length; offset++ {
				work++
				if work&4095 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				if _, found := t.dictionary.find(runes, word.start+offset, word.start+offset+3); found {
					words = append(words, jiebaWordRange{start: word.start + offset, end: word.start + offset + 2})
				}
			}
		}
		words = append(words, word)
	}
	return words, nil
}

func (t *JiebaTokenizer) cutJiebaHMM(ctx context.Context, runes []jiebaRune, start, end int) ([]jiebaWordRange, error) {
	words := make([]jiebaWordRange, 0, end-start)
	left := start
	for right := start; right < end; {
		if right&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if runes[right].value >= 0x80 {
			right++
			continue
		}
		if left != right {
			cut, err := t.cutJiebaHMMInternal(ctx, runes, left, right)
			if err != nil {
				return nil, err
			}
			words = append(words, cut...)
		}
		left = right
		next := jiebaSequentialLetterEnd(runes, left, end)
		if next == left {
			next = jiebaNumberEnd(runes, left, end)
		}
		if next == left {
			next++
		}
		words = append(words, jiebaWordRange{start: left, end: next - 1})
		left = next
		right = next
	}
	if left != end {
		cut, err := t.cutJiebaHMMInternal(ctx, runes, left, end)
		if err != nil {
			return nil, err
		}
		words = append(words, cut...)
	}
	return words, nil
}

func jiebaSequentialLetterEnd(runes []jiebaRune, start, end int) int {
	if start >= end || !jiebaASCIILetter(runes[start].value) {
		return start
	}
	index := start + 1
	for index < end && (jiebaASCIILetter(runes[index].value) || jiebaASCIIDigit(runes[index].value)) {
		index++
	}
	return jiebaDecimalSuffixEnd(runes, index, end)
}

func jiebaNumberEnd(runes []jiebaRune, start, end int) int {
	if start >= end || !jiebaASCIIDigit(runes[start].value) {
		return start
	}
	index := start + 1
	for index < end && (jiebaASCIILetter(runes[index].value) || jiebaASCIIDigit(runes[index].value)) {
		index++
	}
	return jiebaDecimalSuffixEnd(runes, index, end)
}

func jiebaDecimalSuffixEnd(runes []jiebaRune, index, end int) int {
	if index+1 < end && runes[index].value == '.' && jiebaASCIIDigit(runes[index+1].value) {
		index++
		for index < end && jiebaASCIIDigit(runes[index].value) {
			index++
		}
	}
	return index
}

func jiebaASCIILetter(value uint32) bool {
	return value >= 'a' && value <= 'z' || value >= 'A' && value <= 'Z'
}

func jiebaASCIIDigit(value uint32) bool { return value >= '0' && value <= '9' }

func (t *JiebaTokenizer) cutJiebaHMMInternal(ctx context.Context, runes []jiebaRune, start, end int) ([]jiebaWordRange, error) {
	length := end - start
	paths := make([]int, length*4)
	weights := make([]float64, length*4)
	for state := 0; state < 4; state++ {
		weights[state*length] = t.model.start[state] + t.model.emit(state, runes[start].value)
		paths[state*length] = -1
	}
	for offset := 1; offset < length; offset++ {
		if offset&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for state := 0; state < 4; state++ {
			current := offset + state*length
			weights[current] = jiebaMinDouble
			paths[current] = 1
			emission := t.model.emit(state, runes[start+offset].value)
			for previous := 0; previous < 4; previous++ {
				old := offset - 1 + previous*length
				candidate := weights[old] + t.model.transition[previous][state] + emission
				if candidate > weights[current] {
					weights[current] = candidate
					paths[current] = previous
				}
			}
		}
	}
	state := 3
	if weights[length-1+1*length] >= weights[length-1+3*length] {
		state = 1
	}
	statuses := make([]int, length)
	for offset := length - 1; offset >= 0; offset-- {
		statuses[offset] = state
		state = paths[offset+state*length]
	}
	words := make([]jiebaWordRange, 0, length)
	left := start
	for offset, status := range statuses {
		if status%2 == 1 {
			words = append(words, jiebaWordRange{start: left, end: start + offset})
			left = start + offset + 1
		}
	}
	return words, nil
}
