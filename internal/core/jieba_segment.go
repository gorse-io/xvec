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
	"math"
	"strings"
)

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
