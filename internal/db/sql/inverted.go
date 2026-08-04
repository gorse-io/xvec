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
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/gorse-io/zvec/internal/ailego"
)

// InvertedStrategy identifies how an index search produced its candidate set.
type InvertedStrategy uint8

const (
	InvertedExact InvertedStrategy = iota + 1
	InvertedPostingUnion
	InvertedSortedRange
	InvertedTermScan
	InvertedPrefix
	InvertedSuffix
	InvertedPrefixSuffix
	InvertedNull
	InvertedArrayLength
)

func (s InvertedStrategy) String() string {
	switch s {
	case InvertedExact:
		return "EXACT"
	case InvertedPostingUnion:
		return "POSTING_UNION"
	case InvertedSortedRange:
		return "SORTED_RANGE"
	case InvertedTermScan:
		return "TERM_SCAN"
	case InvertedPrefix:
		return "PREFIX"
	case InvertedSuffix:
		return "SUFFIX"
	case InvertedPrefixSuffix:
		return "PREFIX_SUFFIX"
	case InvertedNull:
		return "NULL"
	case InvertedArrayLength:
		return "ARRAY_LENGTH"
	default:
		return "UNKNOWN"
	}
}

// InvertedResult is an immutable search result. Supported=false instructs the
// planner to use forward evaluation for that predicate.
type InvertedResult struct {
	Bitmap    *ailego.Bitmap
	Supported bool
	Strategy  InvertedStrategy
	Terms     int
}

// InvertedIndex stores exact postings for one scalar or homogeneous array
// field. Build calls are serialized; after Seal, searches are concurrent-safe.
type InvertedIndex struct {
	mu          sync.RWMutex
	field       Field
	sealed      bool
	rows        *ailego.Bitmap
	nulls       *ailego.Bitmap
	nonNull     *ailego.Bitmap
	postings    map[scalarKey]*ailego.Bitmap
	ordered     []scalarKey
	arrayLength map[uint32]*ailego.Bitmap
	lengths     []uint32
}

func NewInvertedIndex(field Field) (*InvertedIndex, error) {
	if !field.Filterable || !field.Indexed || !field.Kind.valid() {
		return nil, fmt.Errorf("sql: field %q is not a valid inverted-index field", field.Name)
	}
	return &InvertedIndex{
		field: field, rows: ailego.NewBitmap(0), nulls: ailego.NewBitmap(0), nonNull: ailego.NewBitmap(0),
		postings: make(map[scalarKey]*ailego.Bitmap), arrayLength: make(map[uint32]*ailego.Bitmap),
	}, nil
}

func (i *InvertedIndex) Field() Field {
	if i == nil {
		return Field{}
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.field
}

// Add indexes a snapshot-local row ordinal exactly once.
func (i *InvertedIndex) Add(row uint64, value Value) error {
	if i == nil {
		return fmt.Errorf("sql: nil inverted index")
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.sealed {
		return fmt.Errorf("sql: inverted index %q is sealed", i.field.Name)
	}
	if i.rows.Contains(row) {
		return fmt.Errorf("sql: inverted index %q row %d already exists", i.field.Name, row)
	}
	if value.kind != i.field.Kind || value.array != i.field.Array || !value.kind.valid() {
		return fmt.Errorf("sql: inverted index %q received %s", i.field.Name, value.describe())
	}
	if value.kind == ValueString && !value.null {
		if !value.array && !utf8.ValidString(value.text) {
			return fmt.Errorf("sql: inverted index %q received invalid UTF-8", i.field.Name)
		}
		for _, element := range value.elements {
			if !utf8.ValidString(element.text) {
				return fmt.Errorf("sql: inverted index %q received invalid UTF-8", i.field.Name)
			}
		}
	}
	i.rows.Set(row)
	if value.null {
		i.nulls.Set(row)
		return nil
	}
	i.nonNull.Set(row)
	if value.array {
		length := uint32(len(value.elements))
		posting := i.arrayLength[length]
		if posting == nil {
			posting = ailego.NewBitmap(0)
			i.arrayLength[length] = posting
		}
		posting.Set(row)
		seen := make(map[scalarKey]struct{}, len(value.elements))
		for _, element := range value.elements {
			key := keyFromValue(element)
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}
			i.addPosting(key, row)
		}
		return nil
	}
	i.addPosting(keyFromValue(value), row)
	return nil
}

func (i *InvertedIndex) addPosting(key scalarKey, row uint64) {
	posting := i.postings[key]
	if posting == nil {
		posting = ailego.NewBitmap(0)
		i.postings[key] = posting
	}
	posting.Set(row)
}

// Seal publishes sorted term and array-length dictionaries.
func (i *InvertedIndex) Seal() error {
	if i == nil {
		return fmt.Errorf("sql: nil inverted index")
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.sealed {
		return nil
	}
	i.ordered = make([]scalarKey, 0, len(i.postings))
	for key := range i.postings {
		i.ordered = append(i.ordered, key)
	}
	sort.Slice(i.ordered, func(left, right int) bool {
		comparison, err := compareValues(i.ordered[left].value(), i.ordered[right].value())
		return err == nil && comparison < 0
	})
	i.lengths = make([]uint32, 0, len(i.arrayLength))
	for length := range i.arrayLength {
		i.lengths = append(i.lengths, length)
	}
	sort.Slice(i.lengths, func(left, right int) bool { return i.lengths[left] < i.lengths[right] })
	i.sealed = true
	return nil
}

func (i *InvertedIndex) RowCount() uint64 {
	if i == nil {
		return 0
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.rows.Count()
}

// Search applies a field predicate. General LIKE forms that cannot be served
// without false negatives return Supported=false.
func (i *InvertedIndex) Search(predicate BoundPredicate) (InvertedResult, error) {
	if i == nil {
		return InvertedResult{}, fmt.Errorf("sql: nil inverted index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if !i.sealed {
		return InvertedResult{}, fmt.Errorf("sql: inverted index %q is not sealed", i.field.Name)
	}
	if err := i.validatePredicate(predicate); err != nil {
		return InvertedResult{}, err
	}
	switch predicate.operator {
	case PredicateIsNull:
		bitmap := i.nulls.Clone()
		if predicate.negated {
			bitmap = i.nonNull.Clone()
		}
		return supportedBitmap(bitmap, InvertedNull, 1), nil
	case PredicateLike:
		return i.searchLike(predicate.like)
	case PredicateIn:
		if i.field.Array {
			return InvertedResult{}, fmt.Errorf("sql: IN index search requires a scalar field")
		}
		bitmap := i.unionValues(predicate.set)
		if predicate.negated {
			negated := i.nonNull.Clone()
			negated.AndNot(bitmap)
			bitmap = negated
		}
		return supportedBitmap(bitmap, InvertedPostingUnion, len(predicate.set)), nil
	case PredicateContainAll, PredicateContainAny:
		if !i.field.Array {
			return InvertedResult{}, fmt.Errorf("sql: contain index search requires an array field")
		}
		bitmap := i.searchContain(predicate)
		return supportedBitmap(bitmap, InvertedPostingUnion, len(predicate.set)), nil
	case PredicateEQ, PredicateNE:
		if i.field.Array {
			return InvertedResult{}, fmt.Errorf("sql: scalar comparison index search received an array field")
		}
		bitmap := i.posting(predicate.right)
		if predicate.operator == PredicateNE {
			negated := i.nonNull.Clone()
			negated.AndNot(bitmap)
			bitmap = negated
		}
		return supportedBitmap(bitmap, InvertedExact, 1), nil
	case PredicateLT, PredicateLE, PredicateGT, PredicateGE:
		if i.field.Array {
			return InvertedResult{}, fmt.Errorf("sql: range index search received an array field")
		}
		return i.searchRange(predicate)
	default:
		return InvertedResult{Supported: false}, nil
	}
}

func (i *InvertedIndex) validatePredicate(predicate BoundPredicate) error {
	switch predicate.operator {
	case PredicateIsNull:
		return nil
	case PredicateLike:
		if i.field.Array || i.field.Kind != ValueString || predicate.like == nil {
			return fmt.Errorf("sql: LIKE index search requires a STRING scalar")
		}
		return nil
	case PredicateIn:
		if i.field.Array || len(predicate.set) == 0 {
			return fmt.Errorf("sql: IN index search requires a non-empty scalar set")
		}
	case PredicateContainAll, PredicateContainAny:
		if !i.field.Array {
			return fmt.Errorf("sql: contain index search requires an array field")
		}
	case PredicateEQ, PredicateNE, PredicateLT, PredicateLE, PredicateGT, PredicateGE:
		if i.field.Array || predicate.right.null || predicate.right.array {
			return fmt.Errorf("sql: comparison index search requires a scalar field and value")
		}
	default:
		return nil
	}
	if len(predicate.set) > 0 && predicate.set[0].kind != i.field.Kind {
		return fmt.Errorf("sql: inverted index %q predicate kind is %s, want %s", i.field.Name, predicate.set[0].kind, i.field.Kind)
	}
	if predicate.operator >= PredicateEQ && predicate.operator <= PredicateGE && predicate.right.kind != i.field.Kind {
		return fmt.Errorf("sql: inverted index %q predicate kind is %s, want %s", i.field.Name, predicate.right.kind, i.field.Kind)
	}
	return nil
}

// SearchArrayLength applies a comparison to indexed array lengths.
func (i *InvertedIndex) SearchArrayLength(predicate BoundPredicate) (InvertedResult, error) {
	if i == nil {
		return InvertedResult{}, fmt.Errorf("sql: nil inverted index")
	}
	i.mu.RLock()
	defer i.mu.RUnlock()
	if !i.sealed || !i.field.Array {
		return InvertedResult{}, fmt.Errorf("sql: array-length search requires a sealed array index")
	}
	if predicate.right.kind != ValueUint32 || predicate.right.null || predicate.right.array {
		return InvertedResult{}, fmt.Errorf("sql: array-length predicate requires UINT32")
	}
	if predicate.operator < PredicateEQ || predicate.operator > PredicateGE {
		return InvertedResult{}, fmt.Errorf("sql: array-length search requires a comparison predicate")
	}
	target := uint32(predicate.right.unsigned)
	if predicate.operator == PredicateNE {
		bitmap := i.nonNull.Clone()
		bitmap.AndNot(i.arrayLength[target])
		return supportedBitmap(bitmap, InvertedArrayLength, 1), nil
	}
	bitmap := ailego.NewBitmap(0)
	start, end := orderedUint32Bounds(i.lengths, target, predicate.operator)
	for _, length := range i.lengths[start:end] {
		bitmap.Or(i.arrayLength[length])
	}
	return supportedBitmap(bitmap, InvertedArrayLength, end-start), nil
}

func (i *InvertedIndex) posting(value Value) *ailego.Bitmap {
	posting := i.postings[keyFromValue(value)]
	if posting == nil {
		return ailego.NewBitmap(0)
	}
	return posting.Clone()
}

func (i *InvertedIndex) unionValues(values []Value) *ailego.Bitmap {
	bitmap := ailego.NewBitmap(0)
	for _, value := range values {
		if posting := i.postings[keyFromValue(value)]; posting != nil {
			bitmap.Or(posting)
		}
	}
	return bitmap
}

func (i *InvertedIndex) searchContain(predicate BoundPredicate) *ailego.Bitmap {
	var bitmap *ailego.Bitmap
	if predicate.operator == PredicateContainAll {
		bitmap = i.nonNull.Clone()
		for _, value := range predicate.set {
			posting := i.postings[keyFromValue(value)]
			if posting == nil {
				bitmap.And(nil)
				break
			}
			bitmap.And(posting)
		}
	} else {
		bitmap = i.unionValues(predicate.set)
	}
	if predicate.negated {
		negated := i.nonNull.Clone()
		negated.AndNot(bitmap)
		return negated
	}
	return bitmap
}

func (i *InvertedIndex) searchRange(predicate BoundPredicate) (InvertedResult, error) {
	bitmap := ailego.NewBitmap(0)
	terms := 0
	strategy := InvertedTermScan
	if i.field.RangeOptimized {
		strategy = InvertedSortedRange
		start, end, err := i.orderedBounds(predicate.right, predicate.operator)
		if err != nil {
			return InvertedResult{}, err
		}
		for _, key := range i.ordered[start:end] {
			bitmap.Or(i.postings[key])
		}
		terms = end - start
	} else {
		for key, posting := range i.postings {
			comparison, err := compareValues(key.value(), predicate.right)
			if err != nil {
				return InvertedResult{}, err
			}
			if comparisonMatches(predicate.operator, comparison) {
				bitmap.Or(posting)
				terms++
			}
		}
	}
	return supportedBitmap(bitmap, strategy, terms), nil
}

func (i *InvertedIndex) orderedBounds(target Value, operator PredicateOperator) (int, int, error) {
	var compareErr error
	lower := sort.Search(len(i.ordered), func(index int) bool {
		comparison, err := compareValues(i.ordered[index].value(), target)
		if err != nil {
			compareErr = err
			return true
		}
		return comparison >= 0
	})
	if compareErr != nil {
		return 0, 0, compareErr
	}
	upper := sort.Search(len(i.ordered), func(index int) bool {
		comparison, err := compareValues(i.ordered[index].value(), target)
		if err != nil {
			compareErr = err
			return true
		}
		return comparison > 0
	})
	if compareErr != nil {
		return 0, 0, compareErr
	}
	switch operator {
	case PredicateEQ:
		return lower, upper, nil
	case PredicateNE:
		// A not-equal interval is disjoint, so its exact posting complement is
		// cheaper and clearer than exposing two ordered slices.
		return 0, 0, fmt.Errorf("sql: ordered bounds do not support !=")
	case PredicateLT:
		return 0, lower, nil
	case PredicateLE:
		return 0, upper, nil
	case PredicateGT:
		return upper, len(i.ordered), nil
	case PredicateGE:
		return lower, len(i.ordered), nil
	default:
		return 0, 0, fmt.Errorf("sql: invalid comparison operator %s", operator)
	}
}

func orderedUint32Bounds(values []uint32, target uint32, operator PredicateOperator) (int, int) {
	lower := sort.Search(len(values), func(index int) bool { return values[index] >= target })
	upper := sort.Search(len(values), func(index int) bool { return values[index] > target })
	switch operator {
	case PredicateEQ:
		return lower, upper
	case PredicateNE:
		return 0, 0
	case PredicateLT:
		return 0, lower
	case PredicateLE:
		return 0, upper
	case PredicateGT:
		return upper, len(values)
	case PredicateGE:
		return lower, len(values)
	default:
		return 0, 0
	}
}

func comparisonMatches(operator PredicateOperator, comparison int) bool {
	switch operator {
	case PredicateEQ:
		return comparison == 0
	case PredicateNE:
		return comparison != 0
	case PredicateLT:
		return comparison < 0
	case PredicateLE:
		return comparison <= 0
	case PredicateGT:
		return comparison > 0
	case PredicateGE:
		return comparison >= 0
	default:
		return false
	}
}

func (i *InvertedIndex) searchLike(pattern *LikePattern) (InvertedResult, error) {
	if i.field.Array || i.field.Kind != ValueString || pattern == nil {
		return InvertedResult{}, fmt.Errorf("sql: LIKE index search requires a STRING scalar")
	}
	percents, underscores := unescapedWildcardCounts(pattern.raw)
	if percents > 1 || underscores > 0 {
		return InvertedResult{Supported: false}, nil
	}
	switch pattern.Mode() {
	case LikeExact:
		return supportedBitmap(i.posting(StringValue(pattern.Literal())), InvertedExact, 1), nil
	case LikeAny:
		return supportedBitmap(i.nonNull.Clone(), InvertedPrefix, len(i.postings)), nil
	case LikePrefix:
		return i.scanStringTerms(pattern.Literal(), "", InvertedPrefix), nil
	case LikeSuffix:
		if !i.field.ExtendedWildcard {
			return InvertedResult{Supported: false}, nil
		}
		return i.scanStringTerms("", pattern.Literal(), InvertedSuffix), nil
	case LikeGeneral:
		_, _, ok := singlePercentTerms(pattern.raw)
		if !ok || !i.field.ExtendedWildcard {
			return InvertedResult{Supported: false}, nil
		}
		return i.scanLikeTerms(pattern, InvertedPrefixSuffix), nil
	default:
		// Two-percent contains and underscore/multi-wildcard forms are
		// evaluated forward, matching the pinned index routing rules.
		return InvertedResult{Supported: false}, nil
	}
}

func (i *InvertedIndex) scanLikeTerms(pattern *LikePattern, strategy InvertedStrategy) InvertedResult {
	bitmap := ailego.NewBitmap(0)
	terms := 0
	for key, posting := range i.postings {
		if pattern.Match(key.text) {
			bitmap.Or(posting)
			terms++
		}
	}
	return supportedBitmap(bitmap, strategy, terms)
}

func (i *InvertedIndex) scanStringTerms(prefix, suffix string, strategy InvertedStrategy) InvertedResult {
	bitmap := ailego.NewBitmap(0)
	terms := 0
	for key, posting := range i.postings {
		if strings.HasPrefix(key.text, prefix) && strings.HasSuffix(key.text, suffix) {
			bitmap.Or(posting)
			terms++
		}
	}
	return supportedBitmap(bitmap, strategy, terms)
}

func singlePercentTerms(pattern string) (string, string, bool) {
	var prefix, suffix strings.Builder
	seenPercent := false
	runes := []rune(pattern)
	for index := 0; index < len(runes); index++ {
		current := runes[index]
		if current == '\\' && index+1 < len(runes) {
			index++
			if seenPercent {
				suffix.WriteRune(runes[index])
			} else {
				prefix.WriteRune(runes[index])
			}
			continue
		}
		if current == '_' || current == '%' && seenPercent {
			return "", "", false
		}
		if current == '%' {
			seenPercent = true
			continue
		}
		if seenPercent {
			suffix.WriteRune(current)
		} else {
			prefix.WriteRune(current)
		}
	}
	return prefix.String(), suffix.String(), seenPercent && prefix.Len() > 0 && suffix.Len() > 0
}

func unescapedWildcardCounts(pattern string) (percents, underscores int) {
	runes := []rune(pattern)
	for index := 0; index < len(runes); index++ {
		if runes[index] == '\\' && index+1 < len(runes) {
			index++
			continue
		}
		switch runes[index] {
		case '%':
			percents++
		case '_':
			underscores++
		}
	}
	return percents, underscores
}

func supportedBitmap(bitmap *ailego.Bitmap, strategy InvertedStrategy, terms int) InvertedResult {
	return InvertedResult{Bitmap: bitmap, Supported: true, Strategy: strategy, Terms: terms}
}

type scalarKey struct {
	kind     ValueKind
	text     string
	boolean  bool
	signed   int64
	unsigned uint64
	bits     uint64
}

func keyFromValue(value Value) scalarKey {
	key := scalarKey{kind: value.kind}
	switch value.kind {
	case ValueBinary:
		key.text = string(value.binary)
	case ValueString:
		key.text = value.text
	case ValueBool:
		key.boolean = value.boolean
	case ValueInt32, ValueInt64:
		key.signed = value.signed
	case ValueUint32, ValueUint64:
		key.unsigned = value.unsigned
	case ValueFloat32, ValueFloat64:
		if value.number == 0 {
			key.bits = 0
		} else {
			key.bits = math.Float64bits(value.number)
		}
	}
	return key
}

func (k scalarKey) value() Value {
	switch k.kind {
	case ValueBinary:
		return BinaryValue([]byte(k.text))
	case ValueString:
		return StringValue(k.text)
	case ValueBool:
		return BoolValue(k.boolean)
	case ValueInt32:
		return Int32Value(int32(k.signed))
	case ValueInt64:
		return Int64Value(k.signed)
	case ValueUint32:
		return Uint32Value(uint32(k.unsigned))
	case ValueUint64:
		return Uint64Value(k.unsigned)
	case ValueFloat32:
		value, _ := Float32Value(float32(math.Float64frombits(k.bits)))
		return value
	case ValueFloat64:
		value, _ := Float64Value(math.Float64frombits(k.bits))
		return value
	default:
		return Value{}
	}
}
