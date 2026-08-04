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

package zvec

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/core"
)

const (
	// DefaultMultiQueryTopK is the pinned default final result count.
	DefaultMultiQueryTopK = 10
	// DefaultSubQueryCandidates is the pinned default candidate count per
	// MultiQuery branch.
	DefaultSubQueryCandidates = 10
	// MaxQueryTopK is the pinned upper bound for a MultiQuery final or branch
	// result count.
	MaxQueryTopK = 100_000
)

// FTSClause describes one full-text target. Exactly one of Query and Match
// must be non-empty. Query uses the FTS expression grammar; Match analyzes the
// text as natural language without interpreting operators.
type FTSClause struct {
	Query string
	Match string
}

// SubQuery describes one candidate-producing branch of MultiQuery. Exactly
// one of DenseVector, SparseVector, and FTS must be set. A zero NumCandidates
// selects DefaultSubQueryCandidates.
type SubQuery struct {
	Field         string
	DenseVector   DenseVector
	SparseVector  SparseVector
	FTS           *FTSClause
	Params        QueryParams
	NumCandidates int
}

// RerankBatch contains one projected, score-ordered candidate list and the
// corresponding independent field schema. Batches retain SubQuery order.
// Documents and their field values are owned by the batch and may be changed
// by the reranker without mutating the collection snapshot.
type RerankBatch struct {
	Field     FieldSchema
	Documents []Document
}

// Reranker combines MultiQuery candidate batches. Implementations may adjust
// scores and order but must return distinct documents drawn from the supplied
// batches. Implementations shared by concurrent calls must be concurrency-safe.
type Reranker interface {
	Rerank(ctx context.Context, batches []RerankBatch, topK int) ([]Document, error)
}

// MultiQuery combines vector, sparse-vector, and full-text candidate lists
// through a Reranker. Nil selects NewRRFReranker. At least two sub-queries are
// required. Zero TopK selects DefaultMultiQueryTopK.
type MultiQuery struct {
	Queries    []SubQuery
	TopK       int
	Filter     string
	Projection Projection
	Reranker   Reranker
}

// MultiQuery executes every branch over one live-document snapshot, applies
// one shared scalar filter, and delegates fusion to the configured or default
// Reranker. BM25 corpus statistics include every live document; the scalar
// filter masks FTS candidates without changing IDF or average document length.
func (c *Collection) MultiQuery(ctx context.Context, query MultiQuery) ([]Document, error) {
	const op = "multi query"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(query.Queries) < 2 {
		return nil, invalidArgument(op, "at least two sub-queries are required")
	}
	topK, err := normalizedMultiQueryCount(query.TopK, DefaultMultiQueryTopK, "TopK")
	if err != nil {
		return nil, err
	}
	reranker := query.Reranker
	if isNilInterface(reranker) {
		value := NewRRFReranker()
		reranker = value
	}
	projection := query.Projection.Clone()

	locked := true
	c.mu.RLock()
	defer func() {
		if locked {
			c.mu.RUnlock()
		}
	}()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	if err := projection.Validate(c.schema); err != nil {
		return nil, err
	}
	filterPlan, err := buildFilterPlan(query.Filter, c.schema)
	if err != nil {
		return nil, invalidArgument(op, "invalid filter: %v", err)
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return nil, wrapCollectionError(op, c.path, err)
	}
	candidateFilter, err := evaluateFilterDocuments(ctx, filterPlan, documents)
	if err != nil {
		return nil, wrapFilterEvaluationError(op, c.path, err)
	}

	batches := make([]RerankBatch, len(query.Queries))
	candidateIDs := make(map[uint64]struct{})
	ftsFields := make(map[string]*collectionFTSRuntime)
	for index, subQuery := range query.Queries {
		if err := ctx.Err(); err != nil {
			return nil, wrapCollectionError(op, c.path, err)
		}
		candidateCount, err := normalizedMultiQueryCount(
			subQuery.NumCandidates, DefaultSubQueryCandidates,
			fmt.Sprintf("Queries[%d].NumCandidates", index),
		)
		if err != nil {
			return nil, err
		}
		field, found := c.schema.Field(subQuery.Field)
		if !found {
			return nil, invalidArgument(op, "sub-query %d field %q does not exist", index, subQuery.Field)
		}
		target, err := multiQueryTargetKind(subQuery)
		if err != nil {
			return nil, invalidArgument(op, "sub-query %d: %v", index, err)
		}

		var results []core.Result
		switch target {
		case multiQueryTargetDense, multiQueryTargetSparse:
			if !field.DataType.IsVector() {
				return nil, invalidArgument(op, "sub-query %d field %q is not a vector field", index, field.Name)
			}
			results, err = c.searchVectorSnapshot(
				ctx, op, field, subQuery.DenseVector, subQuery.SparseVector,
				candidateCount, subQuery.Params, documents, candidateFilter,
			)
		case multiQueryTargetFTS:
			runtime := ftsFields[field.Name]
			if runtime == nil {
				runtime, err = buildCollectionFTSRuntime(ctx, field, documents, candidateFilter)
				if err == nil {
					ftsFields[field.Name] = runtime
				}
			}
			if err == nil {
				results, err = searchCollectionFTS(ctx, runtime, subQuery.FTS, subQuery.Params, candidateCount)
			}
		}
		if err != nil {
			return nil, wrapMultiQueryBranchError(op, c.path, index, err)
		}
		materialized, err := c.materializeResults(documents, results, projection)
		if err != nil {
			return nil, err
		}
		for _, result := range results {
			candidateIDs[result.Key] = struct{}{}
		}
		batches[index] = RerankBatch{Field: field.Clone(), Documents: materialized}
	}

	// Release the collection lock before invoking caller code. The immutable
	// snapshot, schema, and candidate batches remain owned by this call.
	schema := c.schema.Clone()
	path := c.path
	c.mu.RUnlock()
	locked = false
	if err := ctx.Err(); err != nil {
		return nil, wrapCollectionError(op, path, err)
	}
	reranked, err := reranker.Rerank(ctx, batches, topK)
	if err != nil {
		return nil, wrapCollectionError(op, path, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, wrapCollectionError(op, path, err)
	}
	return validateAndProjectReranked(op, path, schema, documents, candidateIDs, projection, reranked, topK)
}

func normalizedMultiQueryCount(value, fallback int, name string) (int, error) {
	if value == 0 {
		return fallback, nil
	}
	if value < 0 || value > MaxQueryTopK {
		return 0, invalidArgument("multi query", "%s must be in [1, %d] or zero for the default", name, MaxQueryTopK)
	}
	return value, nil
}

type multiQueryTarget uint8

const (
	multiQueryTargetDense multiQueryTarget = iota + 1
	multiQueryTargetSparse
	multiQueryTargetFTS
)

func multiQueryTargetKind(query SubQuery) (multiQueryTarget, error) {
	hasDense := !isNilInterface(query.DenseVector)
	hasSparse := !isNilInterface(query.SparseVector)
	hasFTS := !isNilInterface(query.FTS)
	count := 0
	if hasDense {
		count++
	}
	if hasSparse {
		count++
	}
	if hasFTS {
		count++
	}
	if count != 1 {
		return 0, fmt.Errorf("exactly one of DenseVector, SparseVector, and FTS must be set")
	}
	if hasDense {
		return multiQueryTargetDense, nil
	}
	if hasSparse {
		return multiQueryTargetSparse, nil
	}
	return multiQueryTargetFTS, nil
}

type collectionFTSRuntime struct {
	analyzer    core.FTSAnalyzer
	dictionary  *core.FTSTermDictionary
	scorer      *core.BM25Scorer
	deleted     *ailego.Bitmap
	documentIDs []uint64
}

func buildCollectionFTSRuntime(
	ctx context.Context,
	field FieldSchema,
	documents []Document,
	candidateFilter core.CandidateFilter,
) (*collectionFTSRuntime, error) {
	if field.DataType != DataTypeString || field.IndexType() != IndexTypeFTS {
		return nil, invalidArgument("multi query", "field %q is not an FTS-indexed STRING field", field.Name)
	}
	if uint64(len(documents)) > uint64(math.MaxUint32) {
		return nil, &Error{
			Code: ErrorCodeResourceExhausted, Op: "multi query",
			Message: fmt.Sprintf("FTS field %q exceeds the uint32 document domain", field.Name),
		}
	}
	params, err := collectionFTSIndexParams(field)
	if err != nil {
		return nil, err
	}
	analyzer, err := newCollectionFTSAnalyzer(ctx, params)
	if err != nil {
		return nil, err
	}
	builder := core.NewFTSFieldBuilder()
	documentIDs := make([]uint64, len(documents))
	var deleted *ailego.Bitmap
	if candidateFilter != nil {
		deleted = ailego.NewBitmap(uint64(len(documents)))
	}
	for index := range documents {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		document := &documents[index]
		documentIDs[index] = document.DocID
		if deleted != nil && !candidateFilter(document.DocID) {
			deleted.Set(uint64(index))
		}
		var text string
		if raw, found := document.Fields[field.Name]; found && raw != nil {
			var ok bool
			text, ok = raw.(string)
			if !ok {
				return nil, fmt.Errorf("stored document %d field %q is %T, want string", document.DocID, field.Name, raw)
			}
		}
		tokens, err := analyzer.Analyze(ctx, text)
		if err != nil {
			return nil, err
		}
		if err := builder.AddDocument(ctx, uint32(index), tokens); err != nil {
			return nil, err
		}
	}
	dictionary, err := builder.Build(ctx)
	if err != nil {
		return nil, err
	}
	// Shared scalar filtering is deliberately absent from corpus statistics:
	// it affects eligibility, not BM25 IDF or length normalization.
	stats, err := core.AggregateFTSCorpusStats(ctx, []core.FTSSegmentView{{Dictionary: dictionary}})
	if err != nil {
		return nil, err
	}
	scorer, err := core.NewBM25Scorer(core.DefaultBM25Params(), stats)
	if err != nil {
		return nil, err
	}
	return &collectionFTSRuntime{
		analyzer: analyzer, dictionary: dictionary, scorer: scorer,
		deleted: deleted, documentIDs: documentIDs,
	}, nil
}

func searchCollectionFTS(
	ctx context.Context,
	runtime *collectionFTSRuntime,
	clause *FTSClause,
	queryParams QueryParams,
	topK int,
) ([]core.Result, error) {
	if runtime == nil || clause == nil {
		return nil, invalidArgument("multi query", "FTS runtime and clause are required")
	}
	hasQuery, hasMatch := clause.Query != "", clause.Match != ""
	if hasQuery == hasMatch {
		return nil, invalidArgument("multi query", "exactly one of FTS Query and Match must be non-empty")
	}
	defaultOperator, err := collectionFTSDefaultOperator(queryParams)
	if err != nil {
		return nil, err
	}
	var node core.FTSQueryNode
	if hasQuery {
		node, err = core.ParseFTSQuery(ctx, clause.Query, runtime.analyzer, defaultOperator)
	} else {
		node, err = core.AnalyzeFTSMatchQuery(ctx, clause.Match, runtime.analyzer, defaultOperator)
	}
	if err != nil {
		return nil, err
	}
	results, err := core.SearchFTS(ctx, runtime.dictionary, node, runtime.scorer, core.FTSSearchOptions{
		TopK:                     topK,
		FTSQueryExecutionOptions: core.FTSQueryExecutionOptions{DeletedDocuments: runtime.deleted},
	})
	if err != nil {
		return nil, err
	}
	output := make([]core.Result, len(results))
	for index, result := range results {
		if uint64(result.DocumentID) >= uint64(len(runtime.documentIDs)) {
			return nil, fmt.Errorf("FTS result document ID %d is outside the snapshot", result.DocumentID)
		}
		output[index] = core.Result{Key: runtime.documentIDs[result.DocumentID], Score: result.Score}
	}
	return output, nil
}

func collectionFTSDefaultOperator(params QueryParams) (core.FTSDefaultOperator, error) {
	value := FTSQueryParams{}
	if !isNilInterface(params) {
		if params.IndexType() != IndexTypeFTS {
			return 0, invalidArgument("multi query", "query parameters for %s cannot be used with FTS", params.IndexType())
		}
		if err := params.Validate(); err != nil {
			return 0, err
		}
		switch typed := params.(type) {
		case FTSQueryParams:
			value = typed
		case *FTSQueryParams:
			if typed == nil {
				return 0, invalidArgument("multi query", "FTS query parameters are nil")
			}
			value = *typed
		default:
			return 0, invalidArgument("multi query", "invalid FTS query parameter value")
		}
	}
	operator, err := core.ParseFTSDefaultOperator(value.DefaultOperator)
	if err != nil {
		return 0, invalidArgument("multi query", "invalid FTS default operator: %v", err)
	}
	return operator, nil
}

func collectionFTSIndexParams(field FieldSchema) (FTSIndexParams, error) {
	if indexParamsNil(field.Index) || field.Index.IndexType() != IndexTypeFTS {
		return FTSIndexParams{}, invalidArgument("multi query", "field %q does not have FTS index parameters", field.Name)
	}
	var params FTSIndexParams
	switch value := field.Index.(type) {
	case FTSIndexParams:
		params = value
	case *FTSIndexParams:
		if value == nil {
			return FTSIndexParams{}, invalidArgument("multi query", "field %q has nil FTS index parameters", field.Name)
		}
		params = *value
	default:
		return FTSIndexParams{}, invalidArgument("multi query", "field %q has invalid FTS index parameters", field.Name)
	}
	if err := params.Validate(); err != nil {
		return FTSIndexParams{}, err
	}
	return params, nil
}

func newCollectionFTSAnalyzer(ctx context.Context, params FTSIndexParams) (core.FTSAnalyzer, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}
	extra, err := decodeFTSExtraParams(params.ExtraParams)
	if err != nil {
		return nil, err
	}
	tokenizerName := params.Tokenizer
	if tokenizerName == "" {
		tokenizerName = "standard"
	}
	var tokenizer core.Tokenizer
	switch tokenizerName {
	case "whitespace":
		tokenizer = core.NewWhitespaceTokenizer()
	case "standard":
		options := core.DefaultStandardTokenizerOptions()
		if value, found := extra["max_token_length"]; found {
			length, _ := jsonPositiveInteger(value)
			options.MaxTokenLength = uint32(length)
		}
		tokenizer, err = core.NewStandardTokenizer(options)
	case "ngram":
		options := core.DefaultNGramTokenizerOptions()
		minimum, _ := ngramSize(extra, "ngram_min", int64(options.Min))
		maximum, _ := ngramSize(extra, "ngram_max", int64(options.Max))
		options.Min, options.Max = uint32(minimum), uint32(maximum)
		if raw, found := extra["token_chars"]; found {
			for _, item := range raw.([]any) {
				switch item.(string) {
				case "letter":
					options.TokenChars |= core.NGramTokenCharLetter
				case "digit":
					options.TokenChars |= core.NGramTokenCharDigit
				case "whitespace":
					options.TokenChars |= core.NGramTokenCharWhitespace
				case "punctuation":
					options.TokenChars |= core.NGramTokenCharPunctuation
				case "symbol":
					options.TokenChars |= core.NGramTokenCharSymbol
				}
			}
		}
		tokenizer, err = core.NewNGramTokenizer(options)
	case "jieba":
		options := core.DefaultJiebaTokenizerOptions()
		if value, found := extra["jieba_dict_dir"]; found {
			options.DictDir = value.(string)
		}
		if value, found := extra["user_dict_path"]; found {
			options.UserDictPath = value.(string)
		}
		if value, found := extra["cut_mode"]; found {
			options.CutMode = core.JiebaCutMode(value.(string))
		}
		tokenizer, err = core.NewJiebaTokenizer(ctx, options)
	default:
		return nil, invalidArgument("multi query", "unknown FTS tokenizer %q", tokenizerName)
	}
	if err != nil {
		return nil, err
	}
	filters := make([]core.TokenFilter, 0, len(params.Filters))
	for _, name := range params.Filters {
		switch name {
		case "lowercase":
			filters = append(filters, core.NewLowercaseTokenFilter())
		case "ascii_folding":
			filters = append(filters, core.NewASCIIFoldingTokenFilter())
		case "stemmer":
			options := core.DefaultStemmerTokenFilterOptions()
			if value, found := extra["stemmer_lang"]; found {
				options.Language = value.(string)
			}
			filter, filterErr := core.NewStemmerTokenFilter(options)
			if filterErr != nil {
				return nil, filterErr
			}
			filters = append(filters, filter)
		default:
			return nil, invalidArgument("multi query", "unknown FTS token filter %q", name)
		}
	}
	return core.NewFTSTokenizerPipeline(tokenizer, filters...)
}

func wrapMultiQueryBranchError(op, path string, index int, err error) error {
	if err == nil {
		return nil
	}
	var structured *Error
	if errors.As(err, &structured) {
		wrapped := wrapCollectionError(op, path, err)
		var result *Error
		if errors.As(wrapped, &result) && result != nil {
			copy := *result
			copy.Op = op
			copy.Path = path
			copy.Message = fmt.Sprintf("sub-query %d: %s", index, errorDetail(result))
			return &copy
		}
		return wrapped
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return wrapCollectionError(op, path, err)
	}
	if errors.Is(err, core.ErrFTSQuerySyntax) ||
		errors.Is(err, core.ErrUnsupportedFTSQuery) ||
		errors.Is(err, core.ErrInvalidFTSQuery) ||
		errors.Is(err, core.ErrFTSQueryTooComplex) ||
		errors.Is(err, core.ErrTokenizerInputTooLarge) {
		return &Error{
			Code: ErrorCodeInvalidArgument, Op: op, Path: path,
			Message: fmt.Sprintf("sub-query %d has an invalid FTS query", index), Err: err,
		}
	}
	return &Error{
		Code: ErrorCodeInternal, Op: op, Path: path,
		Message: fmt.Sprintf("execute sub-query %d", index), Err: err,
	}
}

func errorDetail(err *Error) string {
	if err == nil {
		return "query failed"
	}
	if err.Message != "" {
		return err.Message
	}
	return err.Code.DefaultMessage()
}

func validateAndProjectReranked(
	op, path string,
	schema CollectionSchema,
	documents []Document,
	candidateIDs map[uint64]struct{},
	projection Projection,
	reranked []Document,
	topK int,
) ([]Document, error) {
	if len(reranked) > topK {
		return nil, &Error{
			Code: ErrorCodeInvalidArgument, Op: op, Path: path,
			Message: fmt.Sprintf("reranker returned %d documents, exceeding TopK %d", len(reranked), topK),
		}
	}
	byID := make(map[uint64]Document, len(documents))
	for _, document := range documents {
		byID[document.DocID] = document
	}
	seen := make(map[uint64]struct{}, len(reranked))
	output := make([]Document, len(reranked))
	for index, result := range reranked {
		if _, found := candidateIDs[result.DocID]; !found {
			return nil, invalidRerankerOutput(op, path, "document %d was not returned by any sub-query", result.DocID)
		}
		source, found := byID[result.DocID]
		if !found || source.PrimaryKey != result.PrimaryKey {
			return nil, invalidRerankerOutput(op, path, "document %d has an invalid primary key", result.DocID)
		}
		if _, duplicate := seen[result.DocID]; duplicate {
			return nil, invalidRerankerOutput(op, path, "document %d appears more than once", result.DocID)
		}
		seen[result.DocID] = struct{}{}
		if value := float64(result.Score); math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, invalidRerankerOutput(op, path, "document %d has a non-finite score", result.DocID)
		}
		source.Score = result.Score
		projected, err := ProjectDocument(source, schema, projection)
		if err != nil {
			return nil, err
		}
		output[index] = projected
	}
	return output, nil
}

func invalidRerankerOutput(op, path, format string, args ...any) error {
	return &Error{
		Code: ErrorCodeInvalidArgument, Op: op, Path: path,
		Message: "invalid reranker output: " + fmt.Sprintf(format, args...),
	}
}
