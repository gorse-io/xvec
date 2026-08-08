# FTS query parser and boolean execution

The v0.5 query parser converts zvec's full-text expression language into an
analyzed Go AST and executes it against an immutable term dictionary without
ANTLR, CGO, or generated runtime dependencies. Its grammar and behavior are
pinned to zvec commit `58375ff`.

```go
tokenizer, err := core.NewStandardTokenizer(core.DefaultStandardTokenizerOptions())
analyzer, err := core.NewFTSTokenizerPipeline(
    tokenizer,
    core.NewLowercaseTokenFilter(),
)
query, err := core.ParseFTSQuery(
    ctx,
    `+vector -slow "exact phrase"`,
    analyzer,
    core.FTSDefaultOperatorOR,
)
iterator, err := core.NewFTSQueryIterator(
    ctx, dictionary, query, core.FTSQueryExecutionOptions{},
)
for iterator.Next(ctx) {
    documentID := iterator.DocumentID()
    // consume documentID
}
err = iterator.Err()
```

The same immutable `FTSTokenizerPipeline` can analyze documents and queries,
so punctuation splitting, lowercase, ASCII folding, stemming, n-grams, and
Jieba segmentation stay consistent. A pipeline runs its tokenizer first and
then filters in declaration order. It snapshots the filter slice and is safe
for concurrent calls when its components are safe for concurrent use.
`AnalyzeFTSMatchQuery` is the natural-language counterpart to
`ParseFTSQuery`: it analyzes the entire string without treating `AND`, `OR`,
parentheses, or modifiers as syntax, then combines emitted terms with the
selected default operator.

## Grammar and AST

The lexer applies the baseline's longest-match ordering and emits owned tokens
with byte offsets plus one-based line and zero-based rune columns. It
recognizes case-insensitive `OR`, `AND`, and `NOT`; `+` and `-` modifiers;
parentheses; double-quoted phrases; numbers; regular identifiers; generic
terms and escaped query punctuation. ASCII space, tab, CR, and LF are skipped.
All other input is retained through the catch-all token rule rather than
silently discarded.

The recursive-descent parser preserves these precedence levels:

1. `OR` (lowest)
2. `AND` and binary `NOT`
3. adjacent atoms using `FTSDefaultOperatorOR` or `FTSDefaultOperatorAND`
4. unary `+` / `-` and atoms (highest)

`a NOT b` and `a AND NOT b` both produce `AND(a -b)`. Leading `NOT` is a
syntax error because `NOT` is binary. Modifiers attach to the root of a term,
phrase, multi-token analyzed term, or parenthesized group. A bare lexer term
that the analyzer splits into several tokens becomes an AND/OR subtree using
the configured default operator. A double-quoted phrase always remains a
phrase node, including when analysis produces zero terms. If analysis removes
every bare term, the parser returns `FTSEmptyQueryNode`.

Parser-produced nodes are `FTSTermQueryNode`, `FTSPhraseQueryNode`,
`FTSAndQueryNode`, `FTSOrQueryNode`, and `FTSEmptyQueryNode`. Every node carries
`Must`, `MustNot`, `Should`, and `Boost` metadata used by canonicalization and
the later scoring stage. `String` returns the baseline debug form.

Field-prefixed atoms such as `title:vector` and boosts such as `vector^2` are
recognized by the grammar but return `ErrUnsupportedFTSQuery`, matching the
pinned execution boundary. They are never silently treated as ordinary terms.

## Canonicalization and execution

`SimplifyFTSQuery` makes an owned clone and applies the baseline structural
rewrite. It safely flattens nested AND/OR nodes, merges duplicate term or
phrase siblings by adding their boosts, propagates empty nodes, detects
positive/negative contradictions, folds single-child composites, and rewrites
OR occurrence modifiers into canonical AND `must`, `must-not`, and `should`
buckets. The input AST is never mutated, cycles and non-finite boosts are
rejected, and simplifying an already simplified tree is idempotent.

`NewFTSQueryIterator` performs this rewrite automatically and creates a lazy
iterator over `FTSTermDictionary`:

- term nodes seek directly through compressed posting lists;
- phrase nodes first intersect term postings, then verify exact adjacent
  positions, including repeated terms;
- AND nodes align their cheapest positive iterator with all required children
  and apply negative exclusions;
- OR nodes merge exact child streams without duplicates; and
- empty, missing required terms, and negative-only roots match no documents.

`Next(ctx)` and `Advance(ctx, target)` emit segment-local document IDs in strict
ascending order. `Advance` retains an already-suitable current match. Optional
deletion bits are validated and snapshotted at construction, so concurrent
tombstone changes cannot alter an in-flight query. `Cost` exposes the posting
work estimate, `Valid`/`DocumentID` expose current state, and `Err` returns the
first sticky context or execution error. Should clauses affect scoring only,
so they do not narrow this match-only iterator.

## Errors, cancellation, and limits

Parsing is deliberately two-phase: the complete syntax is validated first,
then semantic support is checked and term analysis runs. Consequently a later
syntax error cannot be hidden by an earlier tokenizer failure, and unsupported
field/boost atoms do not invoke the analyzer.

Syntax, unsupported syntax, and complexity failures return
`FTSQueryParseError`, which records byte offset, line, column, and a stable
message. It supports `errors.As`, while its kind supports `errors.Is` with
`ErrFTSQuerySyntax`, `ErrUnsupportedFTSQuery`, or
`ErrFTSQueryTooComplex`. Invalid configuration matches `ErrInvalidFTSQuery` or
`ErrInvalidFTSAnalyzer`. Context cancellation propagates through lexing,
parsing, tokenization, filters, AST cloning, posting alignment, and phrase
position verification.

The parser accepts at most `MaxFTSQueryTokens` lexer tokens and the same total
number of analyzer-produced tokens, plus `MaxFTSQueryDepth` nested parenthesis
levels. Exceeding a bound returns an explicit complexity error before stack or
memory exhaustion.

The pinned fixture records hashes of the baseline lexer/parser grammars, AST,
parser, rewriter, and term/AND/OR/phrase iterators. Tests cover longest-match
precedence, source locations, operator precedence, modifiers, phrases,
escapes, analyzer splitting and filtering, rewrite idempotence, term and
boolean set semantics, repeated-term phrase positions, missing terms, exact
error priority, ownership, deletion snapshots, seek behavior, cancellation,
concurrency, fuzzing, and resource bounds.

`NewFTSQueryIterator` intentionally remains score-free;
`NewFTSScoredQueryIterator`, block-max BM25 top-k search, and physical FTS index
merging are documented separately. Collection Query and MultiQuery integration
is complete.

```sh
go test ./internal/core -run '^Test(FTSQuery|LexFTS|ParseFTS|AnalyzeFTS|FTSTokenizer)'
go test ./internal/core -run '^$' -fuzz '^FuzzLexFTSQuery$'
go test ./internal/core -run '^$' -fuzz '^FuzzParseFTSQuery$'
go test ./internal/core -run '^$' -fuzz '^FuzzSimplifyFTSQuery$'
go test ./internal/core -run '^$' -fuzz '^FuzzFTSQueryIterator$'
go test ./internal/core -run '^$' -bench '^(BenchmarkParseFTSQuery|BenchmarkFTSQueryIterator)$' -benchmem
```
