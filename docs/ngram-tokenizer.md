# Ngram tokenizer

The v0.5 ngram tokenizer is a pure-Go port of the tokenizer in zvec commit
`58375ff`. It emits overlapping UTF-8 codepoint ngrams without CGO, locale
state, or runtime Unicode data files.

`NGramTokenizer.Tokenize` accepts a context and returns owned `Token` values.
Offsets are uint32 byte offsets into the original input, positions are
contiguous and zero-based, and output is ordered first by starting codepoint
and then by increasing ngram length. Cancellation returns no partial result.

The default minimum and maximum lengths are both 2. Lengths are positive
uint32 values, `Min` must not exceed `Max`, and the difference may be at most
one. Lengths count decoded codepoints rather than bytes.

By default, `TokenChars` is zero and every valid UTF-8 codepoint is retained,
including whitespace, marks, controls, NUL, and unassigned characters. This is
intentional baseline behavior. Each malformed UTF-8 byte always terminates a
segment. An explicit mask limits segments to any union of:

- `NGramTokenCharLetter`;
- `NGramTokenCharDigit`;
- `NGramTokenCharWhitespace`;
- `NGramTokenCharPunctuation`; and
- `NGramTokenCharSymbol`.

Character classes match the baseline's utf8proc 2.11.3 / Unicode 17.0.0
categories. Whitespace follows Elasticsearch-style handling: U+0009–U+000D,
U+001C–U+001F, ordinary space separators, and line/paragraph separators are
included, while non-breaking space U+00A0, figure space U+2007, and narrow
non-breaking space U+202F are excluded.

```go
tokenizer, err := core.NewNGramTokenizer(core.NGramTokenizerOptions{
    Min: 2,
    Max: 3,
    TokenChars: core.NGramTokenCharLetter | core.NGramTokenCharDigit,
})
if err != nil {
    return err
}
tokens, err := tokenizer.Tokenize(ctx, "vector向量")
```

The generated category ranges are covered by the Unicode License V3 notice in
`NOTICE`. The compatibility fixture records the pinned C++ tokenizer source,
utf8proc commit, Unicode version, and data hashes.

This unit provides ngram tokenization only. Token filters, posting lists, FTS
parsing, BM25, and Collection FTS execution remain separate v0.5 units and
continue to return `ErrNotSupported` at their existing public operation
boundaries.

Verification includes fixed cross-language fixtures, Unicode 17 additions,
category boundaries and exclusions, NUL and malformed UTF-8, cancellation,
fuzzed range and length invariants, an example, and a throughput benchmark:

```sh
go test ./internal/core -run '^TestNGramTokenizer'
go test ./internal/core -run '^$' -fuzz '^FuzzNGramTokenizer$'
go test ./internal/core -run '^$' -bench '^BenchmarkNGramTokenizer$' -benchmem
```
