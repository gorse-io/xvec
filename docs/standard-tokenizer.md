# Standard tokenizer

The v0.5 standard tokenizer is a pure-Go port of the tokenizer in zvec commit
`58375ff`. It implements the baseline's Lucene/Elasticsearch-oriented Unicode
word behavior with generated Unicode 17.0.0 property tables. It has no CGO,
locale, ICU, or runtime data-file dependency.

`StandardTokenizer.Tokenize` accepts a context and returns owned `Token`
values. Offsets are uint32 byte offsets into the original input, positions are
contiguous and zero-based, and malformed UTF-8 bytes act as individual token
boundaries. Empty input returns a non-nil empty result; cancellation returns no
partial token list.

The tokenizer applies these baseline rules:

- Latin and other alphabetic scripts join letters and numbers, with the
  baseline apostrophe, period, comma, colon, semicolon, and underscore rules.
- CJK ideographs and Hiragana are emitted one character at a time; trailing
  combining marks remain attached.
- Katakana, Hangul, and Southeast Asian complex-context runs remain grouped.
- Combining marks, format characters, and ZWJ continue tokens but do not start
  ordinary word tokens.
- Unicode emoji sequences include keycaps, variation selectors, modifiers,
  extended-pictographic ZWJ chains, and paired regional indicators.
- `MaxTokenLength` counts decoded codepoints rather than bytes. Its accepted
  range is 1 through 1,048,576 and its default is 255. Long spans split only at
  valid token starts, so connector-only fragments are not emitted.

```go
options := core.DefaultStandardTokenizerOptions()
tokenizer, err := core.NewStandardTokenizer(options)
if err != nil {
    return err
}
tokens, err := tokenizer.Tokenize(ctx, "Go向量 search 3.14")
```

The checked-in Unicode tables are derived from Unicode Character Database
17.0.0 `WordBreakProperty.txt`, `emoji-data.txt`, `LineBreak.txt`, and
`Scripts.txt`. Their Unicode License V3 notice is reproduced in `NOTICE`. The
compatibility fixture records both pinned C++ source hashes, preventing an
unreviewed Unicode-data or baseline change.

Standard tokenization now composes with token filters, posting lists, query
parsing, BM25, and Collection FTS execution through `FTSIndexParams`.

Verification includes the pinned fixture, script and emoji behavior matrices,
invalid UTF-8, cancellation, randomized source-range invariants, an example,
and a throughput benchmark:

```sh
go test ./internal/core -run '^TestStandardTokenizer'
go test ./internal/core -run '^$' -fuzz '^FuzzStandardTokenizer$'
go test ./internal/core -run '^$' -bench '^BenchmarkStandardTokenizer$' -benchmem
```
