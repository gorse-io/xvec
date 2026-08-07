# ASCII-folding token filter

The v0.5 ASCII-folding token filter is a pure-Go port of the filter in zvec
commit `58375ff`. It uses pinned utf8proc 2.11.3 / Unicode 17.0.0 compatibility
decomposition plus the baseline's 143-entry Lucene-inspired supplemental
table. It has no CGO, locale, normalization-library, or runtime data-file
dependency.

`ASCIIFoldingTokenFilter.Filter` accepts a context and returns a new owned token
slice. It preserves token order, byte offsets, and positions, including gaps in
positions. Empty input tokens are removed, matching the baseline. The stateless
filter is safe for concurrent use, and cancellation returns no partial result.

```go
filter := core.NewASCIIFoldingTokenFilter()
tokens, err := filter.Filter(ctx, []core.Token{
    {Text: "café", Offset: 0, Position: 0},
    {Text: "Æsir", Offset: 6, Position: 1},
})
```

For each valid non-ASCII codepoint, the filter first checks the supplemental
table. Otherwise it applies the pinned equivalent of stable NFKD compatibility
decomposition with combining marks stripped, but accepts that result only when
every output byte is ASCII. This yields behavior such as:

- `café` → `cafe`, `ÜBER` → `UBER`;
- `Æ`, `ß`, `Þ`, and `Œ` → `AE`, `ss`, `TH`, and `OE`;
- `ﬁ`, fullwidth `Ａ０`, circled `①`, and Roman `Ⅳ` → `fi`, `A0`, `1`, and
  `IV`; and
- `‹x›` and `←→` → `<x>` and `<-->`.

This is not general transliteration. A codepoint whose decomposition still
contains non-ASCII text passes through unchanged. CJK and Greek tonos therefore
remain intact. A standalone combining mark also remains unchanged because
stripping it would produce an empty mapping, while a precomposed Latin letter
can fold. Each malformed UTF-8 byte is copied unchanged and processing resumes
at the next byte.

The checked-in generated tables contain 1,986 accepted NFKD mappings and the
143 explicit supplemental entries; after overlap, 2,120 codepoints fold to
ASCII. Exhaustive tests reconstruct and hash all pinned source and effective
mappings. The Unicode License V3 notice is reproduced in `NOTICE`.

This unit provides the reusable ASCII-folding primitive. Tokenizer-pipeline
configuration, posting lists, query parsing, BM25, and Collection FTS execution
now compose it through `FTSIndexParams`.

Verification includes the exhaustive table identities, compatibility and
supplemental mappings, unmapped scripts, standalone marks, malformed UTF-8,
empty-token removal, metadata and ownership, cancellation, concurrent use,
fuzzed idempotence, an example, and a throughput benchmark:

```sh
go test ./internal/core -run '^TestASCIIFolding'
go test ./internal/core -run '^$' -fuzz '^FuzzASCIIFoldingTokenFilter$'
go test ./internal/core -run '^$' -bench '^BenchmarkASCIIFoldingTokenFilter$' -benchmem
```
