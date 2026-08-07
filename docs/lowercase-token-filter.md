# Lowercase token filter

The v0.5 lowercase token filter is a pure-Go implementation of the filter in
zvec commit `58375ff`. It uses the exact utf8proc 2.11.3 simple lowercase
mappings from Unicode 17.0.0 rather than the host Go release's Unicode tables,
so results remain stable across supported platforms and Go upgrades.

`LowercaseTokenFilter.Filter` accepts a context and returns a new owned token
slice. It changes only `Token.Text`; byte offsets, output positions, token
order, and the caller's input slice remain unchanged. The stateless filter is
safe for concurrent use, and cancellation returns no partial result.

```go
filter := core.NewLowercaseTokenFilter()
tokens, err := filter.Filter(ctx, []core.Token{
    {Text: "Go", Offset: 0, Position: 0},
    {Text: "ÜBER", Offset: 3, Position: 1},
})
```

The operation is Unicode simple lowercasing, not locale-sensitive lowercasing,
full case folding, or normalization. For example, U+0130 LATIN CAPITAL LETTER I
WITH DOT ABOVE maps to plain `i`, U+1E9E LATIN CAPITAL LETTER SHARP S maps to
`ß`, and Greek sigma does not use word-final context. CJK, punctuation, and
other codepoints without a lowercase mapping pass through unchanged.

Valid UTF-8 is decoded with the same validity rules as the pinned utf8proc
iterator. Each malformed byte is copied unchanged and processing resumes at
the next byte, so adjacent valid uppercase text is still lowercased. A valid
encoded U+FFFD remains distinguishable from malformed bytes.

The generated mapping ranges are covered by the Unicode License V3 notice in
`NOTICE`. An exhaustive identity test recreates all 1,488 changed
codepoint/mapping pairs and verifies the SHA-256 recorded from the pinned
utf8proc data, in addition to the C++ behavior fixture.

This unit adds the reusable `TokenFilter` contract and lowercase primitive.
Tokenizer pipelines, posting lists, query parsing, BM25, and Collection FTS
execution now compose it through `FTSIndexParams`.

Verification includes ASCII, Latin, Cyrillic, Greek, Unicode 17 additions,
simple-versus-full-case distinctions, malformed UTF-8, metadata and ownership,
cancellation, concurrent use, fuzzed idempotence, an example, and a throughput
benchmark:

```sh
go test ./internal/core -run '^TestLowercaseTokenFilter|^TestLowercaseUnicode17'
go test ./internal/core -run '^$' -fuzz '^FuzzLowercaseTokenFilter$'
go test ./internal/core -run '^$' -bench '^BenchmarkLowercaseTokenFilter$' -benchmem
```
