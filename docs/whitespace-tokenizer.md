# Whitespace tokenizer

The v0.5 whitespace tokenizer is the first native Go full-text analysis
primitive. It follows the byte-oriented behavior of zvec commit `58375ff` and
does not require dictionaries or external packages.

`WhitespaceTokenizer.Tokenize` accepts a context and returns ordered `Token`
values. Each token contains owned text, its uint32 byte offset in the original
input, and a contiguous zero-based position. Empty and whitespace-only inputs
return a non-nil empty result. Cancellation returns no partial token list.

The delimiter set is intentionally the six ASCII whitespace bytes recognized
by C `isspace` in the C locale:

- space (`0x20`);
- horizontal tab (`0x09`);
- line feed (`0x0a`);
- vertical tab (`0x0b`);
- form feed (`0x0c`); and
- carriage return (`0x0d`).

All other bytes are preserved, including punctuation, case, embedded NUL,
invalid UTF-8, non-breaking space, and other Unicode whitespace code points.
Offsets are byte offsets rather than rune indexes, matching the baseline token
contract and later posting-position storage. Inputs larger than uint32 offset
capacity fail explicitly.

```go
tokenizer := core.NewWhitespaceTokenizer()
tokens, err := tokenizer.Tokenize(ctx, "  Go\t向量 search")
```

Whitespace tokenization now composes with token filters, posting lists, query
parsing, BM25, and Collection FTS execution through `FTSIndexParams`.

Verification includes fixed byte-offset fixtures, all delimiter bytes,
Unicode and invalid-UTF-8 preservation, cancellation, fuzzed partition
invariants, examples, and a throughput benchmark:

```sh
go test ./internal/core -run '^TestWhitespaceTokenizer'
go test ./internal/core -run '^$' -fuzz '^FuzzWhitespaceTokenizer$'
go test ./internal/core -run '^$' -bench '^BenchmarkWhitespaceTokenizer$' -benchmem
```
