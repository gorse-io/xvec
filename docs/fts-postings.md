# FTS dictionary, postings, and statistics

The v0.5 FTS storage primitive builds an immutable term dictionary from
already-analyzed token streams. It preserves the behavior needed by zvec
commit `58375ff`: dense segment-local document IDs, empty documents in corpus
counts, per-document term frequency and document length, ordered positions,
maximum term frequency, byte-lexical terms, and exact deletion-aware
cross-segment document frequencies.

This is a new Go-native format. It does not read or write the baseline's
RocksDB column families or C++ BitPacked serialization.

## Building a dictionary

`FTSFieldBuilder.AddDocument` accepts document IDs `0, 1, ...` in order and an
already-tokenized `[]Token`. Dense IDs intentionally share the forward row
domain used by a segment. Token positions must be nondecreasing; gaps and
duplicate positions remain representable. Empty terms are byte strings like
any other term, while an empty token list still adds a zero-length document to
the statistics.

```go
builder := core.NewFTSFieldBuilder()
err := builder.AddDocument(ctx, 0, []core.Token{
    {Text: "go", Position: 0},
    {Text: "vector", Position: 1},
    {Text: "go", Position: 2},
})
dictionary, err := builder.Build(ctx)
```

`Build` creates a point-in-time snapshot without consuming the builder. The
dictionary is immutable and safe for concurrent lookup, prefix enumeration,
encoding, and posting iteration. `Lookup` returns document frequency,
maximum term frequency, and an independent iterator. `Prefix` uses the sorted
term table and is ready for the later wildcard-query unit.

## Native format

Each posting list has a 48-byte versioned header, a skip directory, compressed
blocks, and a position section. A block contains at most 128 documents and
independently bitpacks:

- document-ID deltas, with an absolute minimum ID in the block header;
- term frequencies;
- document lengths; and
- byte lengths of the corresponding position payloads.

Positions use unsigned varint deltas. The skip directory records the last
document ID, block offset, and position offset for every block, allowing
`Advance(target)` to binary-search the right block before decoding it. Iterators
decode only one block at a time; position bytes are decoded only when requested.

The outer dictionary has a 64-byte versioned header, the exact document-length
table, a byte-prefix-compressed term table, and concatenated independently
checksummed posting lists. Both levels carry declared lengths, reserved-field
checks, a payload CRC32C, and a header CRC32C. Open rejects truncation, unknown
versions, overlapping sections, impossible allocation counts, invalid bit
widths, nonmonotonic IDs or terms, malformed varints, inconsistent inline
document lengths, and trailing bytes. Encoders and readers own their bytes.

## Cross-segment statistics

`AggregateFTSCorpusStats` accepts immutable dictionaries plus optional
segment-local deletion bitmaps. It snapshots each bitmap, then calculates:

- live document count, including live empty documents;
- total tokens and average document length; and
- live document frequency for every term across all segments.

Deleted-only terms disappear from the result. A deletion bit outside the
dictionary's document domain is an error. Returned term slices and frequency
maps are independent copies.

All building, encoding, opening, and aggregation paths accept
`context.Context`; cancellation returns no partial artifact. Invalid caller
input and corrupt bytes match `ErrInvalidFTSDocument`,
`ErrInvalidFTSPosting`, `ErrInvalidFTSDictionary`, `ErrInvalidFTSStats`,
`ErrCorruptFTSPosting`, or `ErrCorruptFTSDictionary` through `errors.Is`.

The compatibility fixture records the pinned C++ posting/indexer/reducer
source identities and validates term aggregation, tf, document length,
positions, empty-document statistics, and position delta encoding. Additional
tests cover 128-document boundaries, skip seeks, snapshot ownership,
front-coding, deletion-aware multi-segment totals, concurrent readers,
structural corruption with repaired checksums, fuzzing, and benchmarks.

BM25 scoring and block-max bounds, physical segment merging, and Collection
integration remain later v0.5 units. Query lexing, AST canonicalization, and
term/phrase/boolean execution are documented separately. No placeholder score
is stored in this format.

```sh
go test ./internal/core -run '^TestFTS|^TestAggregateFTS'
go test ./internal/core -run '^$' -fuzz '^FuzzFTSPostingListOpen$'
go test ./internal/core -run '^$' -fuzz '^FuzzFTSTermDictionaryOpen$'
go test ./internal/core -run '^$' -bench '^BenchmarkFTS' -benchmem
```
