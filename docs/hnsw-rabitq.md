# HNSW-RaBitQ index

The v0.4 HNSW-RaBitQ implementation combines a deterministic HNSW graph with
portable RaBitQ codes for L2, inner-product, and cosine search. Its behavioral
baseline is zvec commit `58375ff`; its file and code layouts are native Go and
are intentionally not compatible with C++ zvec artifacts.

## Construction

`HNSWRaBitQBuilder` validates and owns FP32 vectors, trains one RaBitQ model,
builds HNSW topology from the retained original vectors, and encodes one code
for each graph position. Building topology from originals matches the pinned
implementation and keeps graph quality independent of quantization error.

Builds are deterministic for a fixed seed, including across worker counts.
Dimensions are limited to 64–4095, `TotalBits` to 1–9, and metrics to L2, IP,
or cosine. Empty builds use a deterministic zero-centroid model so a newly
indexed empty Collection can accept later vectors without retraining its model.

`Add` encodes a vector with the fixed model, updates topology, and publishes a
complete copy-on-write generation. A canceled or failed addition leaves the
visible graph and codes unchanged. Searches and saves may run concurrently
with additions and observe either complete generation.

## Search and refinement

Upper graph layers compare one-bit estimates. At the base layer, the one-bit
probabilistic lower bound rejects candidates that cannot improve the retained
set before the full code is evaluated. Graph nodes excluded by a filter remain
available as traversal bridges. Public scores preserve metric conventions:
L2 and cosine are lower-is-better, while IP is the estimated dot product and is
higher-is-better.

`HNSWRaBitQSearchOptions` provides:

- `EF` for the base-layer candidate budget;
- metric-aware `Radius` and a key `Filter` through `SearchOptions`;
- `Linear` for an exhaustive full-code scan; and
- `Refine` for exact reranking from retained original FP32 vectors.

Refinement requests up to `max(TopK, EF)` approximate candidates, then applies
the filter, radius, and final top-k using exact original-vector distances. A
linear refined query considers every live vector and therefore matches Flat
search for the same input, apart from Collection-internal document IDs.
Graphs at or below the shared small-index threshold also use a full-code scan
to provide deterministic small-collection behavior.

Collection maps `HNSWRaBitQQueryParams.EF`, `QueryOptions.Linear`, and
`QueryOptions.UseRefiner` directly to these controls. CreateIndex validates a
concurrent snapshot backfill before the schema version is published; Optimize,
Stats, filters, radius, and reopen use the same index semantics. Group-by over
RaBitQ codes is available with explicit Linear execution, and `UseRefiner`
switches that full scan to exact original-vector group scores.

## Native persistence

`Save` atomically publishes a single native artifact containing:

- an embedded, independently checksummed native HNSW graph;
- RaBitQ centroids, rotation signs, scale, options, and model fingerprint; and
- fixed-size split-code records aligned with graph positions.

The outer `ZVECHRBQ` header carries an explicit format version, total and
payload lengths, and CRC32C checksums for both header and payload. Open rejects
truncation, trailing bytes, unsupported versions, checksum failures,
inconsistent graph/model metadata, bad fingerprints, invalid factors, and
out-of-range centroid assignments. Reopened artifacts preserve topology,
scores, and incremental-add behavior.

This artifact is available to the internal index API. Collection storage does
not yet persist segment-native ANN artifacts; it rebuilds the runtime
HNSW-RaBitQ index from the durable live document snapshot for query and DDL
validation. No operation silently substitutes Flat or another index.

## Verification

Tests cover all supported metrics, deterministic topology, approximate and
refined recall floors, filters, radius, linear exactness, empty and incremental
indexes, cancellation atomicity, concurrent add/search/save/open, byte-identical
reopen behavior, corruption and truncation, fuzzing, and Collection lifecycle
integration:

```sh
go test ./internal/core -run '^(TestHNSWRaBitQ|ExampleHNSWRaBitQBuilder)'
go test ./internal/core -run '^$' -bench '^BenchmarkHNSWRaBitQSearch$' -benchmem
go test ./internal/core -run '^$' -fuzz '^FuzzHNSWRaBitQDecode$'
```
