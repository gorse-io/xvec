# Dense HNSW search

This v0.3 unit makes the deterministic dense HNSW graph searchable through the
common `DenseSearcher` and `DenseQuerySearcher` contracts. Persistence/reopen
is now implemented by the next unit; incremental writes and collection
orchestration remain separate, so public HNSW collection queries are still
explicitly unsupported.

## Query controls

`HNSWSearchOptions` combines common `TopK`, metric-aware `Radius`, and
candidate `Filter` controls with `EF`. The pinned default EF is 300. Matching
the baseline, EF must be in `[1, 2048]`; when it is smaller than TopK,
execution raises the retained candidate capacity to TopK rather than silently
underfilling the request.

`Search` and `SearchWithOptions` use default EF. `SearchHNSW` accepts the
explicit value that later collection orchestration will obtain from
`HNSWQueryParams`.

## Execution

The pinned implementation uses brute-force scoring at or below 1000 vectors.
This preserves exact small-segment behavior and avoids graph overhead. Filter,
radius, original-vector scores, result order, and key tie-breaking therefore
match dense Flat exactly in that range.

Larger graphs execute two stages:

1. greedily descend from the highest-level entry point to level zero;
2. run best-first level-zero traversal while retaining up to
   `max(EF, TopK)` eligible candidates.

All comparisons use the configured metric and original FP32 vectors. Inner
product ranks larger scores first; L2, cosine distance, and MIPS-L2 rank smaller
scores first. Equal scores use ascending stable node position internally and
ascending document key in returned results.

Filters and radius checks govern result admission but do not mark a rejected
node as a graph barrier. Rejected nodes remain traversable, which is essential
for selective filters. If too few eligible results have been found, traversal
continues beyond ordinary EF termination until the frontier is exhausted or
enough competitive eligible candidates exist.

Tests cover exact Flat parity below the brute-force threshold for all four
metrics, deterministic ties, defaults, empty graphs, validation and
cancellation, EF below TopK, selective filter plus radius traversal, and
recall@10 of at least 0.80 against Flat truth above the graph threshold. A
10,000-vector query benchmark establishes the first latency baseline.
