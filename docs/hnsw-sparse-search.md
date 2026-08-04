# Sparse HNSW search

This independently gated v0.3 unit makes the CSR-backed sparse HNSW graph
searchable through the common `SparseSearcher` and `SparseQuerySearcher`
contracts. Persistence/reopen is now implemented by the next unit;
incremental writes and collection routing are also implemented by later units.

## Query controls

Sparse graph queries use `HNSWSearchOptions`: positive TopK, optional
inner-product Radius minimum, candidate Filter, and EF in `[1, 2048]`. The
pinned default EF is 300. If EF is below TopK, candidate retention grows to
TopK so the request is not silently underfilled.

Canonical sparse queries may be empty. Coordinate/value lengths must match,
coordinates must be strictly increasing, and values must be finite. A finite
input whose accumulated inner product cannot be represented as float32 returns
an explicit score error.

## Execution

Graphs with at most 1,000 vectors use an exact CSR scan, matching Sparse Flat
for filtering, radius admission, score/key ordering, and ties. Larger graphs
greedily descend upper levels, then run best-first level-zero traversal with
the original sparse FP32 values. Rejected filter/radius nodes remain eligible
for graph traversal; they are excluded only from returned candidates.

Inner product ranks larger scores first. Traversal ties use stable node
positions, while returned result ties use ascending document keys.

Collection sparse HNSW fields may use FP16-rounded first-stage values.
`UseRefiner` reloads retained original sparse vectors for the returned graph
candidates, recomputes exact inner products from the unrounded query, and
applies the final filter, radius, and top-k. Sparse HNSW has no public scale
factor at the pinned baseline, so refinement preserves the graph candidate
count instead of silently widening it.

Tests cover exact Sparse Flat parity below the threshold, empty vectors and
stable ties, validation and cancellation, EF below TopK, selective filter plus
radius traversal, original-vector refinement, score overflow, recall@10 of at
least 0.80 above the threshold, and a 10,000-vector search benchmark.
