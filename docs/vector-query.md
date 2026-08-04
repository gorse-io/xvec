# Exact vector query semantics

These semantics describe Flat execution and an ANN query with explicit
`Linear` enabled. HNSW and partial-probe IVF use the same score, radius,
filter, and tie rules, but their candidate set is approximate.

The core query layer applies the same controls to dense and sparse indexes.
`TopK` is required and positive. A zero radius disables range filtering;
otherwise L2, cosine, and MIPS-L2 retain scores less than or equal to the
radius, while inner product retains similarities greater than or equal to it.
The boundary is inclusive, matching the pinned engine's threshold behavior.

A candidate filter runs before scoring. Collection code uses this hook for
deleted-document masks and schema-analyzed SQL scalar predicates. Filtering
before ranking ensures the executor can still return up to `TopK` accepted
documents instead of losing slots occupied by rejected nearest neighbors.

Each segment returns an exact local top-k using O(k) ranking state. The query
orchestrator may search independent segments concurrently and then performs a
second O(k) merge. The merge is independent of segment order: the metric
chooses score direction and equal scores use ascending global document key.
Context cancellation propagates through segment search and merge setup.
