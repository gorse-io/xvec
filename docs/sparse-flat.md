# Sparse Flat index

Sparse Flat is the exact sparse-vector implementation in `internal/core`. It
stores canonical FP32 vectors in compressed sparse row form: document keys and
row offsets identify contiguous coordinate and value arrays. Inputs are cloned,
coordinates must be strictly increasing, and values must be finite.

The index implements sparse `Provider`, `Searcher`, `Streamer`, and one-shot
`Builder` contracts. It accepts incremental documents after build and permits
concurrent exact searches while serializing additions. Primary keys are unique
within an index.

Only inner product is supported, matching the pinned public schema behavior for
sparse vectors. Search merges each candidate's coordinates with the query in
linear time, retains O(k) ranking state, and returns highest scores first with
ascending document key as the deterministic tie-breaker. Empty sparse vectors
are valid and score zero.

Deletion masks, scalar filters, radius, and collection-level result shaping are
applied by the query executor rather than this storage-agnostic index.
