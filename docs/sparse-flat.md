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

The extended sparse query contract applies candidate filters and inclusive IP
radius thresholds during the scan. The collection facade supplies live-version
selection and projected document materialization, while group-by independently
retains a bounded top-k for each scalar group.

An FP32 collection field may store FP16-rounded first-stage values. With
`UseRefiner`, Sparse Flat requests `floor(TopK*ScaleFactor)` candidates
without approximate radius pruning, then reloads the unmodified sparse vectors
and query to recompute exact inner products, filter, radius, and top-k. The same
path is available to Query and MultiQuery and is reconstructed identically
after Optimize or reopen.
