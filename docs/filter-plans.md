# Schema analysis and filter plans

`VectorQuery.Filter` and `GroupByVectorQuery.Filter` execute SQL-style scalar
predicates for both dense and sparse Flat searches. A filter is parsed and
analyzed once per query, evaluated against the consistent live document
snapshot, and supplied to the core searcher as a candidate mask before distance
scoring, radius selection, top-k ranking, or grouping.

Analysis resolves every identifier against the collection schema and converts
literals directly to the field's exact type. Signedness and width are checked;
overflow, mixed list types, invalid Boolean operators, missing fields, vector
fields, and FTS fields produce `ErrInvalidArgument` with the source position.
Binary/string scalars, Booleans, all integer and floating-point scalars, and
the corresponding arrays are represented. Baseline rules restrict arrays to
contain or NULL predicates, prohibit binary `IN`/contain sets, and limit
contain sets to 32 values.

The supported scalar function is `array_length(array_field)`. It returns a
nullable `UINT32` and accepts the six comparison operators with an integer
literal. Function names follow the pinned baseline and are case-sensitive.

The rewriter safely coalesces OR-connected equality/IN terms for the same
field into one IN set. It deliberately does not reproduce the pinned native
inequality-OR rewrite, because that transformation changes results. Empty
contain sets are normalized to `IS NOT NULL` or constant false according to
the pinned behavior, after which the optimizer folds constant AND/OR branches.
Plans are immutable and may be evaluated concurrently.

NULL comparisons use the three-valued rules in
[Scalar filter evaluation](filter-evaluation.md). A missing nullable field is
treated as typed NULL. Only `TRUE` candidates proceed to vector scoring. This
also means filtered group-by searches never create groups from rejected
documents. Filter results are deterministic across Flush and reopen because
they operate on decoded native document values rather than index-specific text
encodings.

This milestone uses forward evaluation. Scalar inverted-index selection and
range/wildcard acceleration are the next independent unit; adding those paths
must preserve the same plan results.
