# Scalar inverted candidate indexes

Scalar fields configured with `NewInvertIndexParams()` participate in indexed
filter candidate selection. Exact postings are available for scalar equality,
inequality, `IN`, NULL checks, array contain predicates, and `array_length`.
Numeric and lexicographic range comparisons use a sorted term dictionary when
`EnableRangeOptimization` is true (the default); disabling it preserves results
and selects a full term scan.

BINARY values use byte-exact posting keys and lexicographic byte ordering.
ARRAY_BINARY supports contain, NULL, and length candidates with the same
three-valued semantics as forward evaluation. SQL string literals supply the
bytes for equality, range, `IN`, and contain predicates; no UTF-8 normalization
is applied to stored binary values.

STRING `LIKE` routing follows the pinned native behavior:

- exact text and `prefix%` use postings without additional options;
- `%suffix` and `prefix%suffix` use the index only when
  `EnableExtendedWildcard` is true;
- `%contains%`, `_`, and patterns with multiple wildcard regions use forward
  evaluation.

Backslash escapes are interpreted by the shared LIKE compiler before a route
is selected. NULL values have their own bitmap and never appear in negated
comparison, `IN`, contain, range, or wildcard results.

Candidate composition is deliberately conservative. Both sides of an `OR`
must have a safe indexed candidate set before their union can be used. An
indexed side of `AND` may be used alone as a superset of the final matches.
Every selected row is then evaluated by the typed forward plan, so index
routing cannot change SQL three-valued semantics.

Postings are built per segment and reused by filtered Query, MultiQuery, and
GroupByQuery calls. Flush encodes a checksummed INVERT artifact for each newly
immutable segment and publishes its schema/segment identity in the manifest.
Reopen loads an exact match; when optional artifact metadata is absent, it
rebuilds postings from documents. Deleted or superseded versions are masked
and then forward-evaluated, so immutable postings never need mutation.
