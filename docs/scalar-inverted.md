# Scalar inverted candidate indexes

Scalar fields configured with `NewInvertIndexParams()` participate in indexed
filter candidate selection. Exact postings are available for scalar equality,
inequality, `IN`, NULL checks, array contain predicates, and `array_length`.
Numeric and lexicographic range comparisons use a sorted term dictionary when
`EnableRangeOptimization` is true (the default); disabling it preserves results
and selects a full term scan.

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

At this v0.2 development point, postings are built in memory from the
collection's consistent live-document snapshot for each filtered query. They
are not a new on-disk format and require no migration; Flush and reopen produce
the same results. DDL persists the INVERT parameters in the schema, and
Optimize atomically compacts the live data from which exact postings are
rebuilt; no separate posting file can become stale.
