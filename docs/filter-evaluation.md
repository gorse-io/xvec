# Scalar filter evaluation

The v0.2 filter runtime in `internal/db/sql` evaluates schema-bound predicates
without reparsing text or coercing values during a document scan. Its `Value`
model preserves every scalar width (`INT32` versus `INT64`, for example),
distinguishes arrays, clones binary data, and represents typed NULL values.
The following analyzer milestone binds public schema and document types into
these runtime values.

Comparisons cover binary, string, Boolean, signed and unsigned integer, and
finite floating-point fields. Binary and string ordering is bytewise. Boolean
fields accept equality and inequality only. `IN` compares a scalar with a
non-empty, homogeneous set. Width and signedness are never silently changed;
the analyzer is responsible for parsing a literal into the exact field type
and reporting overflow.

NULL uses SQL three-valued logic. Ordinary comparisons, `LIKE`, `IN`, and
contain predicates return `UNKNOWN` for NULL input, and a filter retains only
`TRUE`. `AND`, `OR`, and negation implement Kleene truth tables. `IS NULL` and
`IS NOT NULL` are the only predicates that turn NULL directly into a Boolean
result. A missing nullable document field will be bound as typed NULL.

`CONTAIN_ALL` and `CONTAIN_ANY` compare homogeneous array elements by exact
value. The query set is limited to the baseline maximum of 32 values.
Repetition in either side does not change membership. Baseline empty-set
semantics are explicit:

| Predicate | Result for a non-NULL array |
| --- | --- |
| `CONTAIN_ALL ()` | `TRUE` |
| `NOT CONTAIN_ALL ()` | `FALSE` |
| `CONTAIN_ANY ()` | `FALSE` |
| `NOT CONTAIN_ANY ()` | `TRUE` |

`LIKE` is anchored to the whole string. An unescaped `%` matches any sequence
of Unicode code points and `_` matches exactly one code point; backslash makes
the next character literal. The compiled matcher uses Go's linear-time RE2
engine and classifies patterns as exact, match-all, prefix, suffix, contains,
or general. Forward execution supports every form. The classification will
let the inverted-index milestone accelerate supported wildcard forms while
falling back to the same forward semantics for general patterns.
