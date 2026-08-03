# AddColumn

`Collection.AddColumn(ctx, field, expression, options)` adds one field and
backfills every live document before publishing the new schema. The installed
v0.2 unit accepts the six basic numeric types `INT32`, `INT64`, `UINT32`,
`UINT64`, `FLOAT`, and `DOUBLE`. Arrays, strings, booleans, vectors, and FTS
columns are rejected rather than silently approximated. Numeric fields may
include implemented INVERT parameters.

For a non-nullable field, `expression` is required. Expressions support numeric
literals, existing numeric field references, parentheses, unary `+` and `-`,
and binary `+`, `-`, `*`, and `/` with conventional precedence. Results are
cast to the new field type; integer overflow and floating-to-integer truncation
match the permissive cast used by the pinned baseline. NULL input propagates to
NULL and therefore cannot backfill a non-nullable field. Division by zero,
unknown or nonnumeric fields, invalid syntax, and non-finite results fail the
operation.

An empty expression is accepted only for a nullable field and writes an
explicit NULL into every existing document. When the collection has no live
documents there is no expression to evaluate, matching the baseline's deferred
backfill behavior. The expression is not a default for later writes: subsequent
Insert and Upsert calls must still satisfy the published schema themselves.

```go
field := zvec.FieldSchema{
    Name: "adjusted", DataType: zvec.DataTypeInt64,
    Index: zvec.NewInvertIndexParams(),
}
err := collection.AddColumn(ctx, field, "(rating * 2) + 1",
    zvec.AddColumnOptions{Concurrency: 4},
)
```

`Concurrency` bounds parallel expression evaluation; zero selects the library
default and a negative value is invalid. The mutation holds the collection
write lock, rewrites the complete live snapshot into new immutable segments,
and publishes the data and schema through one `CURRENT` commit point. Existing
live document IDs are preserved. Superseded versions and deletion records are
reclaimed as part of the rewrite. Cancellation or any validation, evaluation,
I/O, or publication failure before that commit leaves the prior schema and
documents visible after both continued use and reopen.
