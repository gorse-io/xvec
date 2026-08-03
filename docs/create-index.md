# CreateIndex

`Collection.CreateIndex(ctx, column, params, options)` validates an index
against the target field, backfills the complete current live-document
snapshot, and publishes the updated collection schema through a new immutable
manifest generation. The collection write lock covers all three phases, so
writes cannot cross the schema commit point.

The currently executable index types are:

- unquantized `FlatIndexParams` on supported dense or sparse vector fields;
- `InvertIndexParams` on filterable scalar and array fields other than BINARY.

ANN, quantized Flat, and FTS parameters return `ErrNotSupported` until their
algorithm milestone. A vector index on a scalar field, scalar index on a vector
field, invalid metric/type combination, nil parameters, or negative concurrency
returns `ErrInvalidArgument`. A missing column returns `ErrNotFound`. Different
non-vector index types cannot coexist on one column.

`CreateIndexOptions.Concurrency` bounds concurrent backfill validation. Zero
uses the runtime's default worker count. The operation is idempotent when the
column already has equal parameters; no new manifest is published in that
case. Supplying different parameters of the same implemented type rebuilds and
atomically replaces the definition.

```go
params := zvec.NewInvertIndexParams()
params.EnableExtendedWildcard = true
err := collection.CreateIndex(ctx, "title", params,
    zvec.CreateIndexOptions{Concurrency: 4},
)
```

The native v0.2 Flat and INVERT search structures remain snapshot-local: query
execution reconstructs them from live documents. CreateIndex persists their
validated parameters, not a C++-compatible index artifact. Existing WAL data,
new writes, Close/reopen, and later Flush all use the newly published schema.
If backfill, encoding, cancellation, or pre-commit manifest publication fails,
the previous in-memory and on-disk schema remains active. An error after the
atomic CURRENT replacement is reported, but the committed schema remains the
source of truth.
