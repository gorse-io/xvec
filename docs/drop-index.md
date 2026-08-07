# DropIndex

`Collection.DropIndex(ctx, column)` atomically publishes a schema without the
column's current custom index definition. Scalar fields become unindexed.
Vector fields return to the baseline default, unquantized Flat index with the
inner-product metric, so dropping an ANN, quantized, or custom-metric index does
not make vector queries unavailable.

Dropping an already unindexed scalar or default-Flat vector is an idempotent
success and does not publish a manifest generation. A missing field returns
`ErrNotFound`, an empty name returns `ErrInvalidArgument`, and a read-only
handle returns `ErrPermissionDenied`.

```go
if err := collection.DropIndex(ctx, "rating"); err != nil {
    // Handle the structured zvec error.
}
```

The collection write lock excludes writes and queries until validation and
publication finish. When restoring a vector field to Flat/IP, DropIndex
backfills the complete live snapshot before the commit point. A backfill,
encoding, cancellation, or pre-commit publication failure leaves both the
in-memory schema and CURRENT manifest unchanged. Scalar INVERT/FTS definitions
and later vector definitions can be removed because no replacement artifact is
needed.

Flat, HNSW, HNSW-RaBitQ, IVF, Vamana, DiskANN, FTS, and INVERT collection
runtime structures follow the new schema immediately. Existing per-segment
artifacts carry the old schema hash and cannot be selected; a later Flush or
Optimize publishes the replacement metadata and prunes obsolete files. The
schema transition is durable without Flush and remains in effect after
Close/reopen. Standalone native index artifacts are independent of collection
ownership.
