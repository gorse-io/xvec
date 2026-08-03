# Collection API

The root `zvec` package exposes the first usable collection milestone. It is a
pure-Go embedded database: every I/O, write, and query method accepts a
`context.Context`; schema, options, path, and in-memory statistics getters do
not.

```go
schema := zvec.NewCollectionSchema("books",
    zvec.FieldSchema{Name: "title", DataType: zvec.DataTypeString},
    zvec.FieldSchema{
        Name: "embedding", DataType: zvec.DataTypeVectorFP32, Dimension: 768,
        Index: zvec.NewFlatIndexParams(zvec.MetricTypeCosine),
    },
)
collection, err := zvec.CreateAndOpen(ctx, path, schema, zvec.NewCollectionOptions())
```

`Insert`, `Upsert`, `Update`, and `Delete` return one `WriteResult` per input.
A mixed batch continues past document-local errors, and its non-nil
`BatchWriteError` unwraps all structured causes for `errors.Is` and
`errors.As`. Insert requires complete documents. Update changes only supplied
fields. Upsert uses partial-update semantics for an existing key and requires a
complete document for a new key.

`DeleteByFilter` schema-binds the same SQL predicate used by queries, selects
only current live versions under the collection write lock, and deletes each
match through the WAL. A valid no-match filter succeeds. Completed deletions
survive reopen without Flush; cancellation can return after an already
committed prefix, consistently with other batch mutations.

`Fetch` preserves key order and represents a missing key with a nil document.
Its `Projection` separates scalar selection from vector inclusion: nil output
fields select all scalar fields, an empty non-nil slice selects none, and
`IncludeVectors` controls all vector fields.

`Query` accepts either an explicit dense or sparse vector matching the target
field. Flat search is exact, including metric-aware radius limits,
schema-analyzed SQL scalar filters, and deterministic document-ID tie breaking.
`GroupByQuery` retains a top-k per filtered scalar group and ranks groups by
their best document. ANN and quantized indexes return `ErrNotSupported` until
their stated milestones; the library never silently substitutes a different
algorithm. `CreateIndex` is the first installed DDL operation: it atomically
publishes implemented Flat and INVERT parameters, while later index algorithms
still return `ErrNotSupported`. DropIndex, column DDL, and Optimize remain
`ErrNotSupported` until their independent v0.2 units are installed.

WAL-backed mutations survive `Close` without `Flush`. `Flush` atomically
publishes an immutable segment and rotates the WAL. `Open` can acquire either
the sole writer lock or one of multiple read-only locks. `Destroy` is available
only from a writable handle and removes the collection directory after closing
its files.
