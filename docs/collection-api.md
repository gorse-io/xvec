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

`Fetch` preserves key order and represents a missing key with a nil document.
Its `Projection` separates scalar selection from vector inclusion: nil output
fields select all scalar fields, an empty non-nil slice selects none, and
`IncludeVectors` controls all vector fields.

`Query` accepts either an explicit dense or sparse vector matching the target
field. v0.1 executes unquantized Flat search exactly, including metric-aware
radius limits and deterministic document-ID tie breaking. `GroupByQuery`
retains a top-k per scalar group and ranks groups by their best document. SQL
filters, ANN and quantized indexes return `ErrNotSupported` until their stated
milestones; the library never silently substitutes a different algorithm. The
DDL, DeleteByFilter, and Optimize entry points likewise return
`ErrNotSupported` until the v0.2 atomic metadata executor is installed.

WAL-backed mutations survive `Close` without `Flush`. `Flush` atomically
publishes an immutable segment and rotates the WAL. `Open` can acquire either
the sole writer lock or one of multiple read-only locks. `Destroy` is available
only from a writable handle and removes the collection directory after closing
its files.
