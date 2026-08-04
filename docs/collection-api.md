# Collection API

The root `zvec` package exposes the v0.3 native collection milestone. It is a
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
field. Flat search is exact. Dense HNSW and IVF and sparse inner-product HNSW
use their matching native Go runtimes; an explicit `Linear` query scans the
matching Flat representation for truth comparisons. Query parameters expose
EF or NProbe, metric-aware radius, SQL scalar filters, projection, and bounded
HNSW cache warming. Dense FP16, INT8, and INT4 scalar-code scoring and optional
INT8/INT4 rotation are supported where schema validation permits them. Dense
queries can rerank retained candidates with original vectors. Sparse refinement
and IVF SOAR return `ErrNotSupported`.

`GroupByQuery` retains a top-k per filtered scalar group and ranks groups by
their best document. HNSW/IVF group-by currently requires explicit `Linear`;
quantized or refined group-by remains unsupported. The library never silently
substitutes a different algorithm.

`CreateIndex` atomically publishes implemented Flat, HNSW, IVF, and INVERT
parameters after full-snapshot validation. `DropIndex` atomically clears scalar
metadata or restores vector fields to Flat/IP. `AddColumn` atomically installs
supported numeric fields and backfills the live snapshot. `AlterColumn`
atomically renames or replaces basic numeric fields, and `DropColumn`
atomically removes them. `Optimize` atomically rewrites the current live
snapshot, compacts contiguous document-ID runs up to the schema segment limit,
reclaims deleted and superseded versions, and prunes obsolete native segment,
WAL, and snapshot files. It accepts the implemented Flat/HNSW/IVF and scalar
INVERT definitions, including scalar-quantized and rotated vector definitions.

Collection ANN indexes are currently rebuilt from the durable live snapshot
for each query and DDL validation. The standalone checksummed IVF/HNSW formats
are not yet collection-segment artifacts. This preserves deterministic reopen
behavior but makes runtime index construction part of current query latency.

WAL-backed mutations survive `Close` without `Flush`. `Flush` atomically
publishes an immutable segment and rotates the WAL. `Open` can acquire either
the sole writer lock or one of multiple read-only locks. `Destroy` is available
only from a writable handle and removes the collection directory after closing
its files.
