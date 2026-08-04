# Collection API

The root `zvec` package exposes the v0.5 native collection API. It is a
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
field. Flat search is exact. Dense HNSW, HNSW-RaBitQ, IVF, Vamana, and DiskANN
plus sparse inner-product HNSW use their matching native Go runtimes; an explicit
`Linear` query scans the matching representation for truth comparisons. Query
parameters expose EF or NProbe, metric-aware radius, SQL scalar filters,
projection, and bounded graph cache warming. Dense FP16, INT8, INT4, and RaBitQ
code scoring and optional INT8/INT4 rotation are supported where schema
validation permits them. Dense queries can rerank retained candidates with
original vectors. Sparse refinement and IVF SOAR return `ErrNotSupported`.

`GroupByQuery` retains a top-k per filtered scalar group and ranks groups by
their best document. ANN group-by currently requires explicit `Linear`;
quantized or refined group-by remains unsupported. The library never silently
substitutes a different algorithm.

`MultiQuery` evaluates two or more dense-vector, sparse-vector, or FTS
branches over one immutable live snapshot with a shared SQL filter. A nil
reranker selects reciprocal-rank fusion; weighted score fusion and a
context-aware callback adapter are also available. FTS branches use configured
tokenizers and filters, exact boolean/phrase execution, and deletion-aware BM25
statistics. See the [MultiQuery contract](multi-query.md) for candidate,
projection, and untrusted-reranker output rules.

`CreateIndex` atomically publishes implemented Flat, HNSW, HNSW-RaBitQ, IVF,
Vamana, DiskANN, and INVERT parameters after full-snapshot validation.
`DropIndex` atomically clears scalar metadata or restores vector fields to
Flat/IP. `AddColumn` atomically installs
supported numeric fields and backfills the live snapshot. `AlterColumn`
atomically renames or replaces basic numeric fields, and `DropColumn`
atomically removes them. `Optimize` atomically rewrites the current live
snapshot, compacts contiguous document-ID runs up to the schema segment limit,
reclaims deleted and superseded versions, and prunes obsolete native segment,
WAL, and snapshot files. It accepts the implemented Flat/HNSW/HNSW-RaBitQ/IVF,
Vamana/DiskANN, and scalar INVERT definitions. Scalar quantization and rotation
apply to the supported in-memory indexes; DiskANN uses its separate internal PQ.

Collection ANN indexes are currently rebuilt from the durable live snapshot
for each query and DDL validation. The standalone checksummed IVF, HNSW,
HNSW-RaBitQ, Vamana, and DiskANN formats are not yet collection-segment
artifacts. This preserves deterministic reopen behavior but makes runtime
index construction part of current query latency.

WAL-backed mutations survive `Close` without `Flush`. `Flush` atomically
publishes an immutable segment and rotates the WAL. `Open` can acquire either
the sole writer lock or one of multiple read-only locks. `Destroy` is available
only from a writable handle and removes the collection directory after closing
its files.

[`RuntimeConfig`](runtime-config.md) installs process-wide query and
maintenance admission, planner thresholds, conservative scratch-memory
budgeting, `slog` routing, worker bounds, and Jieba fallback resources. Runtime
and expanded collection statistics are concurrency-safe point-in-time views.
