# Collections

The root `xvec` package exposes a pure-Go embedded collection. I/O, mutation,
query, and maintenance methods accept `context.Context`; schema, option, path,
and in-memory statistic getters do not.

## Create and open

```go
schema := xvec.NewCollectionSchema("books",
    xvec.NewField("title", xvec.DataTypeString),
    xvec.FieldSchema{
        Name: "embedding", DataType: xvec.DataTypeVectorFP32, Dimension: 768,
        Index: xvec.NewHNSWIndexParams(xvec.MetricTypeCosine),
    },
)

collection, err := xvec.CreateAndOpen(
    ctx, "./data/books", schema, xvec.NewCollectionOptions(),
)
```

`CreateAndOpen` creates a native Go collection and acquires its sole writable
handle. `Open` acquires either that writer lock or, with `ReadOnly`, one of
multiple shared reader locks. The Go format is intentionally incompatible with
C++ zvec collections.

`Schema`, `Options`, and `Stats` return independent point-in-time values. `Path`
returns the absolute collection directory.

## Documents and writes

A document has a UTF-8 primary key and a map whose values must use the exact Go
types declared by xvec. Dense and sparse vector fields use the explicit vector
types rather than ordinary slices where the two would be ambiguous.

`Insert`, `Upsert`, `Update`, and `Delete` return one `WriteResult` per input in
input order. A mixed batch continues after document-local failures. Its
non-nil `BatchWriteError` unwraps every cause for `errors.Is` and `errors.As`.

- `Insert` requires a complete document and rejects an existing key.
- `Update` requires an existing key and changes only supplied fields.
- `Upsert` partially updates an existing key but requires a complete document
  for a new key.
- `Delete` removes current versions of the supplied keys.
- `DeleteByFilter` removes every live document for which a valid SQL predicate
  evaluates to `TRUE`. A valid no-match filter succeeds. Cancellation may leave
  an already committed prefix, consistently with other batch mutations.

Writes append to the outer collection WAL before changing in-memory state.
`WALSyncEvery` optionally synchronizes after a number of successful records;
zero disables that automatic boundary. `Flush` and `Close` always synchronize
pending records.

## Fetch, projection, and iteration

`Fetch` preserves requested-key order and returns `nil` for a missing or deleted
key. `Projection` has two independent controls:

- `OutputFields == nil` selects every scalar and array field;
- a non-nil empty `OutputFields` selects none;
- a non-empty list selects named scalar and array fields; and
- `IncludeVectors` includes every vector field.

`CreateIterator` captures an isolated live-document snapshot. Later writes and
deletes are not visible. Documents are decoded and projected lazily, and
`Next` returns `io.EOF` at the end. Close the iterator when finished. While an
iterator is open, schema/index changes, `Optimize`, collection `Close`, and
`Destroy` return `ErrFailedPrecondition`; ordinary writes and `Flush` remain
available.

## Index and schema changes

`CreateIndex` validates and backfills the complete live snapshot before it
atomically publishes the new schema. Implemented definitions include:

- Flat and HNSW on supported dense and sparse vector fields;
- IVF, IVF-RaBitQ, Vamana, and DiskANN on their supported dense fields;
- INVERT on filterable scalar and array fields; and
- FTS on string fields.

`DropIndex` clears scalar index metadata or restores a vector field to the
baseline unquantized Flat/IP definition. Existing document values do not
change.

`AddColumn` adds and backfills a basic numeric field from a baseline-compatible
arithmetic expression. An empty expression is valid only for a nullable field
and writes explicit NULL values.

```go
field := xvec.FieldSchema{
    Name: "adjusted", DataType: xvec.DataTypeInt64, Nullable: false,
}
if err := collection.AddColumn(ctx, field, "score + 10", xvec.AddColumnOptions{}); err != nil {
    return err
}
```

`AlterColumn` either renames a basic numeric field or replaces its numeric type,
name, nullability, and optional INVERT parameters. Rename and replacement forms
are mutually exclusive. `DropColumn` removes a basic numeric field and rewrites
live payloads. Unsupported field kinds are rejected rather than partially
migrated.

All DDL operations are serialized with writes and publish atomically. Failure
before the manifest commit leaves the previous schema and data authoritative.
Read-only handles return `ErrPermissionDenied`.

## Flush, optimize, close, and destroy

`Flush` synchronizes the WAL and, when the writing segment is non-empty,
publishes an immutable segment and starts a new WAL. Repeated queries cache
immutable vector, FTS, and INVERT state per segment.

`Optimize` normally rewrites the complete live snapshot into contiguous
document-ID runs bounded by `MaxDocsPerSegment`. A fresh append-only collection
with one mutable segment and no deletions is already canonical, so Optimize
publishes that segment directly instead of re-encoding every payload. Both
paths build implemented indexes and prune obsolete package-owned artifacts.
The rewrite path reclaims deleted and superseded versions; live document IDs
and the next monotonic ID are preserved.

`Close` is idempotent, synchronizes pending WAL records, and releases files and
the collection lock. `Destroy` is available only on a writable handle; it
closes the collection and recursively removes only its validated collection
directory.

See [Storage and recovery](storage.md) for the commit and crash boundaries.
