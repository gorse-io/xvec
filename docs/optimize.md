# Optimize

`Collection.Optimize(ctx, options)` atomically compacts the current live
snapshot. It removes logically deleted documents and superseded versions,
groups each contiguous document-ID run into immutable segments no larger than
the schema's `MaxDocsPerSegment`, writes fresh primary-key and empty deletion
snapshots, and rotates to a new empty WAL.

```go
if err := collection.Optimize(ctx, zvec.OptimizeOptions{Concurrency: 4}); err != nil {
    return err
}
```

Document IDs and primary keys do not change. Gaps left by removed versions are
preserved, and the next writable document ID remains monotonic even when the
highest historical document was deleted. Queries and Fetch therefore return
the same live documents before and after optimization. A fully deleted
collection is rewritten with no immutable segments but still retains its next
document ID.

The v0.2 implementation supports fields with unquantized Flat vector indexes,
scalar INVERT indexes whose value type is implemented by the filter runtime,
and fields without indexes. If compaction is required and the schema contains
HNSW, IVF, RaBitQ, Vamana, DiskANN, FTS, quantized/rotated Flat, or binary
INVERT state, Optimize returns `ErrNotSupported` before publishing anything.
Those algorithms are enabled by their later milestones rather than silently
rebuilt as a different index.

`CURRENT` is the commit point. Failure before publication leaves the old
manifest, schema, segments, snapshots, and WAL authoritative. Once publication
succeeds, recovery sees the complete compacted version. Optimize then removes
only unreferenced files matching zvec's native segment, WAL, WAL-lock, and
snapshot naming schemes; it preserves unknown files and manifest generations.
A crash during this cleanup can leave harmless garbage, and a later Optimize
retries pruning even when no new compaction is needed.

An already canonical collection is a manifest no-op. `Concurrency` controls
parallel payload validation/copying; zero selects the library default and a
negative value is invalid. Optimize requires a writable handle and honors
context cancellation throughout rewriting and cleanup.
