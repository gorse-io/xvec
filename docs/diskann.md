# DiskANN index

The v0.4 DiskANN implementation is a pure-Go disk graph for dense FP32 and
FP16 inputs, stored through their FP32 scoring view. Its behavioral baseline
is zvec commit `58375ff`, while its files are new native Go artifacts and are
not compatible with the C++ DiskANN format.
It uses ordinary `io.ReaderAt`, so the same implementation works without CGO
on Linux, macOS, and Windows.

## Construction

`DiskANNBuilder` validates and owns unique key/vector pairs. Build first uses
the native Vamana RobustPrune implementation to produce a bounded directed
graph and medoid, then serializes each original vector and its neighbors into
the sector layout described in [DiskANN storage and I/O](diskann-storage.md).
The public `MaxDegree` and construction `ListSize` control graph density;
construction raises its internal Vamana list width to `MaxDegree` when the
baseline permits a smaller configured list.

Every non-empty index trains the pinned 8-bit product quantizer. `PQChunks=0`
selects half the vector dimension; an explicit value must be between one and
the dimension. L2 and inner product use their direct PQ tables. Cosine
normalizes vectors and queries into an L2 traversal space. MIPS-L2 combines PQ
L2 with the norm of each reconstructed code. These scores order the frontier
only: expanded nodes always receive the exact public score from their original
FP32 vector.

Builders are one-shot and cancellation-aware. Empty indexes are valid and
carry no PQ model. Duplicate keys, non-finite or wrong-dimension vectors,
overflowing node counts, and invalid options fail explicitly.

## Search and cache

`SearchDiskANN` starts at the persisted medoid and retains the best
`max(TopK,ListSize)` PQ candidates. It expands candidates in best-first
batches whose total multi-sector footprint respects the pinned 128-sector
limit. Graph filters do not block traversal: filter and metric-aware radius
are applied only after the expanded node has been read and scored exactly.
Equal scores use ascending external keys.

`Linear` reads every node in bounded batches and is the deterministic exact
truth path. `DiskANNIndex` is also a `DenseProvider`; `OriginalVectorRefiner`
can reread retained original vectors and apply exact final top-k/radius logic.
Collection `UseRefiner` requests ten times TopK before this final rerank.

The node cache is a concurrency-safe bounded LRU. `WarmCache` follows graph
edges breadth-first from the medoid and never exceeds the configured cache
capacity. Cache hits never expose mutable vector or neighbor aliases. Search,
provider reads, cache warming, Save, and Close coordinate so Close waits for
active readers and later I/O returns `ErrDiskANNClosed`.

## Complete file format

`DiskANNIndex.Save` atomically publishes one file with:

1. a versioned 4096-byte index header;
2. external keys;
3. PQ chunk offsets and the full 256-by-dimension pivot matrix;
4. one byte per node and chunk;
5. zero alignment padding; and
6. an embedded, independently versioned sector node artifact.

The header records every exact section offset and length, metric mapping,
graph options, PQ configuration, entry point, five section CRC32C values, and
its own CRC32C. Open derives the expected layout instead of trusting offsets,
rejects truncation and trailing bytes, checks zero reserved/padding bytes,
validates key uniqueness and PQ state, verifies every section and the embedded
node file, and retains the `os.File` until Close. Per-record CRC remains a
second random-read integrity boundary after open.

## Collection integration

Dense collection fields accept `DiskANNIndexParams` and
`DiskANNQueryParams`. Query, filters, radius, projection, Linear, refinement,
CreateIndex backfill, DropIndex, Optimize, Stats completeness, and Close/reopen
are connected. Until ANN artifacts become segment-native, Collection rebuilds
the deterministic runtime index from its durable live-document snapshot.

Public FP16/INT8/INT4 scalar quantization on a DiskANN field remains
`NotSupported`: DiskANN's mandatory PQ is controlled separately by
`PQChunks`, and silently treating the two representations as equivalent would
change schema semantics.

## Verification

Tests cover all four dense metrics, exact full-list parity, approximate
recall, filter/radius behavior, refinement, cache ownership and preloading,
concurrent reads, empty indexes, cancellation and Close, atomic save/reopen,
replacement, truncation, trailing data, header/section/record corruption,
fuzzed complete files, race detection, cross-platform builds, and a warm-cache
search benchmark:

    go test ./internal/core -run '^TestDiskANN'
    go test ./internal/core -run '^$' -bench '^BenchmarkDiskANNSearchWarmCache$' -benchmem
    go test ./internal/core -run '^$' -fuzz '^FuzzDiskANNIndexFile$'
