# DiskANN storage and I/O

This v0.4 component provides the pure-Go random-access storage foundation for
the DiskANN builder and searcher. It follows the sector behavior of zvec
commit 58375ff but uses a new native Go format; it does not read C++ DiskANN
segments.

## Sector layout

Every node artifact begins with one 4096-byte header. The header contains an
eight-byte magic value, format version, total and data lengths, metric, node
count, dimension, maximum degree, derived record geometry, a CRC32C for the
complete node section, and a separate header CRC32C. Reserved bytes must be
zero, so future extensions cannot be mistaken for this version.

A fixed-size node record contains:

1. the original FP32 vector;
2. a uint32 outbound degree;
3. maximum-degree uint32 neighbor slots; and
4. a CRC32C over the preceding record bytes.

If a record fits in 4096 bytes, as many records as possible share each sector.
Trailing sector bytes remain zero. Larger records receive an integral number
of sectors per node. The derived layout is checked against the serialized
values on open, and node IDs are limited to uint32 just like the fixed
baseline.

Decode rejects truncated or trailing files, unsupported versions, non-zero
reserved bytes, inconsistent lengths or geometry, whole-section or per-record
checksum failures, non-finite vectors, excess degree, duplicate or out-of-
range neighbors, self-loops, and non-zero unused neighbor slots.

## Portable ReaderAt batching

ParallelReadAt accepts ordinary Go io.ReaderAt implementations, including
os.File on Linux, macOS, and Windows. Each request is filled exactly even when
a reader returns partial progress. Zero-progress and EOF-before-length reads
return an explicit short-read error. Results retain request order, worker
count is bounded by the shared parallel helper, and cancellation is checked
between every partial read.

DiskANNNodeReader verifies the complete data CRC during open. Batch node reads
deduplicate requests that share a packed sector, issue unique reads in
parallel, decode records into the caller's original ID order, and never return
mutable aliases of cached data.

## Cache

DiskANNNodeCache is a bounded, concurrency-safe least-recently-used cache.
Capacity zero disables storage. Get and Put deep-copy vectors and neighbor
lists, eviction is deterministic, and hit, miss, and eviction counters are
available as snapshots. The cache is intentionally independent of the later
BFS cache-selection policy, so the searcher can preload entry-point regions
or rely on demand caching without changing storage semantics.

The complete index container embeds this node artifact on a 4 KiB boundary.
See [DiskANN index](diskann.md) for graph construction, PQ traversal, exact
candidate scoring, persistence, and Collection integration.

## Verification

Tests cover packed-sector request deduplication, multi-sector nodes, empty
artifacts, partial and short ReaderAt behavior, checksum and semantic
corruption, cancellation, LRU eviction and ownership, concurrent cache access,
fuzz decoding, race detection, cross-platform builds, and a warm-read
benchmark:

    go test ./internal/core -run '^(TestDiskANN|TestParallelReadAt)'
    go test ./internal/core -run '^$' -bench '^BenchmarkDiskANNWarmNodeRead$' -benchmem
    go test ./internal/core -run '^$' -fuzz '^FuzzDiskANNNodeFile$'
