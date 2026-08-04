# Sparse HNSW persistence and reopen

This independently gated v0.3 unit gives sparse HNSW a native Go disk format.
It is intentionally incompatible with the C++ collection layout. Incremental
writes are implemented by the following unit; collection routing remains
separate.

## File contract

`SparseHNSWIndex.Save` writes a little-endian artifact with a fixed 112-byte
header. The header records an eight-byte magic, format version, exact file and
payload lengths, node and nonzero counts, inner-product build settings, entry
point, maximum level, level-generator state, and separate CRC32C values for the
header and payload.

Version 1 stores nodes in stable insertion order. Each record contains its key,
maximum level, sparse nonzero count, ordered coordinate/FP32-value pairs, and
ordered neighbor positions for every occupied graph level. Positions are local
to the artifact and are never interpreted as document keys.

`OpenSparseHNSWIndex` verifies checksums and lengths before allocating decoded
state. It validates platform and uint32 position bounds, exact aggregate
nonzero counts, inner-product options, unique keys, strictly increasing
coordinates, finite values, levels, degree limits, self/duplicate/out-of-range
edges, neighbor level availability, and entry/maximum-level consistency. CSR
offsets and the key-position map are rebuilt rather than trusted as redundant
serialized data.

## Publication and recovery boundary

Save first clones one complete published graph generation, then validates and
encodes that snapshot before the shared
atomic publisher writes a private temporary file, syncs it, checks
cancellation, renames it over the destination, and syncs the directory. A
crash therefore exposes an old or new complete generation, never a partially
published graph.

Tests cover empty and populated round trips, topology and exact search identity
after reopen, approximate graph-path identity above the 1,000-vector threshold,
atomic replacement, private modes, cancellation, truncation, trailing bytes,
bit flips, semantic corruption with recomputed checksums, and decoder fuzzing.
