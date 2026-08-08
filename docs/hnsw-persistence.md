# Dense HNSW persistence and reopen

This independently gated v0.3 unit gives the dense HNSW graph its native Go
disk format. The artifact is intentionally incompatible with the C++
collection layout. Incremental graph writes are now implemented by the next
unit, and collection-level query routing is implemented. Flush and Optimize
publish this artifact under exact live-snapshot manifest identity; missing
artifact metadata is rebuilt from durable documents.

## File contract

`HNSWIndex.Save` writes one little-endian file. Its fixed 112-byte header
contains an eight-byte magic, format version, exact total and payload lengths,
node count, dimension, metric, build settings, entry point, maximum level,
level-generator state, and separate CRC32C values for the header and payload.

Version 1 stores nodes in stable insertion order. Each node record contains
its key, maximum level, original FP32 vector, and the ordered neighbor positions
for every occupied level. Positions are format-local and are never exposed as
document keys.

`OpenHNSWIndex` verifies magic and version before accepting an artifact, then
checks both checksums, exact lengths, allocation bounds, build settings, unique
keys, finite vectors, node levels, per-level degree limits, neighbor ranges,
self/duplicate references, neighbor level availability, and consistency of the
entry point and maximum level. It rebuilds the key-position map from records
instead of trusting redundant serialized state. Invalid files return explicit
format, version, or checksum errors and are never repaired silently.

## Publication and recovery boundary

Save fully validates and encodes the graph before writing a private temporary
file in the destination directory. The shared atomic publisher syncs the file,
checks cancellation, renames it over the destination, and syncs the directory.
A crash therefore exposes either the previous complete generation or the new
complete generation. Temporary files left by an operating-system kill are not
recognized as HNSW artifacts.

Tests cover empty and populated round trips, topology and search identity after
reopen, graph-path search above the brute-force threshold, atomic replacement,
cancellation, private file modes, truncation, trailing bytes, header and
payload bit flips, semantic corruption with recomputed checksums, and decoder
fuzzing. The shared atomic-file suite injects subprocess kills at publication
boundaries.
