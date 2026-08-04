# IVF persistence and reopen

This v0.3 unit gives the IVF index its first native Go disk format.
The artifact is intentionally independent from the C++ collection layout and
cannot be opened by either implementation. Collection-level IVF remains
disabled until the following incremental-write and orchestration units are
complete.

## File contract

`IVFIndex.Save` writes one file in little-endian order. A fixed 112-byte header
contains an eight-byte magic, format version, total and payload lengths,
dimension, vector/list counts, metric, build options, training state, and
separate CRC32C values for the header and payload. Version 1 payloads store:

1. every trained FP32 centroid in list order;
2. every key and original FP32 vector in stable insertion order;
3. the effective list number for each vector.

The original vectors remain available for exact refinement after reopen. List
positions, the primary-key map, and centroid assignment counts are rebuilt
from the records rather than serialized as redundant trusted state.

`OpenIVFIndex` validates the magic and supported version before accepting the
artifact. It then verifies both checksums, exact file and derived payload
lengths, platform-safe allocation bounds, configured/effective list counts,
metric and training metadata, unique keys, complete single-list membership,
fixed vector dimensions, and finite centroid/vector values. Truncation,
trailing bytes, bit flips, impossible sizes, duplicate keys, and invalid list
references return explicit format or checksum errors; none are repaired or
silently ignored.

## Publication and failure boundary

Save first takes, encodes, and validates a complete consistent snapshot. It
writes a private temporary file in the destination directory, syncs and closes
it, checks cancellation, atomically replaces the destination, and syncs the
parent directory. Therefore a process crash before the rename leaves the previous
generation visible; a crash after the rename leaves the complete new
generation visible. A stale temporary file can remain after an operating-system
kill, but it is never considered an index artifact.

Tests cover non-empty and empty round trips, identical search results after
reopen, atomic replacement, cancellation, subprocess kill injection at the
temporary-file boundary, header and payload corruption, truncation, trailing
data, semantic corruption with recomputed checksums, and decoder fuzzing.
