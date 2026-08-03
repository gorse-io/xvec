# Native Go disk format

The Go implementation uses its own disk format and does not open collection
files created by the C++ implementation. Format version 1 starts with the
versioned manifest protocol below; later storage files retain their own magic,
version, length, and checksum fields.

## Manifest publication

Each metadata snapshot is stored in an immutable file named
`MANIFEST-<20-digit generation>`. A binary header records the `ZVECMAN` magic,
disk-format version, header size, generation, JSON payload length, and CRC32C of
the payload. The JSON contains schema bytes, persisted segment capacity,
segment metadata, snapshot generations, and the next segment ID.

`CURRENT` is the commit point. It is itself framed and checksummed and names one
manifest. Publication writes and synchronizes the immutable manifest first,
then writes a synchronized temporary pointer and atomically installs or replaces
`CURRENT`. Directory metadata is synchronized where the operating system
supports it. Recovery reads only the manifest named by `CURRENT`; a higher
numbered orphan manifest is never treated as committed.

Manifest writers are serialized by `.version.lock`. A manager also verifies
that `CURRENT` still names the generation it opened, so stale writers fail with
a version conflict instead of losing a newer update. Published files are never
rewritten in place.

## Write-ahead log

A WAL begins with a 32-byte `ZVECWAL` header containing its codec version,
header size, maximum record size, reserved fields, and a CRC32C header checksum.
Each record has its own `ZREC` magic, codec version, header size, monotonically
increasing LSN, payload length, payload CRC32C, header CRC32C, and reserved
field. Payloads are non-empty and limited to 4 MiB.

Recovery validates the complete log before it can be appended. An incomplete
final record header or a header-valid but incomplete final payload is treated as
a crashed append and truncated back to the preceding record. Invalid magic,
versions, LSNs, lengths, reserved fields, header checksums, or the checksum of a
complete payload are reported as corruption and are never silently truncated.
Writers hold an exclusive sidecar advisory lock, and readers replay a stable
valid-prefix snapshot through an independent file handle. A read-only open
uses a shared lock and excludes an incomplete crash tail without truncating it;
the next writable open repairs that tail under the exclusive lock.

## Segments and collection snapshots

An immutable segment starts with a 64-byte `ZVECSEG` header containing the
codec version, segment and document-ID range, document count, payload length,
payload CRC32C, and header CRC32C. Records are stored in contiguous document-ID
order and contain the primary key plus an opaque schema-coded document payload.
Each record also checksums its key and payload. The first file listed for a
segment in the manifest is its data file; later index files can follow it.

Primary-key snapshots (`ZVECPK`) sort keys bytewise and map each key to a
segment/document location. Delete snapshots (`ZVECDEL`) store strictly sorted
global document IDs. Both use a common versioned header with item count,
payload length, payload CRC32C, and header CRC32C. Snapshots and segments are
written as immutable files and atomically installed without replacing an
existing generation.

## Collection recovery and flush

Every manifest names one empty writing segment and its WAL. Opening a
collection loads only the immutable segments and primary-key/delete snapshots
named by `CURRENT`, creates the writing segment at the next global document ID,
and replays the WAL's verified prefix in LSN order. Replayed operations must
match the writing segment, contiguous document IDs, and the recovered
primary-key state. A structurally valid but impossible operation is corruption,
not a request to repair metadata heuristically.

Flush first synchronizes the current WAL. For a non-empty writing segment it
then writes a new immutable segment, complete primary-key and deletion
snapshots, and the next empty WAL. Only after all those files are durable does
it publish a manifest that references them. `CURRENT` remains the sole commit
point: a crash before replacement recovers the old WAL, while a crash after
replacement has every file needed by the new version. An empty flush only
synchronizes the WAL and does not create a new manifest generation.

Artifact names include their segment or snapshot generation. A failed retry
never overwrites an immutable file. Unreferenced artifacts and higher-numbered
orphan manifests are ignored during recovery.

`.collection.lock` controls handle ownership across processes. A writable
collection holds it exclusively for its lifetime. Read-only collections hold
shared locks, allowing multiple readers while preventing a concurrent writer.
Closing without Flush is safe because each successful mutation synchronizes
its WAL record before changing memory.

## WAL operations

Collection mutations inside WAL records use a separate `ZOP1` frame. It stores
the operation kind, target segment ID, assigned global document ID, primary-key
and document-payload lengths, and a CRC32C covering the header, key, and
payload. Insert reserves the next contiguous document ID, synchronizes this WAL
operation, and only then applies the document to the write segment and
primary-key map.
