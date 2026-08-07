# Native disk compatibility

The native Go collection format is separate from the C++ zvec format. Format
version 1 has remained readable since `v0.1.0`: every Go release through the
current v1 hardening line opens an older collection in place without requiring
an offline conversion step.

`Open` treats the manifest, schema, immutable segments, primary-key and delete
snapshots, and the valid WAL prefix as one versioned unit. Fields added to the
format-1 manifest are optional on read. In particular, a v0.1 manifest has no
`writing_segment_start_doc_id`; the reader derives the initial writable ID from
persisted segments and advances it while replaying the historical WAL. The
next insert therefore remains monotonic even when the old WAL contains updated
or subsequently deleted document versions.

The first successful `Flush`, schema DDL operation, or `Optimize` publishes a
new manifest generation using the current complete metadata. Publication uses
the normal atomic `CURRENT` switch, so opening an old collection does not
rewrite files merely as a side effect and a failed migration leaves the old
generation authoritative.

## Historical fixture gate

`testdata/native_v0_1_0_collection.json` contains a deterministic compressed
collection produced by the tagged `v0.1.0` code at commit
`7005638af6c84b87a196afdb393bc37efd0dd7b8`. Its identity records both the
generator and archive SHA-256 values. The fixture includes:

- an immutable segment and primary-key/delete snapshots;
- a later WAL with an update, live inserts, and deleted documents;
- a format-1 manifest that predates `writing_segment_start_doc_id`.

The cross-version test verifies checksum identity before safe extraction, then
opens the artifact with the current library. It checks payloads, deletions,
document IDs, exact vector/filter results, and continuation at the next unused
ID. It then applies a current numeric-column backfill with an INVERT index,
runs `Optimize`, and verifies the migrated generation through read-only reopen.

Because every tagged 0.x release retained collection format 1 and schema codec
1, the oldest-release fixture crosses every intervening reader boundary. The
fixture predates optional index metadata. Current readers therefore rebuild its
indexes once, then `Flush` or `Optimize` can publish checksummed vector, FTS,
and INVERT artifacts per immutable segment without changing format version 1.
Reopen uses an artifact only when its schema hash, segment ID, document count,
and document bounds match.

Readers reject unknown future format or codec versions instead of guessing.
Format upgrades must add an explicit reader/migration path and a historical
fixture before the writer version changes. C++ collection import remains out of
scope.
