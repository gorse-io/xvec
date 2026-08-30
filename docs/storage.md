# Storage and recovery

xvec uses its own native Go disk format and does not open C++ zvec collections.
Format version 3 uses immutable manifests, an outer WAL, immutable segments, a
Pebble primary-key ID map, and Pebble-backed FTS/INVERT artifacts. Versions 1
and 2 are rejected; there is no fallback reader, migration, or dual write.

## Commit point and manifests

Each metadata snapshot is an immutable `MANIFEST-<generation>` file with a
`ZVECMAN` header, format version, generation, payload length, and CRC32C. Its
payload identifies schema, segment capacity and metadata, the ID-map checkpoint,
delete snapshot, next segment ID, and the first document ID reserved by the
current writing segment.

`CURRENT` is the only commit point. Publication synchronizes a new manifest and
all referenced artifacts, writes a synchronized temporary pointer, atomically
installs `CURRENT`, and synchronizes directory metadata where supported.
Recovery reads only the manifest named by `CURRENT`; a higher orphan generation
is never treated as committed.

Writers serialize publication with `.version.lock` and verify that `CURRENT`
still names the generation they opened. Stale writers fail with a version
conflict instead of overwriting a newer update. Published files are immutable.

## Write-ahead log

A WAL starts with a checksummed `ZVECWAL` header. Each `ZREC` record stores codec
version, monotonically increasing LSN, bounded payload length, and header and
payload CRC32C values. Collection operations inside records use a separate
`ZOP1` frame containing operation kind, global document ID, key and payload
lengths, and a checksum.

Recovery validates the complete prefix before append. An incomplete final
header or payload is a crashed append and is truncated to the preceding record
by a writable opener. Invalid magic, version, LSN, length, reserved fields, or a
complete-record checksum is corruption and is never silently truncated. A
read-only opener excludes an incomplete tail without modifying it.

The outer WAL is the only incremental recovery authority. The Pebble ID-map
working copy disables its own WAL and may be discarded. `WALSyncEvery` creates
optional record-count durability boundaries; `Flush` and `Close` synchronize
all pending records.

## Segments, documents, and indexes

An immutable `ZVECSEG` file stores a checksummed contiguous document-ID range.
Records contain primary key and a schema-coded `ZVECDOC` payload. Fields are
name-sorted and preserve public data type, count, length, NULL, arrays, dense
vectors, and canonical sparse vectors without JSON coercion.

The ID map is a Pebble point map from primary key to global document ID.
`CURRENT` explicitly names an immutable checkpoint. Delete snapshots store
strictly sorted global IDs in checksummed `ZVECDEL` frames.

ANN artifacts are immutable `.zvi` files. FTS and INVERT artifacts are
independent `.pebble` directories with a synchronized `ZVEC-INDEX` marker.
Manifest metadata identifies schema hash, segment, document bounds, field, kind,
and path. Every identity component is validated before use. Missing index
metadata can be rebuilt from segment documents; malformed referenced artifacts
are corruption.

## Open and flush

A manifest names one empty writing segment and WAL. Open loads immutable
segments, the named ID-map checkpoint and delete snapshot, then replays the
verified WAL prefix in LSN order. A writer clones the checkpoint into a fresh
disposable Pebble working directory. A reader uses private state and does not
modify the collection directory.

For a non-empty writing segment, Flush:

1. synchronizes the WAL;
2. writes the immutable segment, ID-map checkpoint, delete snapshot, and next
   empty WAL;
3. synchronizes every candidate artifact;
4. publishes a manifest by replacing `CURRENT`;
5. switches the handle to a fresh working copy; and
6. builds and atomically publishes immutable index artifacts for the new segment.

A crash before `CURRENT` replacement recovers the previous checkpoint and WAL.
A crash after replacement has every file needed by the new generation. An empty
Flush only synchronizes the WAL.

## DDL and optimize

Schema-changing rewrites and the general Optimize path use the same commit
protocol. They build replacement segments from the complete live snapshot, a
fresh ID-map checkpoint, empty delete snapshot, and fresh WAL, then publish one
manifest naming the full replacement. A fresh append-only collection with one
mutable segment and no deletions takes a narrower Optimize path: it flushes that
already-canonical segment directly, publishes its indexes, and prunes obsolete
artifacts without decoding and rewriting every payload.

Live document IDs are preserved, including gaps, and the next ID remains
monotonic. Contiguous ID runs become independent segments bounded by
`MaxDocsPerSegment`. Before commit the old schema and files remain authoritative;
after commit recovery sees only the complete replacement. No recovery path
mixes schema and payloads from different generations.

CreateIndex and DropIndex publish schema-only generations after full validation.
AddColumn, AlterColumn, DropColumn, and Optimize publish only after every
replacement artifact is installed and synchronized.

Post-commit pruning removes only obsolete paths matching strict package-owned
naming rules. Unknown names, symlinks, active working state, and manifest files
are not removed. A crash or failure during pruning leaves harmless unreferenced
artifacts, and a later no-op Optimize retries cleanup.

## Locking and failure behavior

`.collection.lock` is held exclusively by a writable collection and shared by
read-only collections. Multiple readers may coexist; no reader may coexist with
a writer. Collection methods also serialize publication and mutation so queries
observe complete versions.

If applying a fully appended WAL operation to in-memory segment, delete, or ID
map state fails, the handle becomes fail-stop and must be reopened for replay.
Recovery does not heuristically repair structurally valid but impossible
operations.

All persisted lengths, counts, names, ranges, checksums, and path ownership are
validated before allocation or deletion. Corruption returns a structured error
instead of falling back to another generation or index representation.
