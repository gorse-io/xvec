# Atomic DDL and Optimize recovery

CreateIndex, DropIndex, AddColumn, AlterColumn, DropColumn, and Optimize use the
same versioned manifest commit protocol. `CURRENT` is the only commit point;
there is no partially visible schema or compaction state.

Schema-only index changes validate their complete live snapshot before writing
a new manifest. Column changes with live documents and Optimize first write all
replacement immutable segments, a primary-key snapshot, an empty deletion
snapshot, and a fresh WAL. They then publish one manifest that references the
complete replacement. Empty column changes use the schema-only path.

Recovery follows one rule:

- If a process stops before `CURRENT` replacement, Open reads the previous
  manifest and ignores every higher manifest, temporary CURRENT file, segment,
  snapshot, and WAL artifact. Future publications skip occupied immutable
  artifact generations rather than overwriting them.
- If a process stops after `CURRENT` replacement, Open reads only the newly
  committed schema and its complete referenced file set. It does not require
  the caller's in-memory mutation to have survived or the collection handle to
  have closed cleanly.

Optimize removes obsolete artifacts only after publication. Cleanup is not
part of correctness: interruption or a filesystem error can leave harmless
unreferenced files, and a later Optimize retries pruning even when the live
layout is already canonical.

The regression suite exercises both sides with real subprocess termination for
all six operations. At the pre-commit boundary it holds `.version.lock`, waits
until the operation is blocked (and, for rewrites, until new segment artifacts
exist), checks that CURRENT is unchanged, and kills the writer. At the
post-commit boundary it waits for the manifest generation to advance and kills
the writer without Close. Every recovered case verifies schema, payloads,
stable document IDs, filtered vector results, and the next monotonic write.
