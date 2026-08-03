# Changelog

All notable changes to the native Go implementation are documented here. The
project follows semantic versioning; public APIs and the disk format remain
subject to compatible migration work throughout the 0.x series.

## Unreleased

- Added atomic `AddColumn` for basic numeric fields, including concurrent
  arithmetic-expression or nullable-NULL backfill, DocID-preserving live data
  rewrite, failure rollback, and reopen recovery.
- Added atomic `AlterColumn` with numeric type conversion, rename, nullability
  validation, index-parameter migration, rollback, and reopen recovery.
- Added atomic `DropColumn` with physical live-payload removal, index metadata
  removal, DocID preservation, rollback, and reopen recovery.
- Added atomic `Optimize` for live-snapshot segment compaction, deletion and
  superseded-version reclamation, DocID-preserving WAL/snapshot rotation,
  conservative obsolete-artifact pruning, and idempotent retry.
- Added a subprocess crash-recovery matrix for CreateIndex, DropIndex,
  AddColumn, AlterColumn, DropColumn, and Optimize on both sides of the
  `CURRENT` commit point, including post-commit cleanup retry.

## v0.1.0 - 2026-08-04

- Added a pure-Go, no-CGO collection lifecycle with atomic versioned manifests,
  checksummed WAL recovery, immutable segments, deletion snapshots, and
  single-writer/multi-reader process locking.
- Added baseline-compatible public enums, validated schemas and index/query
  parameters, explicit document/vector types, deterministic document payloads,
  projections, and structured errors.
- Added WAL-first Insert, partial Upsert and Update, Delete, ordered Fetch,
  Flush, reopen recovery, read-only handles, statistics, and guarded Destroy.
- Added exact dense Flat search for L2, IP, cosine distance, and MIPS-L2, plus
  exact sparse Flat inner-product search, radius limits, stable top-k merging,
  and scalar group-by.
- Added Linux behavior tests, race tests, codec fuzz targets, crash/corruption
  recovery tests, and Windows/macOS no-CGO cross-compilation checks.

See [the v0.1 capability matrix](docs/v0.1.md) for exact support boundaries.
