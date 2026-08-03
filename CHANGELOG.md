# Changelog

All notable changes to the native Go implementation are documented here. The
project follows semantic versioning; public APIs and the disk format remain
subject to compatible migration work throughout the 0.x series.

## Unreleased

- Added baseline-layout FP16, per-vector INT8, and packed INT4 scalar
  quantization primitives, including batch conversion, decoded-equivalent L2,
  IP, cosine, and MIPS-L2 kernels, strict finite/range validation, and even
  dimension enforcement for INT4 schemas.
- Added reversible arbitrary-dimension four-round FHT/Kac rotation with
  restorable random-sign state, concurrent batch transforms, INT8/INT4
  parameter validation, and an exact original-vector candidate refiner with
  scale-factor expansion, filtering, radius, and deterministic ranking.
- Added a deterministic, context-aware k-means framework with reservoir and
  k-means++ initialization, metric-aware parallel assignment, FP64 ordered
  centroid accumulation, explicit empty-cluster policies, spherical updates,
  immutable models, classification, fuzz coverage, and a training benchmark.

## v0.2.0 - 2026-08-04

- Added a positioned SQL filter lexer/parser, typed three-valued predicate
  evaluation for scalars, arrays, NULL, LIKE, prefix/suffix matching and
  contain predicates, plus schema analysis, rewriting, and forward plans.
- Added exact snapshot-local scalar INVERT candidates for equality, IN, NULL,
  array length/contain, sorted range routing, and optional extended wildcards;
  every candidate remains forward-verified.
- Added WAL-backed `DeleteByFilter` over current live document versions.
- Added atomic CreateIndex and DropIndex for implemented Flat/INVERT state.
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

See [the v0.2 capability matrix](docs/v0.2.md) for exact support boundaries.

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
