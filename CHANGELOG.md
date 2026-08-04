# Changelog

All notable changes to the native Go implementation are documented here. The
project follows semantic versioning; public APIs and the disk format remain
subject to compatible migration work throughout the 0.x series.

## Unreleased

- Added deterministic RaBitQ centroid, rotation, and expected-scale training;
  portable 1–9 bit split-code conversion; one-bit and full-bit L2/IP/cosine
  estimation with the baseline probabilistic pruning envelope; restorable
  immutable model state; pinned-library fixtures; fuzzing; and benchmarks.
- Added deterministic original-vector HNSW-RaBitQ construction, coarse-bound
  graph traversal, full-code ranking, optional exact refinement, filters,
  radius and linear scans, versioned checksummed persistence, atomic
  incremental generations, corruption fuzzing, recall gates, and Collection
  query/CreateIndex/Optimize/Stats/reopen integration without fallback.
- Added deterministic Vamana construction with baseline multi-round
  RobustPrune, reverse-link pruning, graph saturation, medoid entry points,
  metric-aware EF/filter/radius search, scalar quantization and refinement,
  atomic streaming generations, versioned checksummed persistence, corruption
  fuzzing, recall gates, benchmarks, and Collection lifecycle integration.
- Added deterministic 8-bit product quantization with baseline 256-entry
  contiguous chunk layout, prefix-bounded training, immutable restorable
  pivot state, model-bound encoding and reconstruction, additive L2 and inner
  product distance tables, batch operations, fuzz coverage, and benchmarks.
- Added the native DiskANN node-storage layer with versioned 4 KiB headers,
  packed and multi-sector records, whole-section and per-node CRC32C,
  portable parallel ReaderAt batching, exact short-read handling, request
  deduplication, and a concurrency-safe bounded LRU node cache.

## v0.3.0 - 2026-08-04

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
- Added one-shot unquantized IVF construction with public-aligned defaults,
  deterministic centroid training and list assignment, empty-index support,
  retryable cancellation, contiguous original-vector ownership, immutable list
  inspection, and the `DenseProvider` contract for later exact refinement.
- Added metric-aware IVF search with explicit/default NProbe, stable centroid
  probing, exact original-vector list scans, filters, metric-specific radius,
  deterministic top-k, empty-index behavior, and full-probe equivalence to
  Flat across all four dense metrics.
- Added a versioned checksummed IVF artifact with atomic save/reopen,
  corruption detection, immutable topology inspection, and concurrency-safe
  incremental assignment for built and reopened indexes.
- Added deterministic dense HNSW construction, diversity pruning, metric-aware
  EF search, filtering, radius, stable ranking, and recall gates across L2, IP,
  cosine, and MIPS-L2.
- Added versioned checksummed dense HNSW persistence and atomic incremental
  graph generations for concurrent search, save, and reopen.
- Added CSR-backed sparse inner-product HNSW construction and search with exact
  small-graph behavior, deterministic approximate traversal, recall gates,
  checksummed persistence, and atomic incremental generations.
- Connected Collection Flat, dense/sparse HNSW, and dense IVF execution without
  fallback, including FP16/INT8/INT4 scoring, deterministic rotation,
  EF/NProbe/Linear controls, filters, radius, bounded cache warming, exact dense
  refinement, CreateIndex/Optimize/Stats integration, write validation, and
  deterministic reopen.
- Added a shared Flat/IVF/HNSW/INT8-HNSW benchmark with latency, allocation,
  and recall reporting, plus a deterministic partial-probe IVF recall floor.

See [the v0.3 capability matrix](docs/v0.3.md) for exact support boundaries.

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
