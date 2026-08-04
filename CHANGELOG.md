# Changelog

All notable changes to the native Go implementation are documented here. The
project follows semantic versioning; public APIs and the disk format remain
subject to compatible migration work throughout the 0.x series.

## Unreleased

- Added baseline-compatible reciprocal rank fusion with a default rank
  constant of 60, primary-key aggregation, score-independent rank
  contributions, deterministic tied-result ordering, cancellation and
  concurrency safety, a pinned source fixture, fuzzing, and a benchmark.
  MultiQuery now uses RRF when its `Reranker` is nil.
- Added snapshot-consistent `Collection.MultiQuery` across dense vectors,
  sparse vectors, and exact BM25 full-text branches with one shared SQL scalar
  filter, per-branch candidate bounds, expression and natural-match FTS,
  complete tokenizer/filter configuration routing, projection-safe reranker
  batches, strict untrusted-output validation, lock-free caller reranking,
  reopen/concurrency coverage, a pinned compatibility fixture, fuzzing, an
  example, and a benchmark. The generic `Reranker` contract is public; weighted
  and callback adapters remain separate v0.5 units.
- Added immutable baseline-compatible BM25 scoring with deletion-aware
  cross-segment statistics, scored exact term/phrase/boolean iterators,
  deterministic bounded top-k search, and streaming native FTS dictionary
  compaction that removes tombstones, densely remaps document IDs, preserves
  positions, recomputes maximum tf and statistics, and supports immediate
  encode/reopen, with compatibility fixtures, fuzzing, race tests, and
  benchmarks.
- Added a pure-Go FTS lexer, tokenizer/filter analysis pipeline, and two-phase
  parser for term, phrase, natural-match, explicit and implicit boolean expressions,
  modifiers, grouping, and escapes, with analyzed AST nodes, source-located
  structured errors, an owned idempotent AST canonicalizer, lazy term/AND/OR
  posting iterators, exact repeated-term phrase-position matching, deletion
  snapshots, seek support, complexity bounds, cancellation, concurrency
  tests, compatibility fixtures, fuzzing, and benchmarks.
- Added a versioned native FTS dictionary with byte-prefix-compressed terms,
  128-document bitpacked postings, inline tf/document length, delta-varint
  positions, skip seeking, nested CRC32C validation, immutable snapshots, and
  exact deletion-aware cross-segment document/token/term statistics, plus
  corruption fuzzing, concurrency, examples, and benchmarks.
- Added a pure-Go Snowball 3.1.1 stemmer token filter with all 36 pinned
  algorithms and 115 case-sensitive libstemmer names and aliases, exact
  cross-language fixtures, explicit invalid-language errors, metadata and
  ownership preservation, cancellation, concurrency, fuzz, benchmark, and
  BSD-3-Clause attribution coverage.
- Added a context-aware ASCII-folding token filter with pinned Unicode 17 NFKD
  mappings, the complete baseline supplemental fold table, byte-preserving
  malformed-UTF-8 fallback, empty-token removal, exhaustive source/effective
  mapping identities, concurrency, fuzz, and benchmark coverage.
- Added a context-aware lowercase token filter with an immutable `TokenFilter`
  contract, exact utf8proc 2.11.3 / Unicode 17 simple mappings, byte-preserving
  malformed-UTF-8 fallback, exhaustive mapping identity, metadata/ownership,
  concurrency, fuzz, and benchmark coverage.
- Added a pure-Go Jieba tokenizer with baseline search, mixed, full, and HMM
  modes; cppjieba resource formats and resolution order; user-dictionary
  isolation; exact C++ fixtures; explicit invalid-input errors; cancellation,
  concurrency, fuzz, and benchmark coverage; and MIT attribution.
- Added a Unicode 17 codepoint ngram tokenizer with single- or adjacent-length
  ranges, baseline general-category filtering, Elasticsearch-compatible
  whitespace exceptions, valid-NUL and malformed-UTF-8 distinctions,
  source-hashed fixtures, fuzz coverage, and a throughput benchmark.
- Added a context-aware Unicode 17 standard tokenizer matching the pinned
  baseline's word, connector, script, combining-mark, malformed-UTF-8, emoji,
  codepoint-length, offset, and position behavior, with licensed generated
  tables, source-hashed fixtures, fuzz coverage, and a throughput benchmark.
- Added a context-aware, byte-compatible whitespace tokenizer with owned term
  text, uint32 byte offsets, contiguous positions, arbitrary-byte handling,
  fuzz coverage, and a throughput benchmark.

## v0.4.0 - 2026-08-04

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
- Added native DiskANN graph construction, PQ-ordered batched traversal,
  metric-specific cosine and MIPS-L2 PQ preparation, exact expanded-node
  scoring and original-vector refinement, cache preloading, atomic complete
  index persistence, corruption detection, and Collection query/DDL/Optimize
  integration on every supported Go platform.
- Added v0.4 release gates for DiskANN recall and byte-identical reopen
  behavior across all four dense metrics, bounded concurrent sector I/O,
  process-kill atomic publication, shared search-quality reporting, artifact
  size, and end-to-end build cost.

See [the v0.4 capability matrix](docs/v0.4.md) for exact support boundaries.

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
