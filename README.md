# zvec

A pure Go, embedded vector database.

> [!WARNING]
> This project is under active development and is not ready for production use.

## Goals

- Pure Go implementation with no CGO or prebuilt native libraries.
- Embedded, durable storage with write-ahead logging and crash recovery.
- Dense and sparse vector search.
- Scalar filtering, full-text search, and hybrid retrieval.
- Linux, macOS, and Windows support.

The implementation is being developed from the storage primitives upward. The
v0.4 release includes versioned storage, WAL recovery, CRUD, SQL filtering,
atomic DDL/compaction, FP16/INT8/INT4 scalar quantization, deterministic
rotation and refinement, k-means, IVF, and dense/sparse HNSW. Collection
queries route those implemented indexes with explicit ANN controls. v0.4 adds
HNSW-RaBitQ, Vamana, product quantization, and native DiskANN execution;
v0.5 full-text primitives and snapshot-level hybrid MultiQuery are now under
active development.

## Module

```text
github.com/gorse-io/zvec
```

## Status

Development is staged in independently tested changes. The public error model,
baseline-compatible enums, explicit vector types, validated schemas and index
parameters, internal package boundaries, and portable low-level storage
primitives are in place. Exact L2, inner-product, cosine, and MIPS-L2 scoring
plus deterministic batch top-k are also implemented. Versioned, checksummed
manifests have an atomic commit point, and the checksummed WAL repairs only
incomplete crash tails while rejecting corruption. Collection operations are
being built on versioned immutable segments, deterministic primary-key maps,
and deletion snapshots. WAL-first batch insert, upsert, immutable-version
update, and logical delete are implemented internally with per-document
results. Ordered fetch resolves live primary keys across mutable and immutable
segments. The internal collection lifecycle now performs WAL recovery, atomic
segment flush and rotation, read-only opens, and cross-process single-writer or
multi-reader locking. The root-package facade exposes creation/open, lifecycle,
ordered Fetch, partial Update/Upsert, per-document batch results, and exact
dense/sparse Flat queries.
Those queries support metric-aware radius limits, deterministic projection,
and group-by ranking across every live document version.
The v0.2 release includes its SQL-style filter lexer, positioned parser,
typed AST, exact scalar/array predicate runtime, schema analyzer, rewriter, and
forward execution plans for dense, sparse, and group-by Flat queries. Scalar
INVERT fields now provide snapshot-local exact postings, sorted range routing,
array-length/contain candidates, and baseline-compatible optional extended
wildcards, with every candidate forward-verified. WAL-backed DeleteByFilter
now selects only current live versions under the collection write lock and is
durable across reopen. CreateIndex now performs concurrent full-snapshot
backfill validation and atomically publishes implemented Flat/INVERT
parameters. DropIndex atomically clears scalar indexes or restores vectors to
the default Flat/IP definition. AddColumn now atomically rewrites every live
document with a nullable NULL or a numeric arithmetic-expression backfill while
preserving its internal document ID. AlterColumn atomically renames or converts
basic numeric fields with the same DocID-preserving rewrite protocol, and
DropColumn physically removes supported numeric fields from that live snapshot.
Optimize now atomically compacts the complete live snapshot into bounded
contiguous-DocID segments, reclaims deleted and superseded versions, rotates
the WAL and snapshots, and conservatively prunes obsolete native artifacts.
Public APIs and on-disk formats are not stable before v1.0.

The v0.3 release contains internal scalar-quantization kernels, reversible
FHT/Kac rotation, exact original-vector candidate refinement, and deterministic
k-means training. Deterministic unquantized IVF construction, list assignment,
metric-aware NProbe search, a versioned checksummed native IVF artifact with
atomic save/reopen, and concurrency-safe incremental assignment are also
present internally.
Deterministic dense HNSW level assignment, bounded construction search,
diversity pruning, reverse-edge maintenance, and topology inspection are now
present internally. Metric-aware graph search adds the baseline exact
small-segment threshold, configurable EF, selective filtering, radius, and
deterministic result ordering. Native checksummed persistence preserves
topology and identical search behavior across reopen without depending on the
C++ format. Built and reopened graphs also accept atomic, deterministic
incremental additions while concurrent readers retain a complete generation.
Sparse inner-product HNSW now has deterministic CSR-backed graph construction
and topology inspection. Metric-aware graph search adds exact small-graph
behavior plus EF, filtering, radius, and deterministic approximate results. A
checksummed native sparse format preserves CSR data and topology across reopen.
Built and reopened sparse graphs accept atomic incremental additions while
concurrent search and persistence retain a complete CSR/topology generation.
Collection queries now route Flat, dense/sparse HNSW, dense IVF, Vamana, and
DiskANN parameters without fallback. Dense FP16/INT8/INT4 scalar-code scoring,
deterministic optional rotation, EF/NProbe, metric-aware radius, scalar
filters, bounded graph cache warming, Linear execution, and exact
original-vector refinement are connected end to end. Until segment-native ANN
artifacts are integrated, the collection rebuilds these runtime indexes from
its durable live snapshot for each query and DDL validation.

The v0.4 release includes a portable RaBitQ trainer, split-code converter, and
one-bit/full-bit distance estimator for L2, IP, and cosine. HNSW-RaBitQ now
builds its graph from original vectors, uses coarse bounds and full codes while
traversing, and optionally reranks candidates exactly from retained originals.
It supports the pinned 1–9 total-bit and 64–4095 dimension ranges,
deterministic training and topology, filtering, radius, linear code scans,
atomic incremental generations, and a versioned checksummed native artifact.
Collection queries, CreateIndex, Optimize, Stats, and reopen behavior are
connected without fallback. The code and file formats are native Go rather
than C++ compatible; Collection still rebuilds the runtime index from its
durable live snapshot until segment-native ANN artifacts are integrated.

Native Vamana now provides deterministic single-layer graph construction,
multi-round alpha RobustPrune, reverse-link pruning, medoid entry selection,
EF search, filters, radius, cache warming, scalar quantization, rotation, and
exact original-vector refinement. Copy-on-write additions publish complete
generations, and a versioned checksummed native format can be reopened and
extended. Collection routes Vamana queries, DDL, Optimize, Stats, Linear, and
reopen behavior without fallback; as with the other in-memory ANN indexes, it
currently rebuilds the runtime graph from the durable live snapshot.

The native PQ component trains 8-bit, 256-entry codebooks over contiguous
dimension chunks, encodes one byte per chunk, reconstructs vectors, and builds
chunk-major L2 or inner-product query tables for constant-time code lookup.
Its immutable state uses the baseline full-pivot layout and can be cloned or
restored by the DiskANN format. Auto chunking, 12 training iterations, the
200,000-vector training cap, deterministic prefix sampling,
batch encoding, and batch lookup are covered without CGO.

Native DiskANN now builds a Vamana-derived bounded graph, trains 8-bit PQ
codes for frontier ordering, and reads original vectors and adjacency from
versioned 4 KiB sector records. Portable parallel ReaderAt batches, a bounded
concurrent LRU, exact expanded-node scoring, filters, radius, Linear truth
queries, original-vector refinement, cache preloading, and a complete atomic
single-file format work without CGO on Linux, macOS, and Windows. Collection
routes DiskANN query, CreateIndex, Optimize, Stats, and reopen behavior from
its durable live snapshot; public scalar quantization on DiskANN remains an
explicit NotSupported combination because internal PQ has separate semantics.

v0.5 development includes byte-compatible whitespace, Unicode 17 standard,
Unicode 17 codepoint-ngram, and pure-Go Jieba tokenizers. All preserve source
byte offsets and emit contiguous positions. Standard tokenization implements
pinned word, script, combining-mark, malformed-UTF-8, and emoji behavior;
ngram tokenization adds optional general-category segmentation and single- or
adjacent-length ranges. Jieba implements baseline search, mixed, full, and HMM
cut modes using caller-provided cppjieba-format resources. An immutable token
filter contract plus Unicode 17 lowercase, baseline-compatible ASCII-folding,
and all 36 Snowball 3.1.1 stemmers provide reusable post-tokenization
transforms. A checksummed native FTS dictionary now stores prefix-compressed
terms, block-bitpacked doc/tf/length payloads, delta-varint positions, and
deletion-aware cross-segment statistics. A baseline-compatible lexer and
two-phase parser build analyzed term, phrase, AND, OR, and empty query ASTs
with source-located errors; an owned canonicalization pass and lazy posting
iterators execute term, exact-phrase, AND, OR, must, must-not, and deletion
semantics. Immutable deletion-aware BM25 scorers provide stable top-k results,
and streaming native dictionary compaction removes tombstones while preserving
positions and densely remapping document IDs. Snapshot-consistent public
[`MultiQuery`](docs/multi-query.md) now combines dense-vector, sparse-vector,
and FTS candidate batches under one scalar filter through an explicit
`Reranker` contract. FTS supports both expression and natural-match clauses,
uses each field's complete analyzer configuration, keeps BM25 corpus statistics
independent of scalar candidate filtering, validates reranker output, and
returns durable reopen-identical results. Baseline-compatible
[`RRFReranker`](docs/rrf-reranker.md) now provides score-independent fusion and
is the nil/default MultiQuery strategy. [`WeightedReranker`](docs/weighted-reranker.md)
adds pinned metric-specific normalization and per-branch weighting; the
callback adapter remains a separate v0.5 unit. None requires CGO.

The current library version is `v0.4.0`; its exact support boundary is recorded
in the [v0.4 capability matrix](docs/v0.4.md) and [changelog](CHANGELOG.md).

## Schema example

```go
index := zvec.NewHNSWIndexParams(zvec.MetricTypeCosine)
schema := zvec.NewCollectionSchema(
    "books",
    zvec.NewField("title", zvec.DataTypeString),
    zvec.FieldSchema{
        Name: "embedding", DataType: zvec.DataTypeVectorFP32,
        Dimension: 768, Index: index,
    },
)
if err := schema.Validate(); err != nil {
    // Invalid schemas return *zvec.Error and match zvec.ErrInvalidArgument.
}
```

Some index parameters for later milestones can be declared and validated now.
An accepted schema does not imply that every configured algorithm is
executable; operations return `ErrNotSupported` until the corresponding
milestone is implemented.

## Architecture

The implementation keeps three internal layers:

- `internal/ailego`: portable I/O, storage, concurrency, and math primitives.
- `internal/core`: index builders, searchers, streamers, providers, refiners,
  reformers, and index algorithms.
- `internal/db`: collection lifecycle, schemas, segments, manifests, and query
  orchestration.

The root `zvec` package is the only public API. Public enum values are locked to
the C++ public headers at commit `58375ff`; they are not copied blindly from the
legacy protobuf, whose DiskANN and Vamana values differ. The Go implementation
uses a separate, versioned disk format and does not read C++ collection files.
The native format and its atomic manifest protocol are documented in
[`docs/disk-format.md`](docs/disk-format.md).

Compatibility fixtures derived from the pinned baseline live in `testdata` and
are exercised by `go test ./...`.

## Documentation

- [Collection API](docs/collection-api.md)
- [v0.4 capability matrix](docs/v0.4.md)
- [v0.3 capability matrix](docs/v0.3.md)
- [v0.2 capability matrix](docs/v0.2.md)
- [Native Go disk format](docs/disk-format.md)
- [Documents and projection](docs/document-projection.md)
- [Exact vector query semantics](docs/vector-query.md)
- [Group-by vector query](docs/group-by-query.md)
- [SQL filter parser](docs/sql-filter.md)
- [Scalar filter evaluation](docs/filter-evaluation.md)
- [Schema analysis and filter plans](docs/filter-plans.md)
- [Scalar inverted candidate indexes](docs/scalar-inverted.md)
- [Delete by filter](docs/delete-by-filter.md)
- [CreateIndex](docs/create-index.md)
- [DropIndex](docs/drop-index.md)
- [AddColumn](docs/add-column.md)
- [AlterColumn](docs/alter-column.md)
- [DropColumn](docs/drop-column.md)
- [Optimize](docs/optimize.md)
- [Atomic DDL and Optimize recovery](docs/atomic-recovery.md)
- [Scalar vector quantization](docs/scalar-quantization.md)
- [Rotation preprocessing and refinement](docs/rotation-refinement.md)
- [K-means training framework](docs/kmeans.md)
- [IVF construction](docs/ivf-build.md)
- [IVF search](docs/ivf-search.md)
- [IVF persistence and reopen](docs/ivf-persistence.md)
- [IVF incremental writes](docs/ivf-incremental.md)
- [Dense HNSW construction](docs/hnsw-build.md)
- [Dense HNSW search](docs/hnsw-search.md)
- [Dense HNSW persistence and reopen](docs/hnsw-persistence.md)
- [Dense HNSW incremental writes](docs/hnsw-incremental.md)
- [Sparse HNSW construction](docs/hnsw-sparse-build.md)
- [Sparse HNSW search](docs/hnsw-sparse-search.md)
- [Sparse HNSW persistence and reopen](docs/hnsw-sparse-persistence.md)
- [Sparse HNSW incremental writes](docs/hnsw-sparse-incremental.md)
- [ANN collection query integration](docs/ann-query-integration.md)
- [RaBitQ training and distance estimation](docs/rabitq.md)
- [HNSW-RaBitQ index](docs/hnsw-rabitq.md)
- [Vamana index](docs/vamana.md)
- [Product quantization](docs/pq.md)
- [DiskANN storage and I/O](docs/diskann-storage.md)
- [DiskANN index](docs/diskann.md)
- [Whitespace tokenizer](docs/whitespace-tokenizer.md)
- [Standard tokenizer](docs/standard-tokenizer.md)
- [Ngram tokenizer](docs/ngram-tokenizer.md)
- [Jieba tokenizer](docs/jieba-tokenizer.md)
- [Lowercase token filter](docs/lowercase-token-filter.md)
- [ASCII-folding token filter](docs/ascii-folding-token-filter.md)
- [Stemmer token filter](docs/stemmer-token-filter.md)
- [FTS dictionary, postings, and statistics](docs/fts-postings.md)
- [FTS query parser and boolean execution](docs/fts-query-parser.md)
- [BM25 scoring and FTS segment merging](docs/fts-bm25-merge.md)

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
