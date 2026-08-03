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
v0.2 milestone covers versioned storage, WAL recovery, CRUD, exact Flat vector
search, SQL filtering, scalar inverted candidates, atomic DDL, and compaction.
The v0.3 work has started with baseline-layout FP16, per-vector INT8, and
packed INT4 scalar quantization primitives. Approximate indexes such as IVF and
HNSW are the next integration layer.

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

The current development branch also contains internal scalar-quantization
kernels for the forthcoming v0.3 indexes. They are intentionally not exposed
as a quantized collection execution path before those indexes can build,
persist, reopen, and search without fallback.

The current library version is `v0.2.0`; its exact support boundary is recorded
in the [v0.2 capability matrix](docs/v0.2.md) and [changelog](CHANGELOG.md).

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

Index parameters for later milestones can be declared and validated now. An
accepted schema does not imply that its ANN or FTS implementation has shipped;
operations must return `ErrNotSupported` until the corresponding milestone is
implemented.

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

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
