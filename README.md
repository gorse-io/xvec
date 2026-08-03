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

The implementation is being developed from the storage primitives upward. The first milestone covers versioned storage, WAL recovery, immutable segments, CRUD operations, and exact flat vector search. Approximate indexes such as HNSW will be built on top of that foundation.

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
and deletion snapshots. Public APIs and on-disk formats are not stable before
v1.0.

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

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
