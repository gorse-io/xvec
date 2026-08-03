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
baseline-compatible enums, and internal package boundaries are in place; storage
and collection operations are not implemented yet. Public APIs and on-disk
formats are not stable before v1.0.

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

Compatibility fixtures derived from the pinned baseline live in `testdata` and
are exercised by `go test ./...`.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
