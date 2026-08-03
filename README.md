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

The repository currently contains the initial project scaffold. Public APIs and on-disk formats are not stable before v1.0.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
