---
home: true
title: xvec documentation
heroText: xvec
tagline: A pure-Go embedded vector database
actions:
  - text: Collections
    link: /collection.html
    type: primary
  - text: Query guide
    link: /query.html
    type: secondary
  - text: Go reference
    link: https://pkg.go.dev/github.com/gorse-io/xvec
    type: secondary
features:
  - title: Embedded and pure Go
    details: Durable local vector storage without CGO, native libraries, or a separate database server.
  - title: Hybrid retrieval
    details: Dense and sparse vectors, scalar filters, BM25 full-text search, grouping, and reranking.
  - title: Crash consistent
    details: Checksummed WAL records, immutable manifests and segments, atomic DDL, and recovery validation.
---

# Documentation

The package reference on [pkg.go.dev](https://pkg.go.dev/github.com/gorse-io/xvec)
is the source of truth for individual exported types, functions, fields, and
methods. These guides cover behavior that spans several API declarations or is
important to operating and maintaining a durable collection.

## User guides

- [Collections](collection.md): schema creation, writes, reads, iteration, DDL,
  compaction, and handle lifecycle.
- [Queries and filters](query.md): vector search, FTS and filter-only queries,
  projections, grouping, SQL predicates, and query parameters.
- [Multi-query and reranking](multi-query.md): hybrid candidate generation,
  reciprocal-rank fusion, weighted fusion, and callback rerankers.
- [Full-text search](full-text-search.md): tokenization, query syntax, BM25,
  dictionaries, postings, and persistence.
- [Runtime configuration](runtime.md): process-wide concurrency, memory
  admission, logging, planner thresholds, Jieba resources, and statistics.

## Maintainer guides

- [Vector indexes](vector-indexes.md): supported index matrix, quantization,
  build/search behavior, persistence, and refinement.
- [Storage and recovery](storage.md): native disk format, WAL, manifests,
  segments, crash consistency, and atomic schema changes.

The maintainer guides describe the current native Go implementation. They are
not compatibility promises for the C++ zvec API or disk format.

## HNSW benchmark

The following results compare xvec-go with
[zvec-go v0.7.0](https://github.com/zvec-ai/zvec-go) using the same
VectorDBBench-compatible workload. The benchmark ran on 2026-08-29 on an Intel
Core i5-8300H (4 cores, 8 logical CPUs), Windows/amd64, with Go 1.27.0. Build
and query concurrency were both set to all 8 logical CPUs.

| Setting | Value |
| --- | --- |
| Dataset | `Performance768D1M` / Cohere 1,000,000 x 768 |
| Metric and result size | Cosine, TopK = 100 |
| HNSW parameters | M = 15, efConstruction = 500, efSearch = 180 |
| Insert batch size | 100 |
| Quantization / refiner | None / disabled |
| Concurrent search duration | 30 seconds at concurrency 8 |

Build time is lower-is-better. The comparison column is the xvec-go value
divided by the zvec-go value.

| Build metric | xvec-go | zvec-go | xvec-go / zvec-go |
| --- | ---: | ---: | ---: |
| Insert time | 173.196 s | 52.957 s | 3.27x |
| Optimize time | 4,134.506 s | 1,742.895 s | 2.37x |
| Total load time | 4,307.760 s | 1,796.366 s | 2.40x |

Query throughput and recall are higher-is-better; latency is lower-is-better.
Recall differs by 0.00043 in absolute value (0.043 percentage points).

| Search metric | xvec-go | zvec-go | xvec-go / zvec-go |
| --- | ---: | ---: | ---: |
| Recall@100 | 0.94333 | 0.94376 | 1.00x |
| Serial QPS | 122.21 | 472.45 | 0.26x |
| Serial average latency | 7.304 ms | 1.976 ms | 3.70x |
| Serial p95 latency | 10.484 ms | 3.420 ms | 3.07x |
| Serial p99 latency | 12.538 ms | 4.116 ms | 3.05x |
| 8-concurrency QPS | 399.69 | 1,205.17 | 0.33x |
| 8-concurrency average latency | 19.992 ms | 6.635 ms | 3.01x |
| 8-concurrency p95 latency | 30.112 ms | 9.997 ms | 3.01x |
| 8-concurrency p99 latency | 44.417 ms | 11.977 ms | 3.71x |

These are single-run measurements from complete load-and-search runs. They are
intended for regression tracking on this machine, not as hardware-independent
performance guarantees.

## Local development

The documentation site uses VuePress and `vuepress-theme-hope`:

```bash
cd docs
npm install
npm run docs:dev
```

Build the static site with `npm run docs:build`. Generated files are written to
`docs/.vuepress/dist` and are not committed.

## Documentation policy

Keep API-local facts in Go comments and executable `Example` functions. Add to
these guides only when a rule crosses API boundaries, defines persisted state,
or explains an operational constraint. Tests and workflow files remain the
source of truth for test coverage and CI configuration.
