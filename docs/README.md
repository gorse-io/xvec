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
