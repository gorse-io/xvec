# xvec

[![CI](https://github.com/gorse-io/xvec/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/xvec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gorse-io/xvec/graph/badge.svg)](https://codecov.io/gh/gorse-io/xvec)
[![Go Reference](https://pkg.go.dev/badge/github.com/gorse-io/xvec.svg)](https://pkg.go.dev/github.com/gorse-io/xvec)
[![Go Version](https://img.shields.io/github/go-mod/go-version/gorse-io/xvec)](go.mod)
[![License](https://img.shields.io/github/license/gorse-io/xvec)](LICENSE)

xvec is an embedded vector database inspired by
[Alibaba zvec](https://github.com/alibaba/zvec). It provides durable local
storage and runs inside your application without CGO, a separate database
server, or prebuilt native libraries.

> [!WARNING]
> xvec is experimental and is not compatible with zvec's API or disk format.
> Unless you specifically need a pure-Go implementation, please use
> [zvec-go](https://github.com/zvec-ai/zvec-go).

## Features

- Dense and sparse vector storage with exact and approximate nearest-neighbor search.
- Flat, HNSW, HNSW-RaBitQ, IVF, IVF-RaBitQ, Vamana, and DiskANN indexes.
- L2, inner-product, cosine, and MIPS-L2 metrics with optional quantization and refinement.
- Scalar filtering, block-max WAND BM25 full-text search, grouping, and hybrid multi-query retrieval.
- Configurable WAL durability batching, crash recovery, segment-native incremental indexes, and atomic compaction.
- Pure Go on Linux, macOS, and Windows.

## Install

xvec requires Go 1.27 or later.

```bash
go get github.com/gorse-io/xvec
```

Then import it in your application:

```go
import "github.com/gorse-io/xvec"
```

## Usage

The following program creates a local collection, stores vectors with metadata,
and returns the two nearest documents.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gorse-io/xvec"
)

func main() {
	ctx := context.Background()

	schema := xvec.NewCollectionSchema("articles",
		xvec.NewField("title", xvec.DataTypeString),
		xvec.NewField("category", xvec.DataTypeString),
		xvec.FieldSchema{
			Name:      "embedding",
			DataType:  xvec.DataTypeVectorFP32,
			Dimension: 3,
			Index:     xvec.NewFlatIndexParams(xvec.MetricTypeCosine),
		},
	)

	collection, err := xvec.CreateAndOpen(
		ctx,
		"./data/articles",
		schema,
		xvec.NewCollectionOptions(),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer collection.Close()

	_, err = collection.Insert(ctx, []xvec.Document{
		{
			PrimaryKey: "go",
			Fields: map[string]any{
				"title":     "The Go Programming Language",
				"category":  "programming",
				"embedding": xvec.VectorFP32{1.0, 0.1, 0.0},
			},
		},
		{
			PrimaryKey: "vector",
			Fields: map[string]any{
				"title":     "Vector Search Fundamentals",
				"category":  "search",
				"embedding": xvec.VectorFP32{0.9, 0.2, 0.1},
			},
		},
		{
			PrimaryKey: "sql",
			Fields: map[string]any{
				"title":     "Database Internals",
				"category":  "database",
				"embedding": xvec.VectorFP32{0.0, 0.2, 1.0},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	results, err := collection.Query(ctx, xvec.VectorQuery{
		Field:       "embedding",
		DenseVector: xvec.VectorFP32{1.0, 0.0, 0.0},
		TopK:        2,
		Projection: xvec.Projection{
			OutputFields: []string{"title", "category"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, result := range results {
		fmt.Printf("%s: %s (score %.4f)\n",
			result.PrimaryKey,
			result.Fields["title"],
			result.Score,
		)
	}
}
```

The collection is persisted under `./data/articles`. Reopen it after restarting
your application with:

```go
collection, err := xvec.Open(
    context.Background(),
    "./data/articles",
    xvec.NewCollectionOptions(),
)
```

Use `Insert`, `Upsert`, `Update`, and `Delete` for document mutations. Call
`Flush` to publish an immutable segment and `Optimize` to compact stored data;
`Close` synchronizes pending WAL records. Set `CollectionOptions.WALSyncEvery`
to synchronize automatically after a chosen number of successful records; zero
disables automatic record-count-based synchronization. `Query` also accepts
`PrimaryKey` as a vector target, a single `FTS` clause, or a filter-only request
with no target. `MultiQuery` fuses dense, sparse, primary-key-vector, and FTS
branches over one snapshot.

Writes maintain a searchable Flat index. Queries reuse prepared segment indexes
and never build ANN indexes: segments without HNSW, IVF, RaBitQ, Vamana, or
DiskANN artifacts are searched through Flat using the configured metric.
`Flush` persists pending data and available scalar/full-text indexes without
building ANN indexes. Call `Optimize` to compact data and build configured ANN
indexes, or `CreateIndex` to build one field's index on existing segments (even
when its parameters already match the schema). New writes continue through
Flat until the next maintenance operation. `Open` loads existing indexes and
prepares Flat fallbacks before returning. `IndexCompleteness` reports the
fraction of live documents covered by the configured ANN index.

`Optimize` briefly takes the collection lock to seal the writing segment and
capture a stable snapshot. Data compaction and index construction run outside
that lock, allowing queries and writes to continue using existing indexes.
Publication takes the lock again to install the prepared segments and indexes,
preserving concurrent updates, deletes, and new writes. In-flight queries keep
their original snapshots; later writes remain searchable through Flat until
the next maintenance operation. Exact/refined searches scan original vectors
without constructing a second index.

### Search only indexed segments

For latency-sensitive vector searches, opt out of scanning unindexed tails:

```go
options := xvec.NewCollectionOptions()
options.SkipUnindexedSegments = true
collection, err := xvec.Open(ctx, "./data/articles", options)
if err != nil {
    return err
}
defer collection.Close()
results, err := collection.Query(ctx, xvec.VectorQuery{
    Field: "embedding",
    DenseVector: xvec.VectorFP32{1, 0, 0},
    TopK: 10,
})
```

`SkipUnindexedSegments` defaults to `false` and belongs to the open handle; it
is not persisted. It applies only to vector `Query` and `GroupByQuery` targets
(dense, sparse, and `PrimaryKey`). For an ANN field, only immutable segments
with matching committed index metadata for that field are candidates. Mutable
and unbuilt immutable segments are excluded, even with `Linear` or selective
filters. Explicit schema Flat fields remain fully searchable. This trades
freshness/recall for latency and can return fewer than `TopK` results, including
none. Updates in excluded segments still hide their older indexed versions;
deletes remain authoritative. A `PrimaryKey` source vector may be fetched from
an excluded segment without making that segment a candidate.

`Fetch`, filter-only queries, full-text search, `MultiQuery`, statistics, writes,
and maintenance retain their existing behavior. Queries still never build
indexes, and corrupt published indexes remain errors rather than being skipped.
`Flush` alone does not make an unbuilt ANN segment eligible; use `Optimize` or
`CreateIndex` to publish its configured ANN index. Each reopened handle must
set the option again. The example above uses an ANN-indexed `embedding` field;
the Flat field in the initial usage example is intentionally unaffected.


### Choosing an index

| Index | Best for |
| --- | --- |
| Flat | Exact search and small collections |
| HNSW | General-purpose low-latency ANN search |
| IVF | Tunable approximate search with list probing |
| IVF-RaBitQ | Inverted-file probing with memory-efficient RaBitQ scoring |
| Vamana | Graph-based search with deterministic native persistence |
| DiskANN | Disk-backed graph search with bounded node caching |

Dense vectors support FP16 and FP32 storage, plus supported scalar
quantization options. Sparse vectors support exact Flat and HNSW
inner-product search. See the [Go reference](https://pkg.go.dev/github.com/gorse-io/xvec)
for the complete API.

## Benchmark

[![Benchmark](https://xvec.gorse.io/benchmarks/e2-standard-8-Performance768D100K-1e7009b3d2.svg)](https://xvec.gorse.io/)

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
