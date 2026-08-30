# xvec

[![CI](https://github.com/gorse-io/xvec/actions/workflows/ci.yml/badge.svg)](https://github.com/gorse-io/xvec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gorse-io/xvec/graph/badge.svg)](https://codecov.io/gh/gorse-io/xvec)
[![Go Reference](https://pkg.go.dev/badge/github.com/gorse-io/xvec.svg)](https://pkg.go.dev/github.com/gorse-io/xvec)
[![Go Version](https://img.shields.io/github/go-mod/go-version/gorse-io/xvec)](go.mod)
[![License](https://img.shields.io/github/license/gorse-io/xvec)](LICENSE)

xvec is an embedded vector database vibe-coded with reference to
[Alibaba zvec](https://github.com/alibaba/zvec). It provides durable local
storage and runs inside your application without CGO, a separate database
server, or prebuilt native libraries.

> [!WARNING]
> xvec is experimental and is not compatible with zvec's API or disk format.
> Unless you specifically need a pure-Go implementation, please use
> [zvec-go](https://github.com/zvec-ai/zvec-go).

## Features

- Dense and sparse vector storage with exact and approximate nearest-neighbor search.
- Flat, HNSW, IVF, IVF-RaBitQ, Vamana, and DiskANN indexes.
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
inner-product search. See the [Collection API](docs/collection-api.md) and
[vector query semantics](docs/vector-query.md) for filters, radius queries,
projections, ANN parameters, grouping, and refinement.

## Documentation

- [Collection API](docs/collection-api.md)
- [Vector query semantics](docs/vector-query.md)
- [Hybrid MultiQuery](docs/multi-query.md)
- [Segment-native indexes](docs/segment-native-indexes.md)
- [Runtime configuration](docs/runtime-config.md)
- [Native Go disk format](docs/disk-format.md)
- [VectorDBBench-compatible Go benchmark](cmd/vector-db-bench/README.md)

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
