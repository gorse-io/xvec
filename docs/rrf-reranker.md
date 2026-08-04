# Reciprocal rank fusion

`RRFReranker` combines MultiQuery branches by rank and deliberately ignores
their raw scores. A document at zero-based rank `r` contributes:

```text
1 / (rank_constant + r + 1)
```

Contributions are summed across every occurrence of the same primary key. This
makes RRF useful when branches have incomparable score domains, such as inner
product, sparse similarity, and BM25.

```go
reranker := zvec.NewRRFReranker() // RankConstant: 60

results, err := collection.MultiQuery(ctx, zvec.MultiQuery{
    Queries: []zvec.SubQuery{
        {Field: "embedding", DenseVector: queryVector, NumCandidates: 50},
        {Field: "body", FTS: &zvec.FTSClause{Match: "vector database"}, NumCandidates: 50},
    },
    TopK:     10,
    Reranker: reranker,
})
```

Leaving `MultiQuery.Reranker` nil selects the same default RRF configuration,
so `Reranker: reranker` may be omitted. Use an explicit value to change the
constant:

```go
reranker := zvec.RRFReranker{RankConstant: 20}
```

`RankConstant` must be non-negative. Zero is a valid explicit value and is not
treated as the default; use `NewRRFReranker` to obtain 60.

RRF output is bounded by `topK`, owns a copy of the first occurrence of each
document, and stores the fused score as `float32`, matching the public document
score type. Ties are made deterministic by primary key and then DocID; this is
stricter than the pinned C++ unordered-map implementation, whose tests only
require the tied set. Input scores and documents are not mutated. The reranker
checks cancellation while accumulating candidates and is safe for concurrent
use.

