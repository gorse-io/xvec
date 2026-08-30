# Multi-query and reranking

`Collection.MultiQuery` runs at least two dense-vector, sparse-vector,
primary-key-vector, or FTS branches over one immutable live-document snapshot.
A shared SQL filter is applied to every branch before candidates are passed to
a reranker.

```go
results, err := collection.MultiQuery(ctx, xvec.MultiQuery{
    Queries: []xvec.SubQuery{
        {
            Field: "embedding",
            DenseVector: xvec.VectorFP32{1, 0},
            NumCandidates: 50,
        },
        {
            Field: "title",
            FTS: &xvec.FTSClause{Match: "vector search"},
            NumCandidates: 50,
        },
    },
    TopK: 10,
    Projection: xvec.Projection{OutputFields: []string{"title"}},
})
```

Each `SubQuery` sets exactly one target. A zero candidate count selects
`DefaultSubQueryCandidates`; a zero final `TopK` selects
`DefaultMultiQueryTopK`. Branch counts and final counts cannot exceed
`MaxQueryTopK`.

Branches execute with the same vector/FTS semantics as `Query`. Primary-key
branches resolve their vector from the same snapshot. FTS corpus statistics
include every live document; a scalar filter masks candidates without changing
IDF or average document length.

## Reranker contract

The reranker receives one independently owned, score-ordered `RerankBatch` per
sub-query in sub-query order. It may change scores and order, but returned
documents must be distinct and drawn from those batches. xvec validates callback
output against the snapshot before projection. Implementations shared by
concurrent calls must be concurrency-safe.

A nil reranker selects reciprocal-rank fusion.

### Reciprocal-rank fusion

RRF combines ranks without comparing raw score domains. A document at zero-based
rank `r` contributes:

```text
1 / (rank_constant + r + 1)
```

The default rank constant is 60. Contributions are summed by primary key,
results are ordered by descending fused score, and ties use primary key then
document ID for deterministic output.

```go
reranker := xvec.NewRRFReranker()
```

### Weighted score fusion

`WeightedReranker` normalizes each branch according to its field metric,
multiplies by the corresponding finite weight, and sums by primary key. The
weight count must match the number of branches; negative weights are allowed.

```go
reranker := xvec.NewWeightedReranker(0.35, 0.65)
```

The first occurrence supplies the returned document payload. Equal fused scores
use deterministic key and document-ID ordering.

### Callback reranking

`NewCallbackReranker` adapts a context-aware function:

```go
reranker := xvec.NewCallbackReranker(func(
    ctx context.Context,
    batches []xvec.RerankBatch,
    topK int,
) ([]xvec.Document, error) {
    return externalRerank(ctx, batches, topK)
})
```

Callback errors are returned unchanged. Panics are converted to structured
internal errors so caller code cannot unwind through the collection boundary.
The adapter does not serialize calls; the callback owns its concurrency policy.

Projection is applied only after reranking, so rerankers see the complete
candidate payload needed for validation and scoring while callers receive only
the requested result shape.
