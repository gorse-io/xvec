# Weighted score fusion

`WeightedReranker` normalizes every branch's native scores, multiplies them by
the corresponding weight, and sums contributions for equal primary keys. It is
useful when the relative importance of branches is known and their score
domains need to be made comparable.

```go
reranker := zvec.NewWeightedReranker(0.35, 0.65)

results, err := collection.MultiQuery(ctx, zvec.MultiQuery{
    Queries: []zvec.SubQuery{
        {Field: "embedding", DenseVector: queryVector, NumCandidates: 50},
        {Field: "body", FTS: &zvec.FTSClause{Match: "vector database"}, NumCandidates: 50},
    },
    TopK:     10,
    Reranker: reranker,
})
```

The number of weights must equal the number of sub-queries. Weights must be
finite; zero and negative weights are supported. `NewWeightedReranker` owns a
copy of its input slice.

The pinned normalization formulas are:

| Field score | Normalized score |
| --- | --- |
| FTS / BM25 | `2 * atan(score) / pi` |
| L2 distance | `1 - 2 * atan(score) / pi` |
| Inner product | `0.5 + atan(score) / pi` |
| Cosine distance | `1 - score / 2` |

Sparse-vector fields use inner-product normalization. MIPS-L2 has no pinned
weighted normalization and returns `ErrInvalidArgument`; it is never silently
treated as L2. Non-finite input scores, contributions, sums, or final `float32`
scores are rejected.

Documents are aggregated by primary key, with the first occurrence supplying
the returned payload. Sorting uses the accumulated `float64` score, then
primary key and DocID for deterministic ties; the public result stores the
score as `float32`, matching `Document.Score`. Inputs are not mutated, context
cancellation is honored, and an immutable reranker value is safe for concurrent
use.

