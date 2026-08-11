# Callback reranking

`CallbackReranker` adapts a Go function to the `Reranker` interface for custom
MultiQuery fusion. The callback receives the request context, candidate
batches in sub-query order, and the final `topK` limit.

```go
reranker := xvec.NewCallbackReranker(func(
    ctx context.Context,
    batches []xvec.RerankBatch,
    topK int,
) ([]xvec.Document, error) {
    // Apply a domain-specific model or merge policy.
    return rerankCandidates(ctx, batches, topK)
})

results, err := collection.MultiQuery(ctx, xvec.MultiQuery{
    Queries: []xvec.SubQuery{
        {Field: "embedding", DenseVector: queryVector, NumCandidates: 50},
        {Field: "body", FTS: &xvec.FTSClause{Match: "vector database"}, NumCandidates: 50},
    },
    TopK:     10,
    Reranker: reranker,
})
```

The pinned C++ callback receives separate result and field arrays. The Go API
pairs each result list with its cloned `FieldSchema` in `RerankBatch`, avoiding
parallel-slice mismatches. It additionally passes `context.Context` and lets a
callback return an error. A nil callback returns `ErrInvalidArgument`. A panic
is contained and returned as `ErrInternal`; a panic value that implements
`error` remains available through `errors.Is` and `errors.As`.

`Collection.MultiQuery` invokes the callback after releasing its read lock, so
the callback may call collection methods. It then validates that output is
bounded by `topK`, unique, finite-scored, and drawn from the input candidate
union with matching primary keys and DocIDs. Selected documents are
rematerialized from the immutable snapshot, preventing callback field-map
changes from altering stored or returned fields.

Cancellation is checked before and after the callback. Long-running callbacks
should also observe the supplied context. A callback value is safe to share
between concurrent queries only when the function and its captured state are
concurrency-safe. Calling the adapter directly delegates output validation to
the caller.
