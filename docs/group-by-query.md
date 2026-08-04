# Group-by vector query

Dense and sparse Flat indexes can retain the best documents from several
distinct scalar groups in one query. Collection fields configured with an ANN
index expose the same complete scan when query parameters explicitly set
`Linear`; non-Linear ANN group traversal remains a separate unsupported
operation. Group-by scans every eligible candidate and does not first take a
global top-k, which could hide all but one group.

`core.GroupByOptions` has the following controls:

- `GroupCount` is the maximum number of returned groups.
- `TopKPerGroup` is the maximum number of documents retained in each group.
- `Radius` and `Filter` use the same metric-aware semantics as regular vector
  queries.
- `Resolve` maps an internal document key to the group value string. An empty
  string is a valid group and matches the pinned C++ baseline representation
  for NULL. Returning `ok=false` excludes the candidate.

Groups are ordered by their best document. Documents within each group are
ordered by the vector metric, with ascending document keys breaking equal
scores. Segment-local results are merged before the global group limit is
applied, and output is deterministic regardless of segment order.

```go
groups, err := index.SearchGroups(ctx, query, core.GroupByOptions{
    GroupCount:   2,
    TopKPerGroup: 3,
    Resolve: func(key uint64) (string, bool) {
        value, found := categoryByDocument[key]
        return value, found
    },
})
```

The root collection facade translates supported scalar field values into these
baseline-compatible strings and materializes projected `zvec.Document`
results. Integer and Boolean values use their ordinary decimal/lowercase form;
FLOAT and DOUBLE use the native baseline's fixed six fractional digits; NULL
uses the empty string group.

An unrefined group scan uses the field's configured representation:
FP16/INT8/INT4 scalar codes and optional rotation for dense vectors, RaBitQ
codes for HNSW-RaBitQ, FP16-rounded sparse values, or original values when no
quantizer is configured. With `UseRefiner`, the complete Linear scan instead
scores retained original dense or sparse vectors before radius admission and
group retention. Optimize and reopen deterministically reconstruct the same
representation and results.
