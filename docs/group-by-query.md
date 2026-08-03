# Group-by vector query

The exact dense and sparse Flat indexes can retain the best documents from
several distinct scalar groups in one query. Group-by scans every eligible
candidate; it does not first take a global top-k, which could hide all but one
group.

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

The root collection facade will translate supported scalar field values into
the baseline-compatible group strings and materialize projected documents.
