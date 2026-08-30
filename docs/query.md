# Queries and filters

`Collection.Query` is the unified single-branch search API. It executes against
one immutable live-document snapshot and supports four target forms:

- `DenseVector` or `SparseVector` searches the named vector field;
- `PrimaryKey` resolves a vector from that document and field in the snapshot;
- `FTS` searches a full-text string field; or
- no target and no field performs a scalar filter scan in ascending document-ID
  order with zero scores.

Set exactly one target for vector or FTS search. `TopK` must be positive for a
targeted query. The index-specific `Params` type must match the field index;
xvec returns `ErrNotSupported` or `ErrInvalidArgument` instead of silently
substituting another algorithm.

```go
params := xvec.NewHNSWQueryParams()
params.EF = 100
params.UseRefiner = true

results, err := collection.Query(ctx, xvec.VectorQuery{
    Field:       "embedding",
    DenseVector: xvec.VectorFP32{1, 0, 0},
    TopK:        10,
    Filter:      `category = "search" AND published >= 2024`,
    Params:      params,
    Projection:  xvec.Projection{OutputFields: []string{"title"}},
})
```

## Scores, radius, and linear execution

L2, cosine distance, and MIPS-L2 rank smaller scores first; inner product ranks
larger scores first. `Radius` uses that metric direction and zero disables
radius filtering.

`Linear` is an explicit request to scan the field's matching representation.
It is useful for exact truth comparisons and never acts as a hidden fallback.
Flat is exact by default. Approximate graph/list indexes use their public EF,
NProbe, or list-size controls.

`UseRefiner` reranks retained candidates with original vectors. Flat, IVF, and
IVF-RaBitQ use their public scale factor; HNSW and Vamana retain their returned
graph candidates; DiskANN uses ten times `TopK`. Approximate radius pruning is
disabled before refinement, while the scalar candidate filter remains active.

## Projection

Projection is applied after matching and scoring. A nil output-field slice
selects all scalar and array fields, a non-nil empty slice selects none, and a
named list selects those fields. `IncludeVectors` independently includes all
vector fields. The special output field `"*"` selects all scalar fields and must
appear alone.

## Group-by query

`GroupByQuery` keeps `TopKPerGroup` documents for each selected string group and
returns the best `GroupCount` groups, ranked by each group's best document.
Flat and explicit linear execution support all implemented representations and
exact refinement. Native HNSW group traversal is supported without refiner, and
IVF-RaBitQ supports native list-probed grouping. Regular IVF, Vamana, and
DiskANN reject non-linear group-by rather than silently scanning. Set `Linear`
when a complete group scan is required.

## SQL filter syntax

Filters are parsed, schema-bound once, and evaluated against the same snapshot
used for search. The core grammar is:

```text
expression  = or-expression
or          = and { OR and }
and         = primary { AND primary }
primary     = predicate | "(" expression ")"
predicate   = identifier comparison value
            | function-call comparison value
            | identifier LIKE value
            | identifier [NOT] IN "(" literal { "," literal } ")"
            | identifier [NOT] (CONTAIN_ALL | CONTAIN_ANY)
                "(" [ literal { "," literal } ] ")"
            | identifier IS [NOT] NULL
```

Keywords are case-insensitive and `AND` binds tighter than `OR`. `IN` requires
at least one literal; contain predicates permit an empty list. `BETWEEN`, unary
Boolean `NOT`, and `NOT LIKE` are not supported productions. SQL line and block
comments are skipped. Parse errors report byte offset, line, and Unicode column.

Schema analysis preserves exact scalar widths and the full unsigned range.
Array contain operations and scalar comparisons require compatible types;
unsafe implicit coercions are rejected.

NULL follows SQL three-valued logic. Comparisons, `LIKE`, `IN`, and contain
predicates return `UNKNOWN` for NULL input, and only `TRUE` candidates survive.
`IS NULL` and `IS NOT NULL` are the direct Boolean NULL tests. A missing nullable
field is treated as typed NULL.

## Inverted candidate plans

Fields with an INVERT index use exact snapshot-local postings cached per segment
and persisted by `Flush` and `Optimize`. The planner intersects indexed `AND`
branches, unions `OR` only when both sides are indexable, and can use one indexed
`AND` branch as a safe prefilter.

Sorted dictionaries accelerate ranges. Prefix `LIKE` is indexable; suffix and a
single middle wildcard require `EnableExtendedWildcard`. Other wildcard forms
fall back to forward evaluation. BINARY keys are byte-exact and bytewise sorted;
ARRAY_BINARY supports contain, NULL, and length candidates without UTF-8
normalization.

When an indexed candidate set covers at least
`RuntimeConfig.InvertToForwardScanRatio` of live documents, xvec switches to an
exact forward scan. This changes cost only, not matching semantics.

## Concurrency and cancellation

Query snapshots are immutable. Writes committed after snapshot acquisition are
not visible. Cancellation propagates through parsing, filter evaluation,
index traversal, scoring, refinement, grouping, and projection. Rejected filter
or radius nodes may remain graph traversal bridges but are never returned.

See [Full-text search](full-text-search.md), [Multi-query and reranking](multi-query.md),
and [Vector indexes](vector-indexes.md) for subsystem-specific details.
