# ANN collection query integration

This integration connects the public Collection query and DDL surfaces to the
native Go Flat, HNSW, IVF, IVF-RaBitQ, Vamana, DiskANN,
scalar-quantization, rotation, and original-vector refinement components. A
query never substitutes a different algorithm silently: its parameter type
must match the field index, and unsupported index/operation combinations
return `NotSupported`.

## Supported runtime matrix

| Field | Index | First-stage representation |
| --- | --- | --- |
| dense FP32 | Flat/HNSW/IVF | original FP32, FP16, INT8, or INT4 |
| dense FP32, 64–4095 dimensions | IVF-RaBitQ | IVF list probing with portable 1–9 bit RaBitQ codes |
| dense FP32 | Vamana | original FP32, FP16, INT8, or INT4 |
| dense FP32 | DiskANN | original FP32 or public FP16/INT8/INT4 scalar codes, plus independent 8-bit PQ frontier codes |
| dense FP16 | DiskANN | input converted to FP32 scoring plus independent 8-bit PQ frontier codes |
| dense FP16/INT8 | Flat/HNSW/IVF | original values converted to FP32 scoring |
| sparse FP32 | Flat/HNSW | original values or FP16-rounded values |
| sparse FP16 | Flat/HNSW | original FP16 values converted to FP32 scoring |

INT4 retains the schema-level even-dimension requirement. Optional FHT/Kac
rotation is accepted only with INT8 or INT4, as required by public parameter
validation. The runtime derives deterministic rotation signs from the schema
and field identity. It caches one index set per segment; Flush publishes native
IVF, IVF-RaBitQ, HNSW, Vamana, and DiskANN artifacts only for newly immutable
segments, while Optimize replaces compacted segment artifacts. Missing artifact
metadata is rebuilt from durable documents.

DiskANN accepts public FP16, INT8, and INT4 `Quantize` settings on FP32 fields.
The scalar representation supplies graph-build vectors and public first-stage
scores, while its required internal PQ remains controlled independently by
`PQChunks`; optional refinement reads retained original vectors. Dense,
scalar-quantized and sparse HNSW indexes execute native non-Linear graph group
traversal. IVF-RaBitQ executes native list-probed group search. IVF, Vamana, and DiskANN reject non-Linear group-by, matching
the pinned baseline; callers can request their explicit `Linear` full scan.
Flat and explicit Linear group-by honor the configured scalar/RaBitQ
representation or exact original-vector refinement. `IVFIndexParams.UseSOAR`
is accepted and preserved as a compatibility hint. The pinned C++ baseline
does not route that flag into its IVF builder or searcher, so both flag values
intentionally use the same native row-major IVF runtime. HNSW's
contiguous-memory request is satisfied by the Go flat backing slice.

## Query controls

`FlatQueryParams`, `HNSWQueryParams`, `IVFQueryParams`,
`IVFRaBitQQueryParams`, `VamanaQueryParams`, and `DiskANNQueryParams` are matched to
the field's index before execution. Graph EF is validated in `[1, 2048]`; IVF
NProbe is positive and capped naturally by the trained list count. TopK,
metric-aware Radius, and the SQL filter candidate mask are forwarded to the
selected runtime index. Filter- or radius-rejected graph nodes remain
traversal bridges.

HNSW prefetch offset and line controls warm a bounded prefix of neighbor vector
storage. Pure Go has no portable non-faulting hardware prefetch intrinsic, so
the implementation performs deterministic synchronous cache-line touches.
Automatic line count uses the vector footprint, and explicit or automatic
counts are capped at 256 lines, matching the pinned baseline bound. These
performance hints cannot change results.

Native HNSW group-by first retains `GroupCount * TopKPerGroup` graph
candidates. If those candidates do not cover enough distinct groups, a second
best-first level-zero expansion continues from them until it reaches the group
limit or exhausts the connected component. Candidate filters and radius use
the configured first-stage representation, rejected nodes remain traversal
bridges, and equal scores use document keys for deterministic ordering.

For Flat/HNSW/IVF/Vamana/DiskANN, setting `Linear` builds the matching Flat
representation and scans it instead of entering the graph or lists.
IVF-RaBitQ instead scans all full RaBitQ codes so it retains the configured representation; adding
`UseRefiner` then reranks all live candidates exactly. These are explicit
execution requests, not error fallbacks, and the refined RaBitQ path provides
deterministic truth queries for recall checks.

## Quantization and refinement

Scalar-quantized indexes keep immutable codes for first-stage scores and a
separate owned copy of every original FP32-view vector. Rotation, when enabled,
is applied before document and query quantization. HNSW graph traversal uses
the codes; IVF selects centroids with the original metric and scores members of
the probed lists from codes.

With `UseRefiner`, approximate radius pruning is disabled at the first stage,
the SQL candidate filter remains active, and final scores/radius/top-k are
recomputed from original vectors. Flat and IVF use
`floor(TopK*ScaleFactor)` candidates. IVF-RaBitQ uses the same scale factor
before exact reranking. HNSW has no public scale factor at the pinned baseline
and therefore uses 1. Vamana follows the pinned no-scale-factor behavior and refines its
returned graph candidates. Missing originals and invalid scale arithmetic are
errors. Sparse Flat follows its public scale factor, while sparse HNSW follows
the same no-scale-factor candidate behavior as dense HNSW; both recompute exact
inner products from retained original sparse vectors. Query and MultiQuery use
this path. Flat or explicit Linear group-by scans all live candidates and
therefore can apply exact dense or sparse refinement without losing groups.
Non-Linear HNSW group-by rejects `UseRefiner`, as in the pinned baseline;
setting `Linear` makes the requested original-vector refinement explicit.
DiskANN uses `floor(TopK*10)` candidates when `UseRefiner` is enabled, then
reads its original FP32 vectors for exact final metric scoring.

`CreateIndex` backfill-validates Flat, HNSW, IVF, IVF-RaBitQ, Vamana, and
DiskANN—including conversion overflow, rotation, scalar/PQ layering, and
RaBitQ training—before
atomically publishing schema parameters.
Writes to an already scalar-quantized field perform the same representation
check before WAL publication, so an unrepresentable vector fails only its
batch item and never creates an incomplete runtime index.
`Optimize` accepts these implemented vector definitions, and Stats reports
their snapshot runtime completeness as 1. A failed backfill leaves the schema
and manifest generation unchanged.

Tests cover parameter mismatch and upper bounds, filtered/radius HNSW,
IVF-RaBitQ, Vamana, and DiskANN recall against explicit Linear truth,
prefetch result invariance, full-probe IVF, FP16/INT8/INT4 and RaBitQ scoring,
deterministic rotation, exact refinement, sparse FP16 HNSW parity below the
exact threshold, sparse FP16 original-vector refinement, native HNSW group
expansion across original/scalar/sparse representations, IVF-RaBitQ group scans, DDL rollback,
Optimize, Stats, reopen, and baseline-compatible unsupported group paths.
