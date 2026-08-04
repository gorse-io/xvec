# ANN collection query integration

This integration connects the public Collection query and DDL surfaces to the
native Go Flat, HNSW, HNSW-RaBitQ, IVF, Vamana, scalar-quantization, rotation, and
original-vector refinement components. A query never substitutes a different
algorithm silently: its parameter type must match the field index, and
unsupported index/operation combinations return `NotSupported`.

## Supported runtime matrix

| Field | Index | First-stage representation |
| --- | --- | --- |
| dense FP32 | Flat/HNSW/IVF | original FP32, FP16, INT8, or INT4 |
| dense FP32, 64–4095 dimensions | HNSW-RaBitQ | portable 1–9 bit RaBitQ codes |
| dense FP32 | Vamana | original FP32, FP16, INT8, or INT4 |
| dense FP16/INT8 | Flat/HNSW/IVF | original values converted to FP32 scoring |
| sparse FP32 | Flat/HNSW | original values or FP16-rounded values |
| sparse FP16 | Flat/HNSW | original FP16 values converted to FP32 scoring |

INT4 retains the schema-level even-dimension requirement. Optional FHT/Kac
rotation is accepted only with INT8 or INT4, as required by public parameter
validation. Because collection ANN artifacts are not segment-native yet, the
runtime derives deterministic rotation signs from the schema and field
identity and rebuilds a query-local index from the durable live-document
snapshot. Reopen therefore produces identical codes and results. The
standalone native IVF/HNSW persistence formats remain available internally;
connecting those artifacts directly to collection segments is a later
lifecycle optimization.

DiskANN remains an explicit `NotSupported` path until its own phase. ANN
group-by traversal is also deferred: an HNSW, HNSW-RaBitQ, IVF, or Vamana
group query must set `Linear`, and quantized/refined group-by currently returns
`NotSupported` rather than falling back. IVF's alternate SOAR memory layout is
also rejected explicitly; the current IVF runtime uses its native row-major
layout. HNSW's contiguous-memory request is satisfied by the Go flat backing
slice.

## Query controls

`FlatQueryParams`, `HNSWQueryParams`, `HNSWRaBitQQueryParams`,
`IVFQueryParams`, and `VamanaQueryParams` are matched to the field's index
before execution. Graph EF is validated in `[1, 2048]`; IVF NProbe is positive
and capped naturally by the trained list count. TopK, metric-aware Radius, and
the SQL filter candidate mask are forwarded to the selected runtime index.
Filter- or radius-rejected graph nodes remain traversal bridges.

HNSW prefetch offset and line controls warm a bounded prefix of neighbor vector
storage. Pure Go has no portable non-faulting hardware prefetch intrinsic, so
the implementation performs deterministic synchronous cache-line touches.
Automatic line count uses the vector footprint, and explicit or automatic
counts are capped at 256 lines, matching the pinned baseline bound. These
performance hints cannot change results.

For Flat/HNSW/IVF/Vamana, setting `Linear` builds the matching Flat representation and
scans it instead of entering the graph or lists. HNSW-RaBitQ instead scans all
full RaBitQ codes so it retains the configured representation; adding
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
`floor(TopK*ScaleFactor)` candidates. HNSW has no public scale factor at the
pinned baseline and therefore uses 1. HNSW-RaBitQ requests up to
`max(TopK, EF)` graph candidates, or all candidates for `Linear`, before exact
reranking. Vamana follows the pinned no-scale-factor behavior and refines its
returned graph candidates. Missing originals and invalid scale arithmetic are errors. Sparse
refinement is not yet implemented and returns `NotSupported` explicitly.

`CreateIndex` backfill-validates Flat, HNSW, HNSW-RaBitQ, IVF, and Vamana—including
conversion overflow, rotation, and RaBitQ training—before atomically publishing
schema parameters.
Writes to an already scalar-quantized field perform the same representation
check before WAL publication, so an unrepresentable vector fails only its
batch item and never creates an incomplete runtime index.
`Optimize` accepts these implemented vector definitions, and Stats reports
their snapshot runtime completeness as 1. A failed backfill leaves the schema
and manifest generation unchanged.

Tests cover parameter mismatch and upper bounds, filtered/radius HNSW,
HNSW-RaBitQ, and Vamana recall against explicit Linear truth, prefetch result invariance,
full-probe IVF, FP16/INT8/INT4 and RaBitQ scoring, deterministic rotation,
exact refinement, sparse FP16 HNSW parity below the exact threshold, DDL
rollback, Optimize, Stats, reopen, and explicit unsupported group/refiner
paths.
