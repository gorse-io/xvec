# Vector indexes

xvec implements native Go exact and approximate vector indexes. A query's
parameter type must match the configured field index; unsupported combinations
return an error instead of silently choosing a different algorithm.

## Support matrix

| Field | Indexes | First-stage representation |
| --- | --- | --- |
| dense FP32 | Flat, HNSW, IVF, Vamana | FP32, FP16, INT8, or INT4 |
| dense FP32, 64–4095 dimensions | IVF-RaBitQ | portable 1–9 bit RaBitQ codes |
| dense FP32 | DiskANN | original/scalar codes plus independent PQ frontier codes |
| dense FP16 | Flat, HNSW, IVF, DiskANN | values converted to FP32 scoring |
| dense INT8 | Flat, HNSW, IVF | values converted to FP32 scoring |
| sparse FP16/FP32 | Flat, HNSW | original or FP16-rounded values, inner product only |

L2, inner product, cosine, and MIPS-L2 are supported where schema validation
permits them. Dense binary and integer vector types remain governed by the
public schema/index validators.

## Flat

Flat scans every candidate and is the correctness oracle for recall tests.
Dense Flat supports configured scalar representations and optional exact
refinement. Sparse Flat merges sorted coordinates in linear time, ranks larger
inner products first, and retains only top-k state.

## HNSW

HNSW uses deterministic level assignment and graph construction. Mutable
segments accept incremental insertion; immutable segment graphs are persisted
and reopened without rebuilding when metadata matches. Search greedily descends
upper levels and performs best-first level-zero traversal.

Graphs with at most 1,000 sparse vectors use an exact CSR scan. Larger sparse
graphs use native traversal. Filter- or radius-rejected nodes remain traversal
bridges. EF is validated and bounded; prefetch controls perform deterministic,
bounded cache-line touches and cannot change results.

Native HNSW grouping retains graph candidates and expands level zero when more
distinct groups are needed. Non-linear HNSW group-by rejects `UseRefiner`; use
explicit `Linear` execution for exact refined grouping.

## IVF and IVF-RaBitQ

IVF trains k-means centroids, assigns vectors to lists, and probes the nearest
`NProbe` lists. Full probing is equivalent to the matching Flat
representation. Incremental writes use fixed published centroids; they do not
retrain existing lists. `UseSOAR` is persisted as a compatibility hint but does
not select a different layout.

IVF-RaBitQ combines IVF probing with rotated, quantized residual codes. It is
available for FP32 vectors with 64–4095 dimensions and L2, IP, or cosine. Its
full-code linear scan and optional exact refinement provide deterministic recall
truth paths.

## Vamana and DiskANN

Vamana builds a deterministic RobustPrune graph and supports FP32 plus the same
public FP16/INT8/INT4 scalar representations as dense HNSW. Search uses its
bounded list and EF controls. Graph artifacts are immutable and checksummed.

DiskANN persists a sector-oriented graph for `ReaderAt` or read-only mmap access.
Its public scalar representation supplies graph-build vectors and first-stage
scores. Independent PQ codes, controlled by `PQChunks`, guide frontier
traversal; optional refinement reads retained original vectors. The per-handle
`MaxBufferSize` bounds the 4 KiB node-sector cache.

IVF-RaBitQ supports native list-probed grouping. Regular IVF, Vamana, and
DiskANN reject non-linear group-by. Callers may explicitly set `Linear` to
request a complete scan.

## Quantization and rotation

FP16, INT8, and INT4 encoders reject empty or non-finite input. FP16 rejects
finite FP32 values that round to infinity. Integer ranges are selected per
vector; constant vectors receive a stable zero-scale representation. INT4 uses
one signed value per element and requires even dimensions for its packed disk
codec.

INT8 and INT4 may enable deterministic FHT/Kac rotation. The same immutable
rotation state is applied to documents and queries and persisted with the index.
Scalar codes are used only for first-stage scoring; original vectors are retained
when refinement is requested.

Product quantization trains independent subspaces and uses precomputed distance
tables. RaBitQ trains a deterministic rotation and split codes, preserving
portable bounds and estimates across architectures.

## Build, persistence, and updates

`CreateIndex` validates and builds the complete live snapshot before atomically
publishing parameters. A failed conversion, training pass, or build leaves the
old schema and manifest unchanged. Writes to a scalar-quantized field perform
the same representation validation before WAL publication.

Indexes are owned by immutable collection segments. `Flush` persists native
HNSW, sparse HNSW, IVF, IVF-RaBitQ, Vamana, and DiskANN artifacts only for newly
sealed segments. Unchanged segment artifacts are reused. `Optimize` rebuilds
indexes for compacted segments. Reopen validates schema hash, segment identity,
document bounds, checksums, and index metadata before publishing runtime state;
missing metadata can be rebuilt from durable documents.

## Query refinement

With `UseRefiner`, approximate radius pruning is disabled during first-stage
candidate generation. The scalar filter remains active, then final metric
scores, radius, and top-k are recomputed from original vectors.

Flat, IVF, and IVF-RaBitQ use `floor(TopK * ScaleFactor)` candidates. HNSW and
Vamana have no public scale factor and refine the candidates returned by graph
search. DiskANN uses `floor(TopK * 10)`. Sparse indexes recompute exact inner
products from retained original sparse values.

Explicit linear execution scans the index's configured representation; adding
`UseRefiner` produces an exact original-vector truth path.
