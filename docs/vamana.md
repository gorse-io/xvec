# Vamana index

The v0.4 Vamana implementation is a pure-Go, single-layer proximity graph for
dense vectors. Its behavioral baseline is zvec commit `58375ff`. The graph and
file formats are native Go and intentionally do not read or write C++ zvec
artifacts.

## Construction and pruning

`VamanaBuilder` owns validated FP32 vectors and inserts them deterministically
in caller order. The default construction settings match the public baseline:
maximum degree `R=64`, construction list size `L=100`, `alpha=1.2`, and maximum
occlusion candidates `C=750`.

Each insertion performs beam search from the current entry point, then applies
DiskANN-style RobustPrune:

1. candidates are sorted nearest-first and capped at `C`;
2. alpha sweeps begin at 1.0 and grow by 1.2 through the configured alpha;
3. a candidate is occluded using the ratio between its query distance and its
   distance to already selected neighbors;
4. at most `R` neighbors are retained; and
5. every retained edge receives a reverse edge, with another RobustPrune when
   the reverse node is already full.

`SaturateGraph` fills remaining degree slots from unselected candidates after
pruning. Build completion selects the vector nearest the component-wise mean
under squared L2 as the graph entry point, matching the baseline dump-time
medoid rule. L2, inner product, cosine, and MIPS-L2 are supported.

`Add` constructs a private topology/vector generation, recomputes the medoid,
and publishes it atomically. Failed or canceled additions do not alter the
visible index. Concurrent search and persistence observe either complete
generation.

## Search and Collection behavior

`VamanaSearchOptions` provides EF, filters, metric-aware radius, and portable
prefetch offset/line hints. Search retains `max(TopK, EFSearch)` candidates.
Filtered and radius-rejected nodes remain traversal bridges. Graphs with at
most 1000 vectors use an exact scan, matching the baseline small-index
threshold. Results use public score conventions and stable key tie-breaking.

Collection supports `VamanaIndexParams` and `VamanaQueryParams` without
fallback. FP16, INT8, and INT4 scalar codes, optional INT8/INT4 FHT/Kac
rotation, Linear execution, exact original-vector refinement, SQL filters,
radius, CreateIndex, Optimize, Stats, and reopen are connected. Go stores
vectors in contiguous flat slices, so `UseContiguousMemory` is inherently
satisfied. The native runtime keeps a key map regardless of `UseIDMap` so
Collection identity and refinement remain available; this only uses extra
memory and does not change results.

Until segment-native ANN artifacts are integrated, Collection reconstructs a
query-local Vamana graph from its durable live snapshot. The standalone core
index nevertheless supports native save/open and continued additions.

## Native persistence

`Save` atomically writes a `ZVECVMNA` artifact containing keys, original FP32
vectors, adjacency lists, construction options, the medoid entry point, and an
explicit format version. The 128-byte header records total/payload lengths,
node and edge counts, and independent CRC32C checksums for header and payload.

Open rejects truncation, trailing bytes, unsupported versions, invalid flags,
checksum failures, impossible sizes or degrees, duplicate keys or neighbors,
self-loops, non-finite vectors, and out-of-range entry points or edges. Neighbor
distances used by later reverse-edge pruning are rebuilt from originals, so a
reopened index can accept additions without auxiliary files.

## Verification

Tests cover deterministic topology, all four metrics, RobustPrune invariants,
saturation, small-index exactness, large-index recall, filters, radius,
quantized scoring, ownership, cancellation atomicity, concurrent
add/search/save/open, equivalent reopen behavior, continued additions,
corruption, fuzzing, Collection lifecycle integration, and a search benchmark:

```sh
go test ./internal/core -run '^(TestVamana|TestScalarQuantizedVamana)'
go test ./internal/core -run '^$' -bench '^BenchmarkVamanaSearch$' -benchmem
go test ./internal/core -run '^$' -fuzz '^FuzzVamanaDecode$'
```
