# Vamana memory optimization

The original `benchmark-vamana.csv` reported xvec peak RSS of 2,797–3,739 MiB
versus 567–783 MiB for zvec on the 100K × 768 workload. The four xvec rows
were replaced with the latest shared-vector rerun on 2026-09-28; zvec retains its historical
measurements. The [previous CSV](benchmark-runs/vamana-memory-20260928/previous.csv)
is preserved for comparison.

The local zvec implementation separates vector and graph storage and dumps
vector/neighbor segments independently (`VamanaEntity::dump_vectors`,
`dump_neighbors` in `src/core/algorithm/vamana/vamana_entity.cc`). Its
`VamanaStreamerEntity` also allocates vector and graph regions from known
record counts. The initial Go changes applied explicit capacity and ownership management
to avoid multiple simultaneous copies:

- Reserve Vamana builder storage using the collection's vector count.
- Feed documents directly to the builder, borrowing FP32 input until `Add`
  copies it; converted FP16 inputs are temporary per document.
- Transfer freshly built/reopened graphs into scalar quantization. The public
  constructor still clones caller-owned graphs to preserve snapshot isolation.
- Stream persistence through a 64 KiB buffer, with incremental CRC32C and
  atomic replacement. This removes the full graph clone and full-file buffers.
- Reuse validation scratch, including the FP16 conversion buffer and neighbor
  deduplication map.

The file version and bytes are unchanged. Regression tests compare SHA-256
checksums captured from the previous encoder for FP32 and FP16 artifacts.
Saving holds the index read lock through persistence, so concurrent Add
publication waits for the save to finish; searches remain concurrent.

## Local allocation measurement

Baseline: `5d9b8f5`; Linux amd64, AMD EPYC 7B12, GOMAXPROCS=8.
`BenchmarkVamanaSaveMemory` uses 2,048 vectors of dimension 768 and excludes
construction from the timed/allocation region. Each measurement ran three
save operations with the same benchmark source on the baseline and new code.

| Storage | Before B/op | After B/op | Before allocations/op | After allocations/op |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 19,807,328 | 68,952 | 4,877 | 18 |
| FP16 | 16,670,208 | 72,024 | 6,955 | 19 |

These are allocated bytes per save, not retained heap or process peak RSS.
The separate 100K rerun below measures end-to-end behavior.

```sh
go test ./internal/core/algorithm -run '^$' \
  -bench BenchmarkVamanaSaveMemory -benchtime=3x -count=1
go test ./...
```

The full test suite passed, including concurrent Add/Search/Save/Open tests
and collection quantization/optimization/reopen tests. The race detector
could not run in this environment because no C compiler is installed.

## First 100K end-to-end rerun (2026-09-28)

The four xvec runs used fresh collections on e2-standard-8 (AMD EPYC 7B12),
Go 1.27.1, CGO_ENABLED=0, GOMAXPROCS=8, GOMEMLIMIT=24GiB, and CPU affinity
0–7. The Cohere shuffled training data, all 1,000 test queries, K=100,
construction degree/list 64/100, alpha 1.2, occlusion limit 750, search list
200, rotation for INT4/INT8, and other settings match the prior CSV.
Concurrent search ran for 30 seconds with 8 workers, followed by a 3-second
cooldown before serial search. Dataset downloads and binary compilation were
excluded from process resource measurements.

| Precision | Previous peak RSS (MiB) | Rerun peak RSS (MiB) | Change | Recall@100 (%) | Serial QPS | Concurrent QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT4 | 3092.26 | 3117.17 | +0.81% | 87.018 | 225.58 | 1409.99 |
| INT8 | 3145.95 | 3152.14 | +0.20% | 98.720 | 227.28 | 1340.87 |
| FP16 | 3739.13 | 3278.91 | -12.31% | 99.363 | 163.43 | 985.35 |
| FP32 | 2797.18 | 2474.98 | -11.52% | 99.378 | 239.45 | 1016.98 |

FP16/FP32 show lower process peak RSS; INT4/INT8 do not. This confirms that
the save-allocation improvement alone does not establish an equivalent reduction
in the whole benchmark's high-water mark. Concurrent QPS is lower in all four
reruns. These are single-run historical comparisons, not a controlled isolation
of the patch: the current base commit also includes later changes, including
SIMD kernels. Graph construction is interleaved and can vary between runs.

[Raw reports and provenance](benchmark-runs/vamana-memory-20260928/) include
JSON results, wait4 resource measurements, exact commands, binary and dataset
SHA-256 hashes, the source patch, and the previous CSV. `backend_version`
identifies base commit `5d9b8f5ff98f13ba0a1d5ee65f9d53dd57456997` plus patch
SHA-256 prefix `7e9600d19f64`; it does not claim the uncommitted changes are
part of that base commit.

## Runtime-memory follow-up (2026-09-28)

Profiling the first patch on the complete INT4 workload exposed three roughly
300 MiB allocations during query startup: the serialized artifact read buffer,
an independently built exact Flat index, and decoded document vectors. The
native graph's decoded vectors added another 336 MiB including topology.
Construction also allocated about 1.8 GiB cumulatively in candidate heap growth;
this is allocation volume, not live heap or peak RSS.

The follow-up removes the first three copies and the repeated quantized Flat
encoding: Vamana uses a lazy exact fallback, a Flat view sharing graph scalar
codes, and encoded immutable document vectors in read-only collections.
`enable_mmap` now uses a temporary mapping when reopening Vamana artifacts;
owned graph data remains valid after unmapping. Construction heaps and query
visit marks are reused between traversals. No forced GC, GOGC change, or memory
limit change was introduced. Linear search, refinement, projected vectors,
read-only reopen, and concurrent operations remain covered by the tests.

The following table compares the first patch with the runtime follow-up.
The four new complete runs used the same environment and parameters as above.

| Precision | First patch RSS (MiB) | Runtime patch RSS (MiB) | Change | zvec RSS (MiB, historical) | Recall@100 (%) | Concurrent QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT4 | 3117.17 | 2241.75 | -28.08% | 566.63 | 87.022 | 1542.97 |
| INT8 | 3152.14 | 2247.39 | -28.70% | 644.31 | 98.722 | 1488.73 |
| FP16 | 3278.91 | 2256.76 | -31.17% | 783.31 | 99.347 | 1025.82 |
| FP32 | 2474.98 | 2147.55 | -13.23% | 770.00 | 99.383 | 845.52 |

Peak RSS is lower by 13–31% versus the first patch, but is still about 2.8–4.0
times the historical zvec measurements. This does not establish memory parity.
The graph still owns original FP32 scoring vectors, and construction and storage
also contribute to the process-wide peak. Further reductions need profiling
of those remaining paths; the diagnostic heap samples above are from before
this follow-up.

INT4/INT8/FP16 concurrent QPS improved versus the first rerun. FP32 full-run QPS
was lower (serial 140.20, concurrent 845.52), so an additional controlled query
comparison reused exactly the same persisted graph with both binaries:

| Same FP32 graph, query-only diagnostic | First patch | Runtime patch |
| --- | ---: | ---: |
| Serial QPS (1,000 queries) | 211.11 | 211.67 |
| Concurrent QPS (8 workers, 10 seconds) | 882.18 | 979.51 |

This diagnostic did not reproduce the large serial regression. It is a separate
single-run check, not proof of invariant performance. The CSV retains the
original complete run, including the lower FP32 QPS, rather than substituting
query-only results. Interleaved graph construction and machine variability limit
causal conclusions from single-run measurements.

[Follow-up artifacts](benchmark-runs/vamana-runtime-20260928/) preserve both
kinds of reports, commands, checksums, the production source patch, and the prior
CSV. That version is base `5d9b8f5ff98f13ba0a1d5ee65f9d53dd57456997` plus
runtime patch `47525295f56e`. The full Go suite and added mmap/storage-sharing
regressions passed; the race-detector limitation noted above still applies.

## Shared-vector follow-up (2026-09-28)

The next patch addresses the remaining original-vector copies and an unrelated
whole-document copy in the optimization predicate:

- Build Vamana directly over immutable collection FP32 rows. This removes the
  builder's additional contiguous FP32 array; quantization shares these rows
  too. Mutable `Add` operations detach using copy-on-write storage.
- Reopen scalar-quantized Vamana against the collection's existing encoded FP32
  originals. The loader verifies each row against the persisted artifact before
  retaining it, then releases the temporary mapping. Original-vector access and
  refinement decode on demand instead of retaining a second FP32 array.
- Omit edge-distance caches for that immutable quantized reader. Graph traversal
  uses topology and scalar codes; cloning for mutation reconstructs the cache.
- Check the writing segment's document count through metadata when deciding
  whether optimization is needed. Previously this predicate cloned all stored
  document payloads through `Documents()` merely to check for an empty segment.

The patch keeps FP32 construction distances, quantization, graph parameters,
refinement behavior, and the on-disk format. It introduces no GC tuning or forced
collection. Regression tests build more than 1,000 nodes to exercise graph
traversal and prefetching, compare borrowed and owned artifact bytes, and cover
L2/IP/cosine, three scalar precisions, mmap on/off, filtered searches, original
vector isolation, resave, and copy-on-write mutation. Missing or mismatched
encoded originals are rejected.

All four complete 100K runs used the same settings and fresh collections. The
CSV now contains these measurements:

| Precision | Previous RSS (MiB) | New RSS (MiB) | Change | zvec RSS (MiB, historical) | Recall@100 (%) | Concurrent QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT4 | 2241.75 | 1575.29 | -29.73% | 566.63 | 87.013 | 1410.06 |
| INT8 | 2247.39 | 1508.05 | -32.90% | 644.31 | 98.726 | 1322.81 |
| FP16 | 2256.76 | 1486.04 | -34.15% | 783.31 | 99.349 | 882.08 |
| FP32 | 2147.55 | 1956.61 | -8.89% | 770.00 | 99.374 | 978.77 |

The quantized runs reduce peak RSS by 30–34%; unquantized FP32 improves by 9%.
The encoded-original reopen path applies only to scalar-quantized indexes;
unquantized reopen still owns a contiguous FP32 scoring array. Peak RSS remains
1.9–2.8 times the historical zvec values, so the gap is smaller but substantial.
zvec was not rerun in this round.

Recall changes are within 0.009 percentage points of the previous rerun. INT4,
INT8, and FP16 concurrent QPS decline by 8.6%, 11.1%, and 14.0%, respectively;
FP32 improves by 15.8%. The Optimize phase is slower in all four runs. The patch
therefore demonstrates a memory improvement, not a general speed improvement.
These are single-run comparisons with interleaved construction, not controlled
attribution of the performance changes. All measured timings, including the
regressions, are retained in the CSV.

The full Go suite and the additional Vamana collection snapshot/mutation test
passed. [Raw reports and provenance](benchmark-runs/vamana-borrowed-20260928/)
include all four reports, the previous CSV, process resource measurements,
commands, and source/binary/dataset hashes. The latest version is base
`5d9b8f5ff98f13ba0a1d5ee65f9d53dd57456997` plus production patch
`586bbd53868e`.
