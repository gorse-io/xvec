# HNSW label filtering on Cohere 100K

This comparison runs `LabelFilterPerformanceCase` with FP16 scalar quantization
on xvec and zvec. The nine percentages describe documents that **match** the
filter. Each label uses its published filtered ground truth; the unfiltered
`neighbors.parquet` is not used.

## Results after optimizing xvec filtering

The optimized xvec improves concurrent throughput by **3.51–220.22×** over its
baseline. At 0.1–1% matching labels, xvec reaches or exceeds the rerun zvec QPS;
at 2–50%, zvec remains **2.07–2.59×** faster. All nine xvec recall values are
unchanged on the same persisted graph. Recall ranges from 99.833% to 100%.
These fixed-EF results do not compare the backends at an identical recall target.

QPS is from the 8-worker concurrent phase; recall is from all 1,000 serial
queries. Both backends were rerun after the xvec change, with their original
collections and identical query settings.

| Matching labels | xvec before QPS | xvec after QPS | Speedup | zvec rerun QPS | xvec recall (%) | zvec recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 149.95 | 33020.39 | 220.22× | 7594.82 | 100.000 | 100.000 |
| 0.2% | 146.74 | 23670.70 | 161.31× | 7581.08 | 99.983 | 99.933 |
| 0.5% | 140.40 | 12330.55 | 87.83× | 7096.68 | 99.961 | 99.865 |
| 1% | 142.06 | 6696.53 | 47.14× | 6126.82 | 99.964 | 99.874 |
| 2% | 128.92 | 2124.23 | 16.48× | 4421.38 | 99.970 | 99.847 |
| 5% | 103.36 | 615.07 | 5.95× | 1551.24 | 99.965 | 99.817 |
| 10% | 78.28 | 317.73 | 4.06× | 751.88 | 99.961 | 99.794 |
| 20% | 50.92 | 178.91 | 3.51× | 370.41 | 99.901 | 99.748 |
| 50% | 33.31 | 241.10 | 7.24× | 623.55 | 99.833 | 99.650 |

Concurrent tail latency remains a gap at higher matching percentages:

| Matching labels | xvec P99 (ms) | zvec P99 (ms) |
| --- | ---: | ---: |
| 0.1% | 0.856 | 3.343 |
| 0.2% | 1.063 | 3.329 |
| 0.5% | 2.326 | 3.402 |
| 1% | 4.070 | 3.782 |
| 2% | 8.614 | 4.785 |
| 5% | 21.817 | 9.695 |
| 10% | 41.224 | 17.534 |
| 20% | 72.456 | 33.664 |
| 50% | 60.667 | 21.972 |

Across the nine query processes, xvec peak RSS is 1,010.23–1,117.70 MiB
(previously 1,045.21–1,369.15 MiB), compared with zvec's 129.26–404.11 MiB.
This still leaves a substantial memory gap. Peak RSS includes loading queries
and ground truth and opening the index; it is not steady-state index memory.

The following **original build measurements** are retained for context.
No index was rebuilt for the optimization rerun, so these are not build-time
or build-memory measurements of the optimized code. The CSV explicitly records
`build_backend_version` separately from the query `backend_version`.

| Backend | Insert (s) | Optimize (s) | Total load (s) | Build peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| xvec | 10.28 | 143.91 | 154.19 | 1466.30 |
| zvec | 7.55 | 95.73 | 103.33 | 789.82 |

## Implementation changes

The local zvec implementation provided the reference: in
`src/db/sqlengine/planner/doc_filter.cc`,
`DocFilter::get_bf_by_keys_and_update` extracts keys directly from a selective
inverted-index result. `vector_recall_node.cc` passes those keys through
`query_params.bf_pks` to vector search.

Previously, xvec rebuilt live-document maps, traversed full segment/global
lists, and scanned all quantized vector keys even when few labels matched.
The changes remove that work from the selective-filter path:

- [Snapshot filter evaluation](../collection_filter.go) reuses immutable live
  masks and document positions. It intersects only selected candidates with
  visibility masks. For a single fully live segment, its local result also
  serves as the global result, avoiding another map and candidate copy.
- [Filter evaluation](../collection.go) iterates inverted bitmap candidates
  directly. Exact candidate sets retain their bitmap and bypass repeated SQL
  evaluation; partial or unindexed conditions still use forward evaluation.
  A cached segment-local document-position map supports bitmap membership.
- [Quantized linear search](../internal/core/algorithm/quantized_flat_searcher.go)
  accepts candidate keys and looks up their vector positions directly. Missing
  keys and duplicate keys are handled while preserving filters, radius, top-K,
  and the existing refinement path.

SQL null semantics, visibility of deleted documents, and mixed indexed/forward
conditions remain covered by tests. The change does not alter precision,
distance scoring, HNSW parameters, graph topology, the file format, or GC settings.
The benchmark measures the combined change; it does not isolate the speedup
contributed by each part or attribute the remaining gap to a specific cost.

## Method

- Dataset: `Small Cohere (768dim, 100K)`, cosine distance, 100,000 training
  vectors and all 1,000 test queries. Label counts and every ground-truth
  neighbor's label were validated against `scalar_labels.parquet`.
- HNSW: M=50, EFConstruction=500, EFSearch=300, K=100; FP16 scalar quantization,
  rotation disabled, refinement disabled, mmap enabled, ID-only results.
- Machine: e2-standard-8, AMD EPYC 7B12, CPU affinity 0–7; Go 1.27.1,
  CGO_ENABLED=0, GOMAXPROCS=8, GOMEMLIMIT=24GiB. Both backends use the same
  Go benchmark binary, 8 optimize workers, and 8 query workers.
- The baseline built one fresh collection per backend with an inverted index
  on `labels`. Both rounds reuse those collections across all nine filters,
  reopening in a fresh process for each measurement. Backend order alternates
  between adjacent percentages. Filesystem caches are not flushed.
- Each query run warms up with 100 queries, measures concurrent search for
  30 seconds, waits 3 seconds, then measures all 1,000 serial queries and recall.
  The query seed is 0. Each condition is measured once per round.
- Build and query process resources are measured separately using `wait4`.
  Downloads and compilation are excluded. Repeated build metrics in the CSV
  refer to the same original build per backend, not nine independent builds.

The xvec base is `bc87d5a7db1fde4251574fe458b520485cb2c92f`. The measured source
adds the archived production patch with SHA-256
`10fe764cfac296ac401afdd491c1e659a07a410f344d830c8fe28bdac44f5db5`.
The CSV query version appends `+filter.10fe764cfac2` to the base revision.
zvec uses the unmodified zvec-go v0.7.0 binding and official v0.7.0 Linux x64
native release. Native FP16 quantization and disabled rotation were checked
through C API getters; release and dataset checksums are archived.

Both collections use their default filter planner. xvec's
`BruteForceByKeysRatio` and zvec's native getter report 0.1: matching at most
10% of documents selects linear search over filtered candidates; 20% and 50%
use filtered HNSW traversal. This measures the default filtering behavior of
HNSW collections, including the selective-filter fallback, rather than forcing
graph traversal at every percentage.

Concurrent QPS is the primary throughput comparison. Serial QPS includes the
harness's recall calculation in elapsed time, whereas serial query latency
times only the search call. Recall uses the selected label's published top 100
neighbors. FP16 approximation can affect recall even during linear search.
These are single measurements without confidence intervals; the small QPS
lead at 1% should be interpreted accordingly.

## Validation and artifacts

`go test ./...` passed on the measured production implementation. New tests
compare snapshot/bitmap filtering against forced forward SQL evaluation,
including multiple segments, deleted-document visibility, nulls, arrays, and
partial-index expressions. Quantized key search is checked against a full
filtered scan for L2/IP/cosine and INT4/INT8/FP16, with radius, duplicate/missing
keys, invalid inputs, and cancellation. Additional single-segment and bitmap
cases passed a targeted rerun before the benchmark binary was built.

- [Current CSV](benchmark-hnsw-label-filter.csv): all 18 query results, timings,
  counts, recall, process resources, and separately identified original builds.
- [Optimization reports and commands](benchmark-runs/hnsw-label-opt-20260928/):
  raw rerun output, source patch, hashes, validation logs, and update scripts.
- [Previous comparison](benchmark-runs/hnsw-label-opt-20260928/previous.md) and
  [previous CSV](benchmark-runs/hnsw-label-opt-20260928/previous.csv): retained
  before-optimization results.
- [Original build and query reports](benchmark-runs/hnsw-label-fp16-20260928/):
  the two builds and the baseline eighteen query runs.
