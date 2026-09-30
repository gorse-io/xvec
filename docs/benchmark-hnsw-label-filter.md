# HNSW label filtering on Cohere 100K

This comparison runs `LabelFilterPerformanceCase` with FP16 scalar quantization
on xvec and zvec. The nine percentages describe documents that **match** the
filter. Each label uses its published filtered ground truth; the unfiltered
`neighbors.parquet` is not used.

## Results after the 50%-focused optimization

At 50% matching labels, xvec improves from **241.10 to
473.31 QPS (1.96×)**. Concurrent P99 falls from
60.667 to 27.377 ms.
zvec reaches 632.03 QPS in this rerun, leaving a
**1.34×** throughput gap instead of the previous 2.59×.
xvec's recall remains 99.833%, compared with zvec's
99.650%. These are fixed-EF results, not matched-recall comparisons.

Both backends were rerun on the original persisted collections. QPS below is
from the 8-worker concurrent phase; recall uses all 1,000 serial queries.
“Previous” refers to the first filter optimization, not the original
unoptimized implementation (which measured 33.31 QPS at 50%).

| Matching labels | xvec previous QPS | xvec current QPS | Speedup | zvec rerun QPS | xvec recall (%) | zvec recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 33020.39 | 36800.23 | 1.11× | 7599.57 | 100.000 | 100.000 |
| 0.2% | 23670.70 | 27155.73 | 1.15× | 7512.54 | 99.983 | 99.933 |
| 0.5% | 12330.55 | 13956.11 | 1.13× | 6953.11 | 99.961 | 99.865 |
| 1% | 6696.53 | 8295.99 | 1.24× | 5924.65 | 99.964 | 99.874 |
| 2% | 2124.23 | 2996.36 | 1.41× | 4157.08 | 99.970 | 99.847 |
| 5% | 615.07 | 731.83 | 1.19× | 1402.98 | 99.965 | 99.817 |
| 10% | 317.73 | 349.99 | 1.10× | 568.08 | 99.961 | 99.794 |
| 20% | 178.91 | 265.42 | 1.48× | 315.56 | 99.901 | 99.748 |
| 50% | 241.10 | 473.31 | 1.96× | 632.03 | 99.833 | 99.650 |

All nine xvec recall values are unchanged.

Throughput increased at every matching percentage in this round, but latency
did not improve uniformly. At 10%, xvec serial average latency rose from
19.732 to 21.488 ms, and concurrent P99
changed from 41.224 to 41.561 ms. zvec also varied
between rounds: its 10% QPS fell from 751.88 to 568.08.
These measurements do not establish uniform latency gains or eliminate run-to-run
variation.

| Matching labels | xvec P99 (ms) | zvec P99 (ms) |
| --- | ---: | ---: |
| 0.1% | 0.819 | 3.312 |
| 0.2% | 0.984 | 3.327 |
| 0.5% | 2.173 | 3.506 |
| 1% | 3.334 | 3.961 |
| 2% | 6.610 | 5.161 |
| 5% | 18.053 | 10.517 |
| 10% | 41.561 | 24.335 |
| 20% | 44.581 | 50.893 |
| 50% | 27.377 | 23.593 |

Across the nine query processes, xvec peak RSS is
985.54–1070.17 MiB, compared with zvec's
125.55–404.98 MiB. A substantial memory gap remains.
Peak RSS includes loading queries and ground truth and opening the index;
it is not steady-state index memory.

## Why the previous 50% result was slow

A separate 15-second CPU profile of the preceding implementation found
49.09% of samples in `evaluateSnapshotFilters` and its callees, with the global
ordinal lookup and match-map construction dominating that work. The collection
has one persisted 100K-document segment **and an empty writable segment**.
The shortcut required exactly one segment, so it failed to activate despite
there being only one segment containing documents. At 50%, that caused another
50K-element mapping pass for every query.

The same profile attributed 11.30% of CPU samples to the quantized-neighbor
prefetch helper and its callees. That helper performed synchronous byte reads
to warm cache lines, including neighbors already visited. These observations
identify xvec costs; they do not apportion the entire cross-backend gap.

## Implementation changes

The first round followed zvec's `DocFilter::get_bf_by_keys_and_update` and
`vector_recall_node.cc`: reuse snapshot visibility, enumerate inverted candidates
directly, retain exact bitmaps, and pass selected keys to quantized linear search.
This round retains those changes and adds:

- [Snapshot filtering](../collection_filter.go) recognizes one nonempty live
  segment even when empty writable segments are present. Its local predicate
  and ordinals are already global, avoiding the redundant mapping pass.
  Multiple segments and deleted-document visibility keep the general path.
- [Filtered HNSW traversal](../internal/core/algorithm/hnsw_quantized_searcher.go)
  gathers unvisited neighbors before prefetching them, following
  `dual_heap_search_neighbors` in zvec's
  `src/core/algorithm/hnsw/hnsw_algorithm.cc`.
- [AMD64 prefetch](../internal/core/algorithm/hnsw_prefetch_amd64.go) issues
  bounded `PREFETCHT0` hints like zvec's `ailego_prefetch`, replacing synchronous
  cache-line reads for quantized vectors. Other architectures and `noasm`
  retain the portable fallback. Explicit prefetch limits are respected.

The final change preserves precision, distance scoring, radius/refinement
behavior, EF, graph topology, the file format, and GC settings. The measurements
cover the combined changes, not controlled per-change speedup estimates.
Diagnostic profiled runs are separate and excluded from the CSV.

The archived measurements were taken on `bc87d5a` plus the recorded patches.
The standalone filtering PR is based on `main` and excludes the Vamana changes
from PR #97. The archives retain their actual measured revisions; these results
have not been rerun on the standalone PR head.

## Method

- Dataset: `Small Cohere (768dim, 100K)`, cosine distance, 100,000 training
  vectors and all 1,000 test queries. Label counts and every ground-truth
  neighbor's label were validated against `scalar_labels.parquet`.
- HNSW: M=50, EFConstruction=500, EFSearch=300, K=100; FP16 scalar quantization,
  rotation disabled, refinement disabled, mmap enabled, ID-only results.
- Machine: e2-standard-8, AMD EPYC 7B12, CPU affinity 0–7; Go 1.27.1,
  CGO_ENABLED=0, GOMAXPROCS=8, GOMEMLIMIT=24GiB. Both backends use the same
  Go benchmark binary, 8 optimize workers, and 8 query workers.
- The baseline built one collection per backend with an inverted index on
  `labels`. Subsequent rounds reuse those exact collections and reopen them
  in a fresh process per measurement. This round runs the 50% pair first,
  then the other eight rates in the original alternating backend order.
  Filesystem caches are not flushed.
- Each query run warms up with 100 queries, measures concurrent search for
  30 seconds, waits 3 seconds, then measures all 1,000 serial queries and recall.
  The seed is 0. Each condition is measured once per round, without a profiler.
- Process resources are measured using `wait4`. Download and compilation are
  excluded. Build metrics remain the original build measurements, with their
  original `build_backend_version`, separately from the query `backend_version`.

The xvec base is `bc87d5a7db1fde4251574fe458b520485cb2c92f`. The measured source
adds the archived production patch with SHA-256
`cab34db9caab697916bcc7dfa401e483109e68f39adce694600209f7d469efd1`.
The query version appends `+filter50.cab34db9caab` to the base.
zvec uses the unmodified zvec-go v0.7.0 binding and official v0.7.0 Linux x64
native release. Native FP16 quantization and disabled rotation were checked
through C API getters; release and dataset checksums are archived.

Both default planners have a brute-force-by-keys ratio of 0.1. Matching at most
10% selects linear search over filtered candidates; 20% and 50% use filtered
HNSW traversal. These results include the default selective-filter fallback
and do not force graph traversal at every percentage.

Concurrent QPS is the primary throughput comparison. Serial QPS includes the
harness's recall calculation in elapsed time; serial latency times only the
search call. Recall uses the label's published top 100 neighbors. Results are
single measurements without confidence intervals, so small differences should
not be interpreted as stable advantages.

The following **original build measurements** are retained for context.
No index was rebuilt for either optimization rerun; these do not measure build
time or build memory of the optimized implementation.

| Backend | Insert (s) | Optimize (s) | Total load (s) | Build peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| xvec | 10.28 | 143.91 | 154.19 | 1466.30 |
| zvec | 7.55 | 95.73 | 103.33 | 789.82 |

## Validation and artifacts

`go test ./...` passed on the measured source. Quantized HNSW tests also pass
with `-tags=noasm`. The new regression cases cover empty segments before and
after the data segment, and identical FP16 filtered-HNSW results with prefetch
disabled, automatic, or maximally requested, including radius searches.
The prior filter/visibility/SQL-semantic and candidate-key tests remain in place.
All eighteen reports, CSV fields, preserved build measurements and source,
binary, native-library and dataset hashes are checked together.

- [Current CSV](benchmark-hnsw-label-filter.csv): all 18 query measurements.

Raw reports, profiles, patches, commands and previous comparisons are retained
locally and are not included in the repository.
