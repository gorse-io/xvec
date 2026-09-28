# HNSW label filtering on Cohere 100K

This comparison runs `LabelFilterPerformanceCase` with FP16 scalar quantization
on xvec and zvec. The nine label percentages are the fraction of documents that
**match** the filter, not the fraction excluded. Each label has its own published
filtered ground truth; the unfiltered `neighbors.parquet` is not used.

## Results

QPS and P99 below are from the 8-worker concurrent phase; recall is from
all 1,000 serial queries. zvec is 7.52–53.12 times faster in these measurements.
xvec's recall is equal or up to 0.183 percentage points higher. These fixed-EF
results are not comparisons at an identical recall target.

| Matching labels | xvec QPS | zvec QPS | zvec/xvec | xvec recall (%) | zvec recall (%) | xvec P99 (ms) | zvec P99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 149.95 | 7964.79 | 53.12× | 100.000 | 100.000 | 83.705 | 3.118 |
| 0.2% | 146.74 | 7335.97 | 49.99× | 99.983 | 99.933 | 84.207 | 3.433 |
| 0.5% | 140.40 | 7240.86 | 51.57× | 99.961 | 99.865 | 89.999 | 3.300 |
| 1% | 142.06 | 6079.44 | 42.80× | 99.964 | 99.874 | 90.874 | 3.854 |
| 2% | 128.92 | 4325.64 | 33.55× | 99.970 | 99.847 | 93.916 | 4.942 |
| 5% | 103.36 | 1513.05 | 14.64× | 99.965 | 99.817 | 109.475 | 9.897 |
| 10% | 78.28 | 769.92 | 9.84× | 99.961 | 99.794 | 139.024 | 16.280 |
| 20% | 50.92 | 382.87 | 7.52× | 99.901 | 99.748 | 201.673 | 31.103 |
| 50% | 33.31 | 693.76 | 20.83× | 99.833 | 99.650 | 311.852 | 18.597 |

Build measurements are separate from the query processes:

| Backend | Insert (s) | Optimize (s) | Total load (s) | Build peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| xvec | 10.28 | 143.91 | 154.19 | 1466.30 |
| zvec | 7.55 | 95.73 | 103.33 | 789.82 |

xvec's low-selectivity throughput stays near 140–150 QPS from 0.1% to 1%,
while zvec reaches 6,079–7,965 QPS. At the 20% graph-search point both are
slower than at the 10% linear-search point. At 50%, zvec throughput recovers,
while xvec falls further. This run does not attribute those transitions to
individual implementation costs.

## Method

- Dataset: `Small Cohere (768dim, 100K)`, cosine distance, 100,000 training
  vectors and all 1,000 test queries. Label counts and every ground-truth
  neighbor's label were validated against `scalar_labels.parquet`.
- HNSW: M=50, EFConstruction=500, EFSearch=300, K=100; FP16 scalar quantization,
  rotation disabled, refinement disabled, mmap enabled, ID-only results.
- Machine: e2-standard-8, AMD EPYC 7B12, CPU affinity 0–7; Go 1.27.1,
  CGO_ENABLED=0, GOMAXPROCS=8, GOMEMLIMIT=24GiB. Both backends use the same
  Go benchmark binary, 8 optimize workers, and 8 query workers.
- Build one fresh collection per backend with an inverted index on `labels`.
  Reuse that collection across all nine filters, reopening in a fresh process
  for each measurement. Alternate backend order between adjacent percentages.
  Filesystem caches are not flushed between runs.
- Each query run warms up with 100 queries, measures concurrent search for
  30 seconds, waits 3 seconds, then measures all 1,000 serial queries and recall.
  The query seed is 0. Each condition is measured once.
- Build and query process resources are measured separately using `wait4`.
  Query peak RSS includes loading the query data and ground truth and opening
  the index; it is not steady-state index memory. Download and compilation
  are excluded. Build metrics repeated in the CSV refer to the same single
  build per backend, not nine independent builds.

xvec is commit `bc87d5a7db1fde4251574fe458b520485cb2c92f`. zvec uses the
unmodified zvec-go v0.7.0 binding and the official v0.7.0 Linux x64 native
release. The native FP16 quantization and disabled rotation were checked
through C API getters; release and dataset checksums are archived.

Both collections use their default filter planner. xvec's
`BruteForceByKeysRatio` and zvec's native getter both report 0.1: matching at
most 10% of documents selects a linear scan over the filtered candidates;
20% and 50% use filtered HNSW traversal. Thus this measures filtering on HNSW
collections with the default planner, including its selective-filter fallback.
It does not force graph traversal at every selectivity.

Concurrent QPS is the primary throughput comparison. Serial QPS includes
the harness's recall calculation in its elapsed time, whereas serial query
latency times only the search call. Recall is measured against the top 100
published neighbors for the selected label. FP16 approximation can affect
recall even when the candidate set is scanned linearly.

The [CSV](benchmark-hnsw-label-filter.csv) preserves all timings and counts.
[Raw reports and commands](benchmark-runs/hnsw-label-fp16-20260928/) preserve
the two builds and eighteen query runs. Results are single measurements, not
confidence intervals or comparisons at a matched recall target.

## Code observation

At this revision, `evaluateSegmentFilters` builds a live-document map and
iterates segment/global document lists for every filtered query.
`evaluateFilterDocuments` also loops over the full document slice, even when
an inverted index supplies a small candidate bitmap. This leaves work that
scales with collection size when very few labels match. These are observations
from the implementation in `collection.go`, not a CPU-profile attribution of
the measured gap; the benchmark does not isolate filtering, distance scoring,
and result materialization costs.
