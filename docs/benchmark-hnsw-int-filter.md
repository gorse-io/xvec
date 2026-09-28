# HNSW integer filtering on Cohere 100K

`NewIntFilterPerformanceCase` compares xvec and zvec using FP16 HNSW and
`id >= dataset_size * filter_rate`. **Filter rate is the excluded fraction**;
for example `filter_rate=0.999` means `id >= 99900`, matching 0.1% (100 rows).
The nine matching percentages mirror the previous label-filter evaluation,
but the predicate and matching document sets differ.

## Results

QPS and P99 are from the 8-worker concurrent phase. Recall@100 is measured
against locally generated exact ground truth across all 1,000 serial queries.
These are fixed-EF, single-run measurements, not matched-recall comparisons or
confidence intervals. Each backend uses a fresh integer-indexed collection;
these are not before/after measurements on the label-filter graphs.
At 0.1%, only 100 documents match K=100, so 100% recall primarily verifies
that all candidates are returned; it does not validate ranking quality.

| Matching documents | Filter | xvec QPS | zvec QPS | zvec/xvec | xvec Recall@100 (%) | zvec Recall@100 (%) | xvec P99 (ms) | zvec P99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | `id >= 99900` | 19482.95 | 7872.05 | 0.40× | 100.000 | 100.000 | 2.096 | 3.208 |
| 0.2% | `id >= 99800` | 12049.10 | 7649.95 | 0.63× | 99.989 | 99.946 | 3.482 | 3.172 |
| 0.5% | `id >= 99500` | 3162.01 | 6211.67 | 1.96× | 99.970 | 99.891 | 14.259 | 4.068 |
| 1% | `id >= 99000` | 2110.18 | 5463.13 | 2.59× | 99.959 | 99.868 | 16.580 | 4.242 |
| 2% | `id >= 98000` | 804.02 | 3630.11 | 4.51× | 99.970 | 99.837 | 29.630 | 5.974 |
| 5% | `id >= 95000` | 277.39 | 1407.49 | 5.07× | 99.965 | 99.827 | 55.104 | 10.797 |
| 10% | `id >= 90000` | 132.99 | 687.86 | 5.17× | 99.961 | 99.783 | 96.977 | 19.565 |
| 20% | `id >= 80000` | 85.76 | 330.98 | 3.86× | 99.884 | 99.752 | 154.675 | 38.920 |
| 50% | `id >= 50000` | 54.95 | 681.16 | 12.40× | 99.806 | 99.665 | 194.909 | 19.196 |

xvec leads at 0.1% and 0.2% matching documents; zvec leads at the remaining
seven percentages. The largest throughput gap is at 50%: zvec reaches
681.16 QPS versus xvec's 54.95
(12.40×). Concurrent P99 is
194.909 ms for xvec versus
19.196 ms for zvec. xvec's Recall@100 is
99.806% versus 99.665% for zvec;
the comparison therefore includes a small recall difference.

| Backend | Insert (s) | Optimize (s) | Total load (s) | Build peak RSS (MiB) | Query peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| xvec | 7.54 | 152.46 | 160.01 | 1766.41 | 1084.16–1259.43 |
| zvec | 10.44 | 93.75 | 104.25 | 782.38 | 127.51–294.02 |

| Backend | Whole query-process wall time range (s) |
| --- | ---: |
| xvec | 168.43–254.63 |
| zvec | 34.16–49.24 |

Build and query RSS are separate process high-water marks. Query RSS includes
loading test queries and ground truth and opening the index; it is not
steady-state index memory. Build values repeated in the CSV represent the same
single build per backend, not nine independent builds.

## Integer-index observations

The query-process wall time includes index opening, input loading, warmup,
concurrent search, cooldown, serial evaluation and teardown. It is recorded
separately in the CSV and must not be treated as a direct cold-open timer.
Opening the xvec integer collection repeatedly incurred substantial startup
work before the measured query phase.

Static inspection identifies two paths to profile next:

- [`InvertedIndex.searchRange`](../internal/db/sqlengine/inverted.go) binary
  searches the ordered terms, then unions one posting for every matching
  integer value. With unique IDs, the nine ranges require 100 through 50,000
  postings per query. [`Bitmap.Or`](../internal/ailego/container/bitmap.go)
  clones each source bitmap before merging. This cost differs from looking
  up a single equal-label posting.
- [`OpenInvertedIndex`](../internal/db/sqlengine/inverted_pebble.go) reads and
  validates 100,000 distinct term postings. Its `bitmapSubset` check creates
  dense snapshots of both the posting and the same non-NULL row-domain bitmap
  on every call, repeatedly traversing that domain.

These are code-based explanations to investigate, not measured CPU-profile
attributions. This evaluation keeps the vector library unchanged and does not
isolate the proportion of time spent filtering versus searching vectors.

## Ground truth

Small Cohere does not publish integer-filter ground truth. This evaluation
explicitly uses **locally generated exhaustive cosine neighbors**, rather than
relabeling unfiltered or label-filter neighbors as integer-filter truth.

- Source: the same 100,000 original FP32 training vectors and 1,000 test vectors
  as the preceding label benchmark; dimension 768, IDs 0 through 99,999.
- Computation: convert source values to float64, normalize, and exhaustively
  score every matching document for every query. Select the top 100 by cosine
  similarity, breaking ties by ascending integer ID. No ANN or FP16 encoding
  is used to generate truth.
- Validation: all source IDs are unique and complete, vectors are finite and
  nonzero, and every output query has 100 unique neighbors meeting its integer
  condition. For queries 0, 137 and 999, an independent full-coordinate
  reduction and full sort verifies all nine filtered results. Unfiltered
  top-100 sets for those queries also match the published truth exactly.
- All nine generated Parquet files, SHA-256 hashes, Python dependency versions,
  generation script and validation output are archived.

The harness now supports an explicit `--local-int-ground-truth` flag, requiring
`--skip-download`. It preserves the default rejection of unpublished dataset
combinations and records `local_int_ground_truth: true` in each raw report.
The CSV identifies the source and the exact ground-truth filename.

The archived measurements were taken on `bc87d5a` plus the recorded patches.
The standalone filtering PR is based on `main` and excludes the Vamana changes
from PR #97. The archives retain their actual measured revisions; these results
have not been rerun on the standalone PR head.

## Method

- `Small Cohere (768dim, 100K)`, cosine, `NewIntFilterPerformanceCase`.
- FP16 HNSW: M=50, EFConstruction=500, EFSearch=300, K=100; rotation and
  refinement disabled, mmap enabled, ID-only query results.
- Fresh collection per backend, including a range-enabled inverted index on
  the integer `id` field. Load batch size 100, maximum documents per segment
  10,000,000, 8 optimize workers. Reuse each collection across its nine filters.
- Each query condition opens in a fresh process, warms up with 100 queries,
  runs 8 concurrent workers for 30 seconds, cools down for 3 seconds, then
  evaluates all 1,000 queries serially. Seed 0; alternate backend order between
  percentages. Filesystem caches are not flushed.
- Machine: e2-standard-8, AMD EPYC 7B12, CPUs 0–7, Go 1.27.1,
  CGO_ENABLED=0, GOMAXPROCS=8, GOMEMLIMIT=24GiB. Both use the same Go harness.
- Resources use `wait4`. Data generation, validation, downloads and compilation
  are excluded from measurements.

Both backends use their default filter planner with a brute-force-by-keys
ratio of 0.1. At most 10% matching documents uses candidate linear search;
20% and 50% use filtered graph search. This evaluates default HNSW-collection
filtering, including the selective-filter fallback.

Concurrent QPS is the main throughput metric. Serial QPS includes the harness's
recall calculation; serial latency measures only the search call. FP16
approximation can lower recall even when candidates are scanned linearly.

xvec uses base `bc87d5a7db1fde4251574fe458b520485cb2c92f` with the archived patch
`9c8ec3044bf43841eab5f9611be1bda6c5cc2ef4b93e7bc2fa4c206485659e8e`. Its vector-search implementation is unchanged from
`bc87d5a7db1fde4251574fe458b520485cb2c92f+filter50.cab34db9caab`; this turn adds only benchmark support
for explicitly identified local integer ground truth. zvec uses the unmodified
zvec-go v0.7.0 binding and official v0.7.0 native library, with the same verified
native-library hash as the preceding label comparison.

## Artifacts and checks

The benchmark-harness tests pass, including rejection of implicit unpublished
ground truth and validation/reporting of the explicit local option. The result
audit checks both builds and all eighteen query runs, CSV/raw-report agreement,
filter thresholds, matching counts, ground-truth provenance, and source,
binary, dataset and native-library hashes.

- [CSV](benchmark-hnsw-int-filter.csv)
- [Raw reports, commands, exact neighbors and generation code](benchmark-runs/hnsw-int-fp16-20260928/)
- [Previous label-filter evaluation](benchmark-hnsw-label-filter.md)
