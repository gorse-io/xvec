# HNSW integer filtering on Cohere 100K

`NewIntFilterPerformanceCase` compares FP16 HNSW using
`id >= dataset_size * filter_rate`. **Filter rate is the excluded fraction**;
`filter_rate=0.999` matches 0.1% (100 rows). The tables show matching percentages.

## End-to-end rerun after integer range aggregation, 2026-09-30

The xvec integer-range aggregation revision and zvec v0.7.0 were rerun on their original persisted
100K integer-indexed collections. The graphs were not rebuilt or re-optimized.
All nine conditions use fresh query processes. QPS and P99 come from the
8-worker concurrent phase; Recall@100 uses all 1,000 serial queries against
locally generated exhaustive ground truth.

| Matching documents | Filter | xvec QPS | zvec QPS | zvec/xvec | xvec Recall@100 (%) | zvec Recall@100 (%) | xvec P99 (ms) | zvec P99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | `id >= 99900` | 25265.40 | 10234.98 | 0.41× | 100.000 | 100.000 | 1.211 | 2.432 |
| 0.2% | `id >= 99800` | 15526.45 | 9419.15 | 0.61× | 99.989 | 99.946 | 1.728 | 2.587 |
| 0.5% | `id >= 99500` | 8806.20 | 7611.97 | 0.86× | 99.970 | 99.891 | 3.580 | 3.152 |
| 1% | `id >= 99000` | 6264.32 | 6559.28 | 1.05× | 99.959 | 99.868 | 3.554 | 3.454 |
| 2% | `id >= 98000` | 2807.03 | 4572.43 | 1.63× | 99.970 | 99.837 | 8.452 | 4.693 |
| 5% | `id >= 95000` | 978.40 | 1983.76 | 2.03× | 99.965 | 99.827 | 18.540 | 8.272 |
| 10% | `id >= 90000` | 564.17 | 1068.51 | 1.89× | 99.961 | 99.783 | 30.740 | 12.719 |
| 20% | `id >= 80000` | 418.96 | 561.12 | 1.34× | 99.884 | 99.752 | 37.866 | 21.555 |
| 50% | `id >= 50000` | 688.86 | 1015.85 | 1.47× | 99.806 | 99.665 | 30.167 | 12.720 |

At 50% matching, xvec reaches **688.86 QPS**, compared with
zvec's **1015.85 QPS**. Their Recall@100 values are
99.806% and 99.665%, respectively.

## Controlled xvec before/after comparison

The pre-optimization PR source was also rerun at 10%, 20%, and 50%, using
the identical xvec collection, query vectors and ground truth. Only the query
binary changes. These end-to-end numbers are distinct from the scalar-filter
microbenchmarks. Both xvec revisions return the same aggregate recall.

| Matching documents | xvec before QPS | xvec after QPS | Speedup | Before P99 (ms) | After P99 (ms) | Recall@100 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 176.25 | 564.17 | 3.20× | 78.737 | 30.740 | 99.961 |
| 20% | 119.73 | 418.96 | 3.50× | 111.345 | 37.866 | 99.884 |
| 50% | 79.41 | 688.86 | 8.68× | 139.313 | 30.167 | 99.806 |

## Portable prefetch comparison, 2026-09-30

The PR subsequently removes the AMD64-specific Go and assembly prefetch files
and uses the portable helper on every architecture, including `noasm` builds.
The portable helper synchronously reads one byte per requested cache line;
the preceding AMD64 implementation issues `PREFETCHT0` hints.

Both implementations were rerun at 20% and 50% matching on the same original
persisted xvec collection, without a profiler. All dataset and xvec index
artifact hashes are unchanged. Each run uses the parameters documented below:
100 warmup queries, 8 workers for 30 seconds, 3 seconds of cooldown, and all
1,000 serial recall queries. Run order is AMD64 then portable at 50%, followed
by portable then AMD64 at 20%. Filesystem caches are not flushed.

| Matching documents | AMD64 QPS | Portable QPS | Change | AMD64 P99 (ms) | Portable P99 (ms) | Recall@100 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 437.30 | 402.43 | -7.97% | 36.967 | 43.330 | 99.884 |
| 50% | 668.97 | 659.25 | -1.45% | 31.253 | 31.961 | 99.806 |

The portable revision is `4c8d22c42eee5205fb68b0a4093987e346f2d0c5`;
the AMD64 binary is the clean `bc810b5b0c948b4728fa1a997d1f58c16aba4e98`
build used in the range-aggregation rerun. These are single measurements
without confidence intervals; the observed QPS differences are not estimates
of a statistically established regression. Aggregate recall is identical
between implementations and to the preceding measurements.

The [prefetch comparison CSV](benchmark-hnsw-int-filter-prefetch.csv) records
all four runs. The nine-rate xvec/zvec table and range-aggregation before/after
table above retain their original measured source and results; they do not
represent the later portable revision. zvec and label filtering were not
rerun for this simplification. Raw reports, resources, binary hashes, and
reproduction scripts remain local under the ignored `docs/benchmark-runs/`.

## Implementation

Like zvec's pre-aggregated range postings, xvec now caches the union of every
256 ordered integer terms. Queries merge complete blocks and individual
boundary terms instead of unioning every matching posting. The cache applies
to range-enabled scalar integer fields and is rebuilt on sealing or opening,
without changing the persisted format. A 50%-matching range over 100K unique
integer terms uses 530 bitmap unions instead of 50,000.

Index-opening validation now checks a posting's set bits against the non-NULL
domain instead of rebuilding dense snapshots of the full domain for every
term. Both changes preserve exact candidates; vector scoring, EF, quantization
and refinement settings are unchanged.

## Method and provenance

- Dataset: Small Cohere (768dim, 100K), cosine, 100,000 training vectors,
  all 1,000 test queries; integer IDs 0 through 99,999.
- FP16 HNSW: M=50, EFConstruction=500, EFSearch=300, K=100; rotation and
  refinement disabled, mmap enabled, ID-only results.
- Same e2-standard-8 host, AMD EPYC 7B12, CPU affinity 0–7, Go 1.27.1,
  CGO_ENABLED=0, GOMAXPROCS=8, GOMEMLIMIT=24GiB; 8 query workers.
- Each condition warms up 100 queries, measures concurrent search for
  30 seconds, cools down for 3 seconds, then evaluates all 1,000 queries
  serially. Seed 0. Filesystem caches are not flushed.
- Execution order: 50%, 20%, 10%, 0.1%, 0.2%, 0.5%, 1%, 2%, 5% matching.
  Alternate xvec/zvec order at each rate; run the pre-optimization xvec
  condition immediately after each of the first three pairs.
- Single measurement per condition, without a profiler or confidence
  intervals. Fixed-EF comparison; the backends have slightly different recall.
- Both default planners use candidate linear search at at most 10%
  matching documents and filtered HNSW traversal at 20% and 50%.

The range-aggregation xvec and harness revision is `bc810b5b0c948b4728fa1a997d1f58c16aba4e98`;
the controlled pre-optimization revision is `0042d0f1ac276c930a576412ae32e0189162e3c3`.
These are clean standalone PR builds, excluding the Vamana changes from PR #97.
zvec uses the official v0.7.0 native release and unmodified zvec-go binding;
its native-library hash matches the previous evaluation.

The reused xvec collection was built on `bc87d5a7db1fde4251574fe458b520485cb2c92f+intfilter.9c8ec3044bf4`;
zvec's was built on v0.7.0. The xvec graph file and zvec scalar payload hashes
were unchanged after all 21 query processes. zvec rewrites container and chunk
update timestamps on close. Restoring only those two timestamps and the
container footer CRC reproduces each native index file's original SHA-256,
verifying that all other bytes, including graphs and vectors, are unchanged.
A separate one-query audit confirms these metadata writes and is excluded
from the results. Dataset, ground-truth, binaries and native-library hashes
are recorded and validated locally.

| Backend | Original load (s) | Original build peak RSS (MiB) | Rerun query peak RSS (MiB) | Whole query-process wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| xvec | 160.01 | 1766.41 | 1008.09–1262.36 | 38.54–51.95 |
| zvec | 104.25 | 782.38 | 169.19–302.94 | 33.96–42.31 |

Build values are the **original 2026-09-28 measurements**, repeated in the
CSV for context. No build time or build memory was measured for the optimized
source. Query RSS is a process high-water mark including index opening and
input loading, not steady-state index memory. Whole-process wall time includes
opening, warmup, concurrent and serial phases, and teardown; it is not a
standalone cold-open timer.

Concurrent QPS is the primary throughput metric. Serial QPS includes recall
calculation; serial latency times only the search call. At 0.1%, exactly
100 documents match K=100, so full recall primarily checks candidate inclusion.

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

## Results and checks

- [Range-aggregation xvec/zvec results](benchmark-hnsw-int-filter.csv): all 18
  query conditions, with separate query-source and collection-build versions.
- [Controlled xvec before/after results](benchmark-hnsw-int-filter-before-after.csv):
  six conditions at 10%, 20%, and 50% matching.
- [Label-filter comparison](benchmark-hnsw-label-filter.md).

All 21 query runs completed successfully. CSV parameters, metrics and process
resources were checked against the raw reports. Recall matches the original
results at every rate and matches between xvec revisions for all three
controlled conditions. Immutable graph/vector, dataset and ground-truth hashes
were verified. Raw reports, commands, exact neighbors and scripts remain local
under the ignored `docs/benchmark-runs/` directory.
