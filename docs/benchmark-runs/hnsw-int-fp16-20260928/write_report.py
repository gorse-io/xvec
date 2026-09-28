import csv,json,pathlib
repo=pathlib.Path('/home/zhenghaoz/xvec');root=pathlib.Path(__file__).resolve().parent
rows=list(csv.DictReader((repo/'docs/benchmark-hnsw-int-filter.csv').open()));m=json.loads((root/'metadata.json').read_text())
f=lambda r,k:float(r[k])
table=['| Matching documents | Filter | xvec QPS | zvec QPS | zvec/xvec | xvec Recall@100 (%) | zvec Recall@100 (%) | xvec P99 (ms) | zvec P99 (ms) |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for x,z in zip(rows[::2],rows[1::2]):table.append(f"| {f(x,'matching_fraction')*100:g}% | `{x['filter_expression']}` | {f(x,'concurrent_qps'):.2f} | {f(z,'concurrent_qps'):.2f} | {f(z,'concurrent_qps')/f(x,'concurrent_qps'):.2f}× | {f(x,'recall_at_k_pct'):.3f} | {f(z,'recall_at_k_pct'):.3f} | {f(x,'concurrent_latency_p99_ms'):.3f} | {f(z,'concurrent_latency_p99_ms'):.3f} |")
report='''# HNSW integer filtering on Cohere 100K

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

'''+ '\n'.join(table)+'\n\n'
x50,z50=rows[-2:]
report+=f'''xvec leads at 0.1% and 0.2% matching documents; zvec leads at the remaining
seven percentages. The largest throughput gap is at 50%: zvec reaches
{f(z50,'concurrent_qps'):.2f} QPS versus xvec's {f(x50,'concurrent_qps'):.2f}
({f(z50,'concurrent_qps')/f(x50,'concurrent_qps'):.2f}×). Concurrent P99 is
{f(x50,'concurrent_latency_p99_ms'):.3f} ms for xvec versus
{f(z50,'concurrent_latency_p99_ms'):.3f} ms for zvec. xvec's Recall@100 is
{f(x50,'recall_at_k_pct'):.3f}% versus {f(z50,'recall_at_k_pct'):.3f}% for zvec;
the comparison therefore includes a small recall difference.

'''
report+='| Backend | Insert (s) | Optimize (s) | Total load (s) | Build peak RSS (MiB) | Query peak RSS range (MiB) |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
for b in ['xvec','zvec']:
 r=next(r for r in rows if r['backend']==b);rss=[f(r,'query_peak_rss_mib') for r in rows if r['backend']==b]
 report+=f"| {b} | {f(r,'insert_duration_sec'):.2f} | {f(r,'optimize_duration_sec'):.2f} | {f(r,'load_duration_sec'):.2f} | {f(r,'build_peak_rss_mib'):.2f} | {min(rss):.2f}–{max(rss):.2f} |\n"
report+='\n| Backend | Whole query-process wall time range (s) |\n| --- | ---: |\n'
for b in ['xvec','zvec']:
 walls=[f(r,'query_wall_seconds') for r in rows if r['backend']==b]
 report+=f"| {b} | {min(walls):.2f}–{max(walls):.2f} |\n"
report+='''
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

'''
report+=f'''xvec uses base `{m['xvec_revision']}` with the archived patch
`{m['source_patch_sha256']}`. Its vector-search implementation is unchanged from
`{m['library_implementation_version']}`; this turn adds only benchmark support
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
'''
(repo/'docs/benchmark-hnsw-int-filter.md').write_text(report)
print('Wrote integer-filter report.')
