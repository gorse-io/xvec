# Cohere 100K FP16 HNSW integer filtering, 2026-09-28

This runs `NewIntFilterPerformanceCase` using **locally generated exact ground
truth**, because Small Cohere has no published integer-filter neighbors.
See [the comparison](../../benchmark-hnsw-int-filter.md) and
[CSV](../../benchmark-hnsw-int-filter.csv).

- Two `*-build.*` sets: fresh 100K collections with an inverted index on `id`.
- Eighteen `*-int-match-*.{json,log,resources.json}` sets: query runs at matching
  fractions 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, and 50%. Filenames describe
  the matching percentage; `filter_rate` is the excluded fraction.
- `ground-truth/`: the nine generated neighbor files; each has 1,000 queries
  and 100 neighbors satisfying its `id >= 100000 * filter_rate` condition.
- `generate_truth.py`, `requirements.txt`, `generate-truth.log`, and
  `dataset-validation.json`: exact generation and checks. Float64 exhaustive
  cosine scoring uses original FP32 source values; ties use ascending ID.
  Three queries are verified independently via full-coordinate reductions,
  and against the original unfiltered published top-100 neighbor sets.
- `metadata.json`, `source.patch`, `build-info.txt`: measured source, binary,
  dataset/native-library hashes, commands, environment and execution order.
- `run.py`, `summarize.py`, `write_report.py`, `validate_reports.py`: run, export,
  report and audit scripts; `validation.log` records the final artifact checks.
- `harness-tests.log`: tests for the explicit local-ground-truth option.

The production vector-search implementation is the previous 50%-focused label
optimization. The benchmark harness adds `--local-int-ground-truth`, requiring
`--skip-download` and recording the source in each report. It does not download
or claim official ground truth for an unsupported dataset.

Both backends use FP16 HNSW, M=50, EFConstruction=500, EFSearch=300, K=100,
8 workers, 100 warmup queries, 30 seconds of concurrent search, a 3-second
cooldown, and all 1,000 serial queries. Each backend builds once and reopens
that collection in a fresh process per query run. Backend order alternates
across percentages. Build/query RSS are separate process high-water marks.

Scripts reference this machine's paths; adjust them for reproduction. Recreate
the recorded Python environment, supply the two source Parquet files, generate
neighbors, apply the recorded source patch to its base revision and build with
`CGO_ENABLED=0`. Supply the official zvec-go v0.7.0 native library, then use a
fresh output directory for `run.py`. Generation, downloads, compilation and
validation are outside measured processes. Results are single measurements.
