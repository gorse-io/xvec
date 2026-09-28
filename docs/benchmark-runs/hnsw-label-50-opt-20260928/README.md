# HNSW filtering and prefetch rerun, 2026-09-28

This measures the 50%-focused optimization after the first
[filtered-query optimization](../hnsw-label-opt-20260928/).
Both backends reuse their original persisted FP16 Cohere 100K HNSW collections.
No graph is rebuilt; original build measurements retain their original version.

All eighteen query runs use 100 warmup queries, 30 seconds of 8-worker
concurrent search, a 3-second cooldown, and all 1,000 serial queries.
The 50% pair runs first, followed by the other eight matching percentages
in the original alternating backend order. Filesystem caches are not flushed.

- `previous.csv`, `previous.md`: the preceding comparison.
- Backend JSON/log/resource files: the new unprofiled measurements.
- `metadata.json`, `source.patch`, `build-info.txt`: exact production source,
  binary and dataset hashes, environment, commands, and exit statuses.
- `run.py`, `update_csv.py`, `update_report.py`, `validate_reports.py`:
  measurement, export and consistency-check procedures; their paths refer to
  this machine. `validation.log` records the completed consistency checks.
- `tests-full.log`, `tests-noasm.log`: full Go suite and portable-prefetch tests.

`before.cpu` and `step2.cpu` are separate diagnostic CPU profiles. Their reports
are `before-profile.*` and `step2-profile.*`; text summaries are
`profile-before-top.txt` and `profile-after-top.txt`. A temporary Go build overlay
adds `profile_init.go` (archived as `profile_init.go.txt`) to the benchmark
command: sampling starts 10 seconds
after process initialization and lasts 15 seconds, inside concurrent search.
The baseline profile uses the previous production patch; `step2` uses this
directory's production patch. `profile-command.json` records the baseline
diagnostic command; the latter substitutes the diagnostic binary/output/profile
paths. Diagnostic binaries add only profiling initialization; formal runs use
a normal build without the overlay. Diagnostic QPS is excluded from the CSV.

The retained changes handle empty writable segments in the single-data-segment
filter shortcut and follow zvec's filtered HNSW prefetch sequence: collect
unvisited neighbors, then issue CPU prefetch hints before batch scoring.
AMD64 uses bounded `PREFETCHT0`; other architectures and `noasm` builds retain
the portable fallback. Distance scoring, EF, precision and graph topology stay
unchanged. An exploratory norm-cache variant was not retained.

See [the comparison](../../benchmark-hnsw-label-filter.md) for the results.
