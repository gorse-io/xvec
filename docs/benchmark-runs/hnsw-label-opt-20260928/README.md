# HNSW filtered-query optimization rerun, 2026-09-28

This reruns both backends' FP16 label-filter queries against exactly the same
persisted Cohere 100K collections as
[`../hnsw-label-fp16-20260928/`](../hnsw-label-fp16-20260928/).
No graph is rebuilt. All eighteen query runs use 100 warmup queries,
30 seconds of 8-worker concurrent search, a 3-second cooldown, and all 1,000
serial queries. Backend order alternates by matching-label percentage.

- `previous.csv` and `previous.md`: the pre-optimization comparison.
- `xvec-*.json`, `zvec-*.json`, `*.resources.json`, `*.log`: original reports,
  query-process resource measurements, and output from this rerun.
- `metadata.json`: source revision/patch, binary/native-library/dataset hashes,
  commands, execution order, environment, and exit statuses.
- `source.patch`: production changes against the recorded base, including the
  new snapshot-filter implementation; whitespace is preserved for its checksum.
- `build-info.txt`: benchmark binary build settings and dependencies.
- `tests-full.log`, `tests-new.log`: Go validation output.
- `run.py`, `update_csv.py`: exact measurement and CSV update procedures.
- `validate_reports.py`: checks CSV/raw-report agreement, unchanged recall and
  original build metrics, source/binary/native/dataset hashes, and patch validity.

The original two build reports remain in the previous run directory. The CSV's
`build_backend_version` and build metrics refer to those builds;
`backend_version` and query metrics refer to this rerun. Build times/RSS have
not been relabeled as measurements of the optimized implementation.

The scripts reference this machine's paths. To reproduce, apply the source patch
to the recorded base, build with `CGO_ENABLED=0`, and point `previous` at the
prior run's data, collections, metadata, and native library. Use a fresh output
directory. Downloads and compilation are outside the measured processes.

See [the comparison](../../benchmark-hnsw-label-filter.md) for results,
implementation changes, and limitations.
