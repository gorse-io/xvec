# Cohere 100K HNSW label-filter benchmark, FP16, 2026-09-28

See the [comparison](../../benchmark-hnsw-label-filter.md) and
[CSV](../../benchmark-hnsw-label-filter.csv).

- `*-fp16-build.json`: one fresh 100K build per backend, with querying disabled.
- `*-fp16-label-*.json`: nine query-only runs per backend, each reusing its
  backend's built collection with a different label equality filter.
- `*.resources.json` and `*.log`: process resource measurements and raw output.
  Build and query RSS are separate high-water marks, not an end-to-end maximum.
- `metadata.json`: exact commands, versions, binary/native-library/dataset
  SHA-256 hashes, runtime settings, exit statuses, and measurement order.
- `build-info.txt`: the benchmark binary's Go build settings and dependencies.
- `dataset-validation.json`: actual label counts and validation of all published
  ground-truth neighbors against the label field.
- `native-validation.json`: native version, FP16 quantization, disabled rotation,
  and default brute-force-by-keys threshold verified through C API getters.
- `downloads.json` and `native-archive.sha256`: dataset download URLs and the
  native release archive's verified publisher checksum.
- `run_fp16.py`, `summarize.py`, `inspect_labels.go`, and `check_native.py`:
  measurement, result mapping, and validation procedures.

For reproduction, put the scripts in a fresh working directory and adjust
the `repo` path. Build `vector-db-bench` from the recorded xvec revision with
`CGO_ENABLED=0`. Download the recorded dataset files into `dataset/`, and extract
the official zvec-go v0.7.0 Linux x64 release into `native/` (including the
archive itself). Verify the publisher checksum before running. Run the dataset
and native validation helpers, then `run_fp16.py`; it refuses to overwrite
existing collections or reports. Downloads and compilation happen outside
the measured child processes.

The earlier interrupted FP32 build is excluded from these artifacts and results.
