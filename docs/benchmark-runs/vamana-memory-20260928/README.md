# Vamana memory rerun, 2026-09-28

These artifacts support the four updated xvec rows in
[`../../benchmark-vamana.csv`](../../benchmark-vamana.csv).
The zvec rows are unchanged historical measurements.

- `xvec-*.json`: original benchmark reports and separate `*.resources.json`
  files from `wait4` (KiB RSS, seconds for wall/user/system time).
- `xvec-*.log`: benchmark output.
- `metadata.json`: build version, command lines, environment settings, binary,
  source-patch and dataset hashes, and successful exit statuses.
- `source.patch`: the exact production-code patch applied to the recorded base.
- `previous.csv`: measurements before this rerun.
- `run.py`: the sequential runner and resource measurement implementation.
- `update_csv.py`: validation and mapping of raw metrics into CSV fields.

The scripts record this machine's paths; adjust `root` and `repo` when
reproducing elsewhere. Use the source patch with the recorded base commit,
build with `CGO_ENABLED=0`, and download the three files from
`https://assets.zilliz.com/benchmark/cohere_small_100k/` into `dataset/` before
running. Downloads and compilation occur outside the measured child processes.

See [the analysis](../../benchmark-vamana-memory.md) for results and limitations.
