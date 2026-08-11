# vector-db-bench

`vector-db-bench` is a native Go benchmark driver for xvec. It follows the
VectorDBBench Cohere performance workload used by the Alibaba zvec benchmark
guide:

- Cohere 1M and 10M Parquet datasets;
- HNSW loading and optimization;
- serial Recall@K, QPS, and latency percentiles;
- sustained concurrent QPS and latency at multiple worker counts;
- JSON results containing the run configuration and host information.

The program downloads the same public VectorDBBench files from
`https://assets.zilliz.com/benchmark` unless `--skip-download` is set. Be aware
that the 1M shuffled training file is about 3.1 GB and the ten 10M shards total
substantially more.

## Build

```bash
go build -o vector-db-bench ./cmd/vector-db-bench
```

## Cohere 1M

The following configuration mirrors the published 1M Alibaba zvec benchmark. The
first invocation downloads the data, recreates the collection, loads it, and
runs both search phases.

```bash
./vector-db-bench \
  --path ./Performance768D1M \
  --case-type Performance768D1M \
  --num-concurrency 12,14,16,18,20 \
  --m 15 \
  --ef-search 180 \
  --output result-cohere-1m.json
```

To rerun only the search phases against that collection:

```bash
./vector-db-bench \
  --path ./Performance768D1M \
  --case-type Performance768D1M \
  --num-concurrency 12,14,16,18,20 \
  --m 15 \
  --ef-search 180 \
  --skip-drop-old \
  --skip-load \
  --output result-cohere-1m-search.json
```

## Cohere 10M

This mirrors the published INT8/refiner configuration:

```bash
./vector-db-bench \
  --path ./Performance768D10M \
  --case-type Performance768D10M \
  --num-concurrency 12,14,16,18,20 \
  --quantize-type int8 \
  --m 50 \
  --ef-search 118 \
  --is-using-refiner \
  --output result-cohere-10m.json
```

`--concurrency-duration` defaults to 30 seconds per worker count. It accepts
either a Go duration such as `45s` or a number of seconds such as `45`. Like
VectorDBBench, concurrent search runs before serial recall/latency measurement;
`--serial-cooldown` optionally inserts a quiet period between those phases.
`--load-limit` and `--query-limit` are useful for smoke tests; do not use them
for comparable published results. Use `--dry-run` to validate and print the
resolved configuration without downloading data.

## Dataset compatibility

The built-in cases use the VectorDBBench schema:

| File | Required columns |
| --- | --- |
| `shuffle_train*.parquet` | `id` (`INT64`), `emb` (`LIST<FLOAT>`) |
| `test.parquet` | `id` (`INT64`), `emb` (`LIST<FLOAT>`) |
| `neighbors.parquet` | `id` (`INT64`), `neighbors_id` (`LIST<INT64>`) |

A local custom dataset can be exercised without downloads:

```bash
./vector-db-bench \
  --path ./custom-collection \
  --case-type Custom \
  --dataset-dir ./dataset \
  --train-files train.parquet \
  --dimension 384 \
  --metric cosine \
  --skip-download
```

The JSON schema is versioned as `vector-db-bench/v1`. Its
`vectordbbench_metrics` object also exposes VectorDBBench-compatible metric
names such as `load_duration`, `qps`, `recall`, `conc_num_list`, and
`conc_qps_list`. Progress and the compact human-readable summary are written
to stderr, keeping stdout valid JSON when `--output` is omitted.
