# vector-db-bench

`vector-db-bench` is a native Go benchmark driver for comparing
[xvec](https://github.com/gorse-io/xvec) and
[zvec-go](https://github.com/zvec-ai/zvec-go). It follows the VectorDBBench
vector search performance workloads:

- Cohere, LAION, BioASQ, and OpenAI Parquet datasets;
- HNSW or DiskANN loading and optimization;
- serial Recall@K, QPS, and latency percentiles;
- sustained concurrent QPS and latency at multiple worker counts;
- JSON results containing the run configuration and host information.

The program downloads the same public VectorDBBench files from
`https://assets.zilliz.com/benchmark` unless `--skip-download` is set. Large
cases can require hundreds of gigabytes of downloads and additional space for
the database collection.

## Build

```bash
CGO_ENABLED=0 go build -o vector-db-bench ./cmd/vector-db-bench
```

The pure-Go build keeps xvec and zvec-go in one binary without linking zvec at
build time. Running the `zvec` backend requires the zvec C API shared library.
Download the archive for your platform from the
[zvec-go v0.6.0 release](https://github.com/zvec-ai/zvec-go/releases/tag/v0.6.0),
extract it, and point `ZVEC_LIBRARY_PATH` to the extracted library or its
directory. The xvec backend does not require this library.

For example, on Linux x86-64:

```bash
export ZVEC_LIBRARY_PATH=/path/to/linux_amd64/libzvec_c_api.so
```

## Built-in datasets

The built-in cases mirror VectorDBBench's standard vector performance dataset
presets:

| Case type | Dataset | Dimension | Size | Metric |
| --- | --- | ---: | ---: | --- |
| `Performance768D100K` | Cohere | 768 | 100K | cosine |
| `Performance768D1M` | Cohere | 768 | 1M | cosine |
| `Performance768D10M` | Cohere | 768 | 10M | cosine |
| `Performance768D100M` | LAION | 768 | 100M | L2 |
| `Performance1024D1M` | BioASQ | 1024 | 1M | cosine |
| `Performance1024D10M` | BioASQ | 1024 | 10M | cosine |
| `Performance1536D50K` | OpenAI | 1536 | 50K | cosine |
| `Performance1536D500K` | OpenAI | 1536 | 500K | cosine |
| `Performance1536D5M` | OpenAI | 1536 | 5M | cosine |

Pass one of these names to `--case-type`. The dataset directory, dimensions,
metric, and training shards are resolved automatically.

## Cohere 1M example

The following configuration mirrors the published 1M Alibaba zvec benchmark. The
first invocation downloads the data, recreates the collection, loads it, and
runs both search phases.

```bash
./vector-db-bench xvec \
  --path ./Performance768D1M \
  --case-type Performance768D1M \
  --num-concurrency 12,14,16,18,20 \
  --m 15 \
  --ef-search 180 \
  --output result-cohere-1m.json
```

Run the same workload against zvec-go by changing only the backend and output
path:

```bash
./vector-db-bench zvec \
  --path ./Performance768D1M-zvec \
  --case-type Performance768D1M \
  --num-concurrency 12,14,16,18,20 \
  --m 15 \
  --ef-search 180 \
  --output result-zvec-cohere-1m.json
```

To rerun only the search phases against that collection:

```bash
./vector-db-bench xvec \
  --path ./Performance768D1M \
  --case-type Performance768D1M \
  --num-concurrency 12,14,16,18,20 \
  --m 15 \
  --ef-search 180 \
  --skip-drop-old \
  --skip-load \
  --output result-cohere-1m-search.json
```

## Cohere 10M example

This mirrors the published INT8/refiner configuration:

```bash
./vector-db-bench xvec \
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

## DiskANN comparison

Select DiskANN explicitly and pass the same construction and search parameters
to both backends. For example:

```bash
GOMAXPROCS=4 taskset -c 0-3 ./vector-db-bench xvec \
  --path ./Performance1536D50K-xvec-diskann \
  --case-type Performance1536D50K \
  --index-type diskann \
  --diskann-max-degree 100 \
  --diskann-build-list 50 \
  --diskann-pq-chunks 0 \
  --diskann-query-list 300 \
  --optimize-concurrency 4 \
  --num-concurrency 1,2,4 \
  --output result-xvec-openai-50k-diskann.json
```

Use the same command for `zvec`, changing only the backend, collection path,
and output path. When `--optimize-concurrency` is positive, it also pins zvec's
global query and optimize thread pools to that value. `taskset` and
`GOMAXPROCS` should still be identical for both processes.

## Dataset compatibility

The built-in cases use the VectorDBBench schema:

| File | Required columns |
| --- | --- |
| `shuffle_train*.parquet` | `id` (`INT64`), `emb` (`LIST<FLOAT>`) |
| `test.parquet` | `id` (`INT64`), `emb` (`LIST<FLOAT>`) |
| `neighbors.parquet` | `id` (`INT64`), `neighbors_id` (`LIST<INT64>`) |

A local custom dataset can be exercised without downloads:

```bash
./vector-db-bench xvec \
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
