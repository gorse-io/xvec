# vector-db-bench

`vector-db-bench` is a native Go benchmark driver for comparing
[xvec](https://github.com/gorse-io/xvec) and
[zvec-go](https://github.com/zvec-ai/zvec-go). It follows the VectorDBBench
vector and full-text search performance workloads:

- Cohere, LAION, BioASQ, and OpenAI Parquet datasets;
- MS MARCO and HotpotQA BM25 datasets with semantic qrels;
- Flat, HNSW, IVF, DiskANN, or Vamana loading and optimization;
- serial Recall@K, QPS, and latency percentiles;
- FTS Recall@K, MRR@K, and NDCG@K;
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

## Full-text search benchmark

The `FTSBm25Performance` case mirrors VectorDBBench's June 2026 full-text
search workload. It supports the same six dataset presets:

| Dataset preset | Documents |
| --- | ---: |
| `MS MARCO Small (100K documents)` | 100K |
| `MS MARCO Medium (1M documents)` | 1M |
| `MS MARCO Large (8.8M documents)` | 8,841,823 |
| `HotpotQA Small (100K documents)` | 100K |
| `HotpotQA Medium (1M documents)` | 1M |
| `HotpotQA Large (5.2M documents)` | 5,233,329 |

VectorDBBench obtains these corpora and semantic qrels through `ir_datasets`
rather than `assets.zilliz.com`. This Go driver downloads and prepares the same
published source archives directly: `collectionandqueries.tar.gz` for MS MARCO
and the BEIR `hotpotqa.zip` archive for HotpotQA. Downloads are verified against
the dataset publishers' MD5 checksums; Python and `ir_datasets` are not needed.

Run the corpus against either backend:

```bash
./vector-db-bench xvec \
  --path ./fts-msmarco-xvec \
  --case-type FTSBm25Performance \
  --dataset-with-size-type "MS MARCO Small (100K documents)" \
  --dataset-dir ./dataset/msmarco_small_100k \
  --payload-profile ids_only \
  --num-concurrency 40,80 \
  --output result-fts-msmarco-xvec.json
```

`--payload-profile text` includes the indexed text field in each result so its
materialization cost is measured. Analyzer settings are controlled by
`--fts-tokenizer`, `--fts-token-filters`, `--fts-extra-params`, and
`--fts-default-operator`. The default is standard tokenization plus lowercase,
with adjacent query terms combined by OR.

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

## IVF comparison

Select IVF explicitly and keep the centroid training and probe parameters the
same for both backends. The defaults shown below match xvec and zvec-go:

```powershell
./vector-db-bench.exe xvec `
  --path ./.bench/collections/cohere-1m-xvec-ivf `
  --case-type Performance768D1M `
  --index-type ivf `
  --ivf-n-list 1024 `
  --ivf-n-iterations 10 `
  --ivf-n-probe 10 `
  --ivf-scale-factor 10 `
  --optimize-concurrency 8 `
  --num-concurrency 8 `
  --output ./.bench/xvec-ivf.json
```

Run zvec with the same arguments, changing only the backend, collection path,
and output path. `--ivf-use-soar` enables SOAR list assignment, while
`--is-using-refiner` enables query-time refinement using
`--ivf-scale-factor` candidates. Both are disabled by default.

## Vamana comparison

zvec-go v0.7 exposes zvec's native Vamana index. Select it independently from
DiskANN with `--index-type vamana`. The benchmark pins the shared native
defaults for both backends: maximum degree 64, construction list size 100,
alpha 1.2, graph saturation disabled, two-pass construction disabled, and
search EF 200. zvec-go does not yet expose the Vamana query-parameter wrapper,
so non-default EF or refiner settings are rejected for zvec instead of silently
running different workloads.

```powershell
./vector-db-bench.exe zvec `
  --path ./.bench/collections/openai-50k-zvec-vamana `
  --case-type Performance1536D50K `
  --index-type vamana `
  --ef-search 200 `
  --num-concurrency 8 `
  --output ./.bench/zvec-vamana.json
```

Run xvec after zvec with the same arguments, changing only the backend,
collection path, and output path.

## Dataset compatibility

The built-in cases use the VectorDBBench schema:

| File | Required columns |
| --- | --- |
| `shuffle_train*.parquet` | `id` (`INT64`), `emb` (`LIST<FLOAT>`) |
| `test.parquet` | `id` (`INT64`), `emb` (`LIST<FLOAT>`) |
| `neighbors.parquet` | `id` (`INT64`), `neighbors_id` (`LIST<INT64>`) |

For offline or preprocessed runs, `--skip-download` accepts either the original
archive in `--dataset-dir` or three exported files: `documents.jsonl` (`id`,
`text`, `filter_id`), `queries.jsonl` (`id`, `text`), and `qrels.jsonl`
(`query_id`, `doc_id`, `relevance`). Positive graded qrels are retained exactly
for Recall, MRR, and NDCG calculations.

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
