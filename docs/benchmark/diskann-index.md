---
title: DiskANN Index
---

# DiskANN Index Benchmark

This benchmark measures xvec with the complete VectorDBBench-compatible
`Performance1024D10M` workload. The run used commit `abd53d7`.

zvec-go v0.7.0 does not provide a Windows DiskANN implementation. Its native
library returned `DiskAnn is not supported on this platform` during the smoke
test, so a comparable zvec-go result is not available on this benchmark host.

The standard xvec collection path retains the writable FP32 document segment
in memory. On this 32 GB host it reached 10.35 GB after 1.1 million rows and
could not safely buffer a single 10-million-row segment. The completed run
therefore used xvec's core DiskANN API to build the ten official one-million-row
training shards as ten independent indexes. Queries search the ten segments in
parallel and merge their results into one global TopK, matching xvec's
multi-segment query semantics. All 10 million vectors remain part of the test.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | `Performance1024D10M` / BioASQ 10,000,000 × 1,024 |
| Metric and result size | Cosine, TopK = 100 |
| Index | DiskANN, max degree = 100, build list = 50, query list = 300 |
| Product quantization chunks | Automatic (`0`) |
| Segment layout | 10 × 1,000,000 rows, global TopK merge |
| Memory mapping | Enabled |
| Build and query workers | 8 |
| Node cache | 64 MiB per segment |
| Warmup | 100 queries |
| Concurrent search | 30 seconds at concurrency 8 |
| Serial search | All 3,106 queries after a 3-second cooldown |
| Result payload | IDs only |

## Results

- Index build

| Build metric | xvec |
| --- | ---: |
| Dataset read and builder add time | **525.647 s** |
| DiskANN build and persistence time | **13,110.577 s** |
| Total load time | **13,672.903 s (3.798 h)** |
| Add throughput | **19,024.16 rows/s** |
| Persisted DiskANN size | **81.147 GiB** |

- Index query

| Search metric | xvec |
| --- | ---: |
| Recall@100 | **0.97230** |
| Serial queries | **3,106** |
| Serial QPS | **1.26** |
| Serial average latency | **792.819 ms** |
| Serial p95 latency | **1,546.572 ms** |
| Serial p99 latency | **1,720.174 ms** |
| 8-concurrency completed queries | **16** |
| 8-concurrency QPS | **0.53** |
| 8-concurrency average latency | **11,028.890 ms** |
| 8-concurrency p95 latency | **11,792.652 ms** |
| 8-concurrency p99 latency | **11,938.751 ms** |
