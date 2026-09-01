---
title: Flat Index
---

# Flat Index Benchmark

This benchmark compares xvec with zvec-go v0.7.0 using the same
VectorDBBench-compatible `Performance1536D50K` workload. The xvec run used commit
`f01518f`, which includes the Flat index optimizations described by the commit history.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | `Performance1536D50K` / OpenAI 50,000 × 1,536 |
| Metric and result size | Cosine, TopK = 100 |
| Index | Flat, exact FP32 search |
| Insert batch size | 100 |
| Quantization / refiner | None / disabled |
| Memory mapping | Enabled |
| Optimize and query threads | 8 |
| Warmup | 100 queries |
| Concurrent search | 30 seconds at concurrency 8 |
| Serial search | 1,000 queries after a 3-second cooldown |
| Result payload | IDs only |

## Results

- Index build

| Build metric | xvec | zvec-go | xvec / zvec-go |
| --- | ---: | ---: | ---: |
| Insert time | 3.677 s | 4.525 s | 0.81x |
| Optimize time | 4.638 s | 2.533 s | 1.83x |
| Total load time | 8.321 s | 7.127 s | 1.17x |
| Insert throughput | 13,598.20 rows/s | 11,048.60 rows/s | 1.23x |

- Index query

| Search metric | xvec | zvec-go | xvec / zvec-go |
| --- | ---: | ---: | ---: |
| Recall@100 | 1.00000 | 1.00000 | 1.00x |
| Serial QPS | 49.41 | 41.98 | 1.18x |
| Serial average latency | 20.079 ms | 23.787 ms | 0.84x |
| Serial p95 latency | 25.752 ms | 30.123 ms | 0.85x |
| Serial p99 latency | 31.755 ms | 37.953 ms | 0.84x |
| 8-concurrency QPS | 77.32 | 87.58 | 0.88x |
| 8-concurrency average latency | 103.326 ms | 91.049 ms | 1.14x |
| 8-concurrency p95 latency | 137.676 ms | 119.438 ms | 1.15x |
| 8-concurrency p99 latency | 166.717 ms | 147.548 ms | 1.13x |
