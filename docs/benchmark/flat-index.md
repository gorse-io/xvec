---
title: Flat Index
---

# Flat Index Benchmark

This benchmark compares xvec with zvec-go v0.7.0 using the same
VectorDBBench-compatible `Performance1536D50K` workload. The xvec run used commit
`46bb378`, which includes the Flat index optimizations described by the commit history.

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
| Insert time | **3.525 s** | 4.525 s | 0.78x |
| Optimize time | 3.256 s | **2.533 s** | 1.29x |
| Total load time | **6.805 s** | 7.127 s | 0.95x |
| Insert throughput | **14,184.60 rows/s** | 11,048.60 rows/s | 1.28x |

- Index query

| Search metric | xvec | zvec-go | xvec / zvec-go |
| --- | ---: | ---: | ---: |
| Recall@100 | **1.00000** | **1.00000** | 1.00x |
| Serial QPS | **50.29** | 41.98 | 1.20x |
| Serial average latency | **19.703 ms** | 23.787 ms | 0.83x |
| Serial p95 latency | **24.255 ms** | 30.123 ms | 0.81x |
| Serial p99 latency | **29.860 ms** | 37.953 ms | 0.79x |
| 8-concurrency QPS | 77.09 | **87.58** | 0.88x |
| 8-concurrency average latency | 103.612 ms | **91.049 ms** | 1.14x |
| 8-concurrency p95 latency | 137.655 ms | **119.438 ms** | 1.15x |
| 8-concurrency p99 latency | 171.419 ms | **147.548 ms** | 1.16x |
