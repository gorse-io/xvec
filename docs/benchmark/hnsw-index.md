---
title: HNSW Index
---

# HNSW Index Benchmark

This benchmark compares xvec with zvec-go v0.7.0 using the same
VectorDBBench-compatible `Performance768D1M` workload. The xvec run used commit
`7efa347`.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | `Performance768D1M` / Cohere 1,000,000 × 768 |
| Metric and result size | Cosine, TopK = 100 |
| Index | HNSW, M = 15, efConstruction = 500, efSearch = 180 |
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
| Insert time | **44.862 s** | 83.229 s | 0.54x |
| Optimize time | **1,765.111 s** | 1,866.387 s | 0.95x |
| Total load time | **1,810.004 s** | 1,950.396 s | 0.93x |
| Insert throughput | **22,290.46 rows/s** | 12,014.97 rows/s | 1.86x |

- Index query

| Search metric | xvec | zvec-go | xvec / zvec-go |
| --- | ---: | ---: | ---: |
| Recall@100 | 0.94331 | **0.94340** | 1.00x |
| Serial QPS | 169.25 | **458.84** | 0.37x |
| Serial average latency | 5.675 ms | **2.029 ms** | 2.80x |
| Serial p95 latency | 8.594 ms | **3.527 ms** | 2.44x |
| Serial p99 latency | 10.199 ms | **4.525 ms** | 2.25x |
| 8-concurrency QPS | 378.35 | **1,207.17** | 0.31x |
| 8-concurrency average latency | 21.121 ms | **6.624 ms** | 3.19x |
| 8-concurrency p95 latency | 33.252 ms | **10.033 ms** | 3.31x |
| 8-concurrency p99 latency | 49.541 ms | **11.754 ms** | 4.22x |
