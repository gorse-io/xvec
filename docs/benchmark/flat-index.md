---
title: Flat Index
---

# Flat Index Benchmark

This benchmark compares xvec with zvec-go v0.7.0 using the same
VectorDBBench-compatible `Performance1536D50K` workload. The xvec run used the local
working tree based on commit `ab00d4c`.

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
| Insert time | 4.009 s | 4.525 s | 0.89x |
| Optimize time | 6.132 s | 2.533 s | 2.42x |
| Total load time | 10.158 s | 7.127 s | 1.43x |
| Insert throughput | 12,473.48 rows/s | 11,048.60 rows/s | 1.13x |

- Index query

| Search metric | xvec | zvec-go | xvec / zvec-go |
| --- | ---: | ---: | ---: |
| Recall@100 | 1.00000 | 1.00000 | 1.00x |
| Serial QPS | 38.32 | 41.98 | 0.91x |
| Serial average latency | 25.910 ms | 23.787 ms | 1.09x |
| Serial p95 latency | 33.589 ms | 30.123 ms | 1.12x |
| Serial p99 latency | 61.492 ms | 37.953 ms | 1.62x |
| 8-concurrency QPS | 84.96 | 87.58 | 0.97x |
| 8-concurrency average latency | 93.992 ms | 91.049 ms | 1.03x |
| 8-concurrency p95 latency | 128.550 ms | 119.438 ms | 1.08x |
| 8-concurrency p99 latency | 157.195 ms | 147.548 ms | 1.07x |
