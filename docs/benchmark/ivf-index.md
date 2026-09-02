---
title: IVF Index
---

# IVF Index Benchmark

This benchmark compares xvec with zvec-go v0.7.0 using the same
VectorDBBench-compatible `Performance768D1M` workload. The xvec benchmark
binary was built from commit `7efa347`.

## Configuration

| Setting | Value |
| --- | --- |
| Dataset | `Performance768D1M` / Cohere 1,000,000 × 768 |
| Metric and result size | Cosine, TopK = 100 |
| Index | IVF, nlist = 1,024, training iterations = 10, nprobe = 10 |
| Scale factor / SOAR | 10 / disabled |
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
| Insert time | **47.667 s** | 75.011 s | 0.64x |
| Optimize time | **626.660 s** | 2,363.152 s | 0.27x |
| Total load time | **674.370 s** | 2,439.906 s | 0.28x |
| Insert throughput | **20,978.93 rows/s** | 13,331.32 rows/s | 1.57x |

- Index query

| Search metric | xvec | zvec-go | xvec / zvec-go |
| --- | ---: | ---: | ---: |
| Recall@100 | 0.76488 | **0.88889** | 0.86x |
| Serial QPS | **113.07** | 39.88 | 2.84x |
| Serial average latency | **8.542 ms** | 25.039 ms | 0.34x |
| Serial p95 latency | **10.883 ms** | 41.981 ms | 0.26x |
| Serial p99 latency | **12.201 ms** | 47.015 ms | 0.26x |
| 8-concurrency QPS | **247.58** | 71.43 | 3.47x |
| 8-concurrency average latency | **32.278 ms** | 111.498 ms | 0.29x |
| 8-concurrency p95 latency | **40.601 ms** | 185.018 ms | 0.22x |
| 8-concurrency p99 latency | **45.670 ms** | 212.870 ms | 0.21x |

With the same IVF parameters, xvec builds the index in 28% of zvec-go's total
load time and reaches 3.47× its 8-concurrency QPS. zvec-go retains the higher
Recall@100 by 0.12401, so the throughput gain comes with a recall tradeoff.
