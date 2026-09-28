import csv,json,pathlib
repo=pathlib.Path('/home/zhenghaoz/xvec');root=pathlib.Path(__file__).resolve().parent;archive=repo/'docs/benchmark-runs/hnsw-label-50-opt-20260928'
rows=list(csv.DictReader((repo/'docs/benchmark-hnsw-label-filter.csv').open()))
old={(r['backend'],r['label_percentage']):r for r in csv.DictReader((archive/'previous.csv').open())}
by={(r['backend'],r['label_percentage']):r for r in rows}
x=by['xvec','0.5'];z=by['zvec','0.5'];b=old['xvec','0.5'];meta=json.loads((archive/'metadata.json').read_text())
f=lambda r,k:float(r[k])
table=['| Matching labels | xvec previous QPS | xvec current QPS | Speedup | zvec rerun QPS | xvec recall (%) | zvec recall (%) |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for a,c in zip(rows[::2],rows[1::2]):
 before=old['xvec',a['label_percentage']]
 table.append(f"| {100*f(a,'label_percentage'):g}% | {f(before,'concurrent_qps'):.2f} | {f(a,'concurrent_qps'):.2f} | {f(a,'concurrent_qps')/f(before,'concurrent_qps'):.2f}× | {f(c,'concurrent_qps'):.2f} | {f(a,'recall_at_k_pct'):.3f} | {f(c,'recall_at_k_pct'):.3f} |")
report=f'''# HNSW label filtering on Cohere 100K

This comparison runs `LabelFilterPerformanceCase` with FP16 scalar quantization
on xvec and zvec. The nine percentages describe documents that **match** the
filter. Each label uses its published filtered ground truth; the unfiltered
`neighbors.parquet` is not used.

## Results after the 50%-focused optimization

At 50% matching labels, xvec improves from **{f(b,'concurrent_qps'):.2f} to
{f(x,'concurrent_qps'):.2f} QPS ({f(x,'concurrent_qps')/f(b,'concurrent_qps'):.2f}×)**. Concurrent P99 falls from
{f(b,'concurrent_latency_p99_ms'):.3f} to {f(x,'concurrent_latency_p99_ms'):.3f} ms.
zvec reaches {f(z,'concurrent_qps'):.2f} QPS in this rerun, leaving a
**{f(z,'concurrent_qps')/f(x,'concurrent_qps'):.2f}×** throughput gap instead of the previous 2.59×.
xvec's recall remains {f(x,'recall_at_k_pct'):.3f}%, compared with zvec's
{f(z,'recall_at_k_pct'):.3f}%. These are fixed-EF results, not matched-recall comparisons.

Both backends were rerun on the original persisted collections. QPS below is
from the 8-worker concurrent phase; recall uses all 1,000 serial queries.
“Previous” refers to the first filter optimization, not the original
unoptimized implementation (which measured 33.31 QPS at 50%).

'''+ '\n'.join(table)+'\n\n'
assert all(abs(f(r,'recall_at_k_pct')-f(old['xvec',r['label_percentage']],'recall_at_k_pct'))<1e-6 for r in rows if r['backend']=='xvec')
report+='All nine xvec recall values are unchanged.\n\n'
a10=by['xvec','0.1'];b10=old['xvec','0.1'];z10=by['zvec','0.1'];pz10=old['zvec','0.1']
report+=f"Throughput increased at every matching percentage in this round, but latency\ndid not improve uniformly. At 10%, xvec serial average latency rose from\n{f(b10,'serial_latency_avg_ms'):.3f} to {f(a10,'serial_latency_avg_ms'):.3f} ms, and concurrent P99\nchanged from {f(b10,'concurrent_latency_p99_ms'):.3f} to {f(a10,'concurrent_latency_p99_ms'):.3f} ms. zvec also varied\nbetween rounds: its 10% QPS fell from {f(pz10,'concurrent_qps'):.2f} to {f(z10,'concurrent_qps'):.2f}.\nThese measurements do not establish uniform latency gains or eliminate run-to-run\nvariation.\n\n"

report+='| Matching labels | xvec P99 (ms) | zvec P99 (ms) |\n| --- | ---: | ---: |\n'
for a,c in zip(rows[::2],rows[1::2]):report+=f"| {100*f(a,'label_percentage'):g}% | {f(a,'concurrent_latency_p99_ms'):.3f} | {f(c,'concurrent_latency_p99_ms'):.3f} |\n"
mem={backend:[f(r,'query_peak_rss_mib') for r in rows if r['backend']==backend] for backend in ['xvec','zvec']}
report+=f'''
Across the nine query processes, xvec peak RSS is
{min(mem['xvec']):.2f}–{max(mem['xvec']):.2f} MiB, compared with zvec's
{min(mem['zvec']):.2f}–{max(mem['zvec']):.2f} MiB. A substantial memory gap remains.
Peak RSS includes loading queries and ground truth and opening the index;
it is not steady-state index memory.

## Why the previous 50% result was slow

A separate 15-second CPU profile of the preceding implementation found
49.09% of samples in `evaluateSnapshotFilters` and its callees, with the global
ordinal lookup and match-map construction dominating that work. The collection
has one persisted 100K-document segment **and an empty writable segment**.
The shortcut required exactly one segment, so it failed to activate despite
there being only one segment containing documents. At 50%, that caused another
50K-element mapping pass for every query.

The same profile attributed 11.30% of CPU samples to the quantized-neighbor
prefetch helper and its callees. That helper performed synchronous byte reads
to warm cache lines, including neighbors already visited. These observations
identify xvec costs; they do not apportion the entire cross-backend gap.

## Implementation changes

The first round followed zvec's `DocFilter::get_bf_by_keys_and_update` and
`vector_recall_node.cc`: reuse snapshot visibility, enumerate inverted candidates
directly, retain exact bitmaps, and pass selected keys to quantized linear search.
This round retains those changes and adds:

- [Snapshot filtering](../collection_filter.go) recognizes one nonempty live
  segment even when empty writable segments are present. Its local predicate
  and ordinals are already global, avoiding the redundant mapping pass.
  Multiple segments and deleted-document visibility keep the general path.
- [Filtered HNSW traversal](../internal/core/algorithm/hnsw_quantized_searcher.go)
  gathers unvisited neighbors before prefetching them, following
  `dual_heap_search_neighbors` in zvec's
  `src/core/algorithm/hnsw/hnsw_algorithm.cc`.
- [AMD64 prefetch](../internal/core/algorithm/hnsw_prefetch_amd64.go) issues
  bounded `PREFETCHT0` hints like zvec's `ailego_prefetch`, replacing synchronous
  cache-line reads for quantized vectors. Other architectures and `noasm`
  retain the portable fallback. Explicit prefetch limits are respected.

The final change preserves precision, distance scoring, radius/refinement
behavior, EF, graph topology, the file format, and GC settings. The measurements
cover the combined changes, not controlled per-change speedup estimates.
Diagnostic profiled runs are separate and excluded from the CSV.

## Method

- Dataset: `Small Cohere (768dim, 100K)`, cosine distance, 100,000 training
  vectors and all 1,000 test queries. Label counts and every ground-truth
  neighbor's label were validated against `scalar_labels.parquet`.
- HNSW: M=50, EFConstruction=500, EFSearch=300, K=100; FP16 scalar quantization,
  rotation disabled, refinement disabled, mmap enabled, ID-only results.
- Machine: e2-standard-8, AMD EPYC 7B12, CPU affinity 0–7; Go 1.27.1,
  CGO_ENABLED=0, GOMAXPROCS=8, GOMEMLIMIT=24GiB. Both backends use the same
  Go benchmark binary, 8 optimize workers, and 8 query workers.
- The baseline built one collection per backend with an inverted index on
  `labels`. Subsequent rounds reuse those exact collections and reopen them
  in a fresh process per measurement. This round runs the 50% pair first,
  then the other eight rates in the original alternating backend order.
  Filesystem caches are not flushed.
- Each query run warms up with 100 queries, measures concurrent search for
  30 seconds, waits 3 seconds, then measures all 1,000 serial queries and recall.
  The seed is 0. Each condition is measured once per round, without a profiler.
- Process resources are measured using `wait4`. Download and compilation are
  excluded. Build metrics remain the original build measurements, with their
  original `build_backend_version`, separately from the query `backend_version`.

The xvec base is `bc87d5a7db1fde4251574fe458b520485cb2c92f`. The measured source
adds the archived production patch with SHA-256
`{meta['source_patch_sha256']}`.
The query version appends `+filter50.{meta['source_patch_sha256'][:12]}` to the base.
zvec uses the unmodified zvec-go v0.7.0 binding and official v0.7.0 Linux x64
native release. Native FP16 quantization and disabled rotation were checked
through C API getters; release and dataset checksums are archived.

Both default planners have a brute-force-by-keys ratio of 0.1. Matching at most
10% selects linear search over filtered candidates; 20% and 50% use filtered
HNSW traversal. These results include the default selective-filter fallback
and do not force graph traversal at every percentage.

Concurrent QPS is the primary throughput comparison. Serial QPS includes the
harness's recall calculation in elapsed time; serial latency times only the
search call. Recall uses the label's published top 100 neighbors. Results are
single measurements without confidence intervals, so small differences should
not be interpreted as stable advantages.

The following **original build measurements** are retained for context.
No index was rebuilt for either optimization rerun; these do not measure build
time or build memory of the optimized implementation.

| Backend | Insert (s) | Optimize (s) | Total load (s) | Build peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| xvec | 10.28 | 143.91 | 154.19 | 1466.30 |
| zvec | 7.55 | 95.73 | 103.33 | 789.82 |

## Validation and artifacts

`go test ./...` passed on the measured source. Quantized HNSW tests also pass
with `-tags=noasm`. The new regression cases cover empty segments before and
after the data segment, and identical FP16 filtered-HNSW results with prefetch
disabled, automatic, or maximally requested, including radius searches.
The prior filter/visibility/SQL-semantic and candidate-key tests remain in place.
All eighteen reports, CSV fields, preserved build measurements and source,
binary, native-library and dataset hashes are checked together.

- [Current CSV](benchmark-hnsw-label-filter.csv): all 18 query measurements.
- [Current raw reports, profiles, patch and commands](benchmark-runs/hnsw-label-50-opt-20260928/).
- [Previous comparison](benchmark-runs/hnsw-label-50-opt-20260928/previous.md)
  and [CSV](benchmark-runs/hnsw-label-50-opt-20260928/previous.csv).
- [First optimization reports](benchmark-runs/hnsw-label-opt-20260928/).
- [Original builds and baseline reports](benchmark-runs/hnsw-label-fp16-20260928/).
'''
(repo/'docs/benchmark-hnsw-label-filter.md').write_text(report)
print('Updated benchmark report.')
