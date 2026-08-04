# Runtime configuration and resource governance

`RuntimeConfig` controls process-wide query and maintenance admission. Install
it before the first `CreateAndOpen`, `Open`, or `CurrentRuntimeStats` call:

```go
config := zvec.NewRuntimeConfig()
config.MemoryLimitBytes = 512 << 20
config.QueryConcurrency = 8
config.OptimizeConcurrency = 2
config.LogLevel = zvec.LogLevelInfo
config.Logger = slog.Default()

if err := zvec.ConfigureRuntime(config); err != nil {
    return err
}
```

The first valid call wins for the lifetime of the process. Later calls are
successful no-ops, matching the pinned baseline lifecycle. Invalid calls fail
without freezing configuration. Creating or opening a collection and reading
`CurrentRuntimeStats` install defaults if configuration has not already been
set. Each collection handle retains the process configuration active when it
was opened.

`NewRuntimeConfig` is the recommended starting point. It selects the current
`GOMAXPROCS`, capped by `MaxRuntimeConcurrency`, for both concurrency limits;
warning-level logging; the pinned planner ratios; and no explicit memory
admission limit. `Validate` is safe to call without installing the value.

## Concurrency and cancellation

`QueryConcurrency` bounds simultaneous `Query`, `GroupByQuery`, and
`MultiQuery` operations. `OptimizeConcurrency` independently bounds
`CreateIndex`, `DropIndex`, `AddColumn`, `AlterColumn`, `DropColumn`, and
`Optimize`. Waiting operations observe their context; cancellation returns the
context error without starting the task. Both values must be between 1 and
`MaxRuntimeConcurrency`.

The configured values also bound ANN build/search workers and DDL rewrite
workers. An explicit per-operation worker request may lower, but cannot exceed,
the maintenance limit.

`QueryThreadBinding` and `OptimizeThreadBinding` are public for baseline
configuration parity, but `true` returns `ErrNotSupported`: portable Go does
not bind individual goroutines to scheduler worker threads.

## Memory admission

`MemoryLimitBytes` is a conservative scratch-admission budget shared by query
and maintenance tasks. A non-zero value must be at least
`MinRuntimeMemoryLimit` (100 MiB). Tasks reserve an estimate derived from the
collection's retained encoded data and their operation class. Reservations
wait context-sensitively when concurrent work fills the budget; a single
estimate larger than the limit returns `ErrResourceExhausted` immediately.

This is deliberately not presented as an exact Go heap cap. The Go runtime,
application allocations, filesystem mappings, and implementation overhead can
make process memory differ from the estimate. A zero value disables admission
limiting and leaves heap sizing to Go. Use the runtime's own memory controls as
needed when a hard process envelope is required.

`CollectionStats.StorageMemoryBytes` reports the conservative retained encoded
size used as the estimate's base. It includes mutable and immutable segment
records, keys, payloads, headers, and deletion snapshots. It is not RSS or
`runtime.MemStats.HeapAlloc`.

## Planner ratios and DiskANN cache

The default planner thresholds match the pinned baseline:

| Field | Default | Planner effect |
| --- | ---: | --- |
| `InvertToForwardScanRatio` | 0.90 | An inverted candidate set covering at least this fraction switches to a forward scan. |
| `BruteForceByKeysRatio` | 0.10 | A vector query whose scalar filter matches at most this fraction uses an exact candidate scan. |
| `FTSBruteForceByKeysRatio` | 0.05 | An FTS branch whose scalar filter matches at most this fraction seeks only those candidate ordinals. |

All thresholds must be finite and in `[0, 1]`. The alternatives are exact
execution routes: changing a threshold can change cost, not matching or score
semantics.

`CollectionOptions.MaxBufferSize` is per handle and separate from the process
budget. It determines how many 4 KiB DiskANN node sectors may be retained in
that query's cache. Zero selects `DefaultMaxBufferSize` (64 MiB); a value below
one sector disables node caching.

## Logging

Set `Logger` to any `*slog.Logger`. Query lifecycle records use debug severity,
maintenance lifecycle records use info, and memory-budget rejections use warn.
`LogLevel` filters zvec records before they reach the handler. A nil logger
disables zvec logging. Zvec neither owns nor closes the logger or its handler,
so file output and rotation remain application policy. `LogLevelFatal` maps to
the `slog` error level and never terminates the process.

## Jieba fallback

`JiebaDictionaryDir` installs the lowest-priority process-wide Jieba resource
directory with the runtime configuration. A field's explicit tokenizer
directory takes priority, followed by `ZVEC_JIEBA_DICT_DIR`, then this fallback.
`SetDefaultJiebaDictDir` and `DefaultJiebaDictDir` expose the same fallback
directly.

## Statistics

`CurrentRuntimeStats` returns concurrency-safe point-in-time counters:

- configured memory limit, current and peak reservations, and waiters;
- active, peak, queued, and completed query tasks;
- active, peak, queued, and completed maintenance tasks.

`Collection.Stats` additionally returns live document count, immutable segment
count, mutable document count, retained deletion count, the encoded-memory
estimate, and per-field index completeness. Counters are observational and may
change immediately under concurrent work.
