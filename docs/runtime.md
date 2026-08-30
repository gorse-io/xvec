# Runtime configuration

`RuntimeConfig` controls process-wide query and maintenance admission. Install
it before the first `CreateAndOpen`, `Open`, or `CurrentRuntimeStats` call:

```go
config := xvec.NewRuntimeConfig()
config.MemoryLimitBytes = 512 << 20
config.QueryConcurrency = 8
config.OptimizeConcurrency = 2
config.LogLevel = xvec.LogLevelInfo
config.Logger = slog.Default()

if err := xvec.ConfigureRuntime(config); err != nil {
    return err
}
```

The first valid call wins for the process lifetime. Later calls are successful
no-ops. Invalid calls fail without freezing configuration. Opening a collection
or reading `CurrentRuntimeStats` installs defaults if configuration has not
already been set. Each collection retains the process configuration active when
it was opened.

`NewRuntimeConfig` uses the current `GOMAXPROCS`, capped by
`MaxRuntimeConcurrency`, for query and maintenance limits; warning-level
logging; the default planner ratios; and no explicit memory admission limit.
`Validate` checks a value without installing it.

## Concurrency and cancellation

`QueryConcurrency` bounds simultaneous `Query`, `GroupByQuery`, and
`MultiQuery` operations. `OptimizeConcurrency` independently bounds DDL and
`Optimize`. Waiting operations observe their context and do not start after
cancellation. Explicit per-operation concurrency may lower but cannot exceed
the maintenance limit.

Thread-binding fields exist for baseline configuration parity, but setting them
to true returns `ErrNotSupported`: portable Go does not bind goroutines to
scheduler worker threads.

## Memory admission

`MemoryLimitBytes` is a conservative scratch budget shared by query and
maintenance tasks. A non-zero value must be at least `MinRuntimeMemoryLimit`
(100 MiB). Work reserves an estimate based on retained encoded collection data
and operation class. Reservations wait context-sensitively; a single estimate
larger than the limit returns `ErrResourceExhausted`.

This is not a Go heap or RSS cap. Application allocations, mappings, runtime
overhead, and filesystem caches can make process memory differ from the
estimate. Zero disables admission limiting. `CollectionStats.StorageMemoryBytes`
is the encoded-size estimate used as the base, not heap usage.

## Planner ratios and DiskANN cache

| Field | Default | Effect |
| --- | ---: | --- |
| `InvertToForwardScanRatio` | 0.90 | Switch an almost-full inverted candidate set to a forward scan. |
| `BruteForceByKeysRatio` | 0.10 | Use an exact vector candidate scan for a sufficiently selective scalar filter. |
| `FTSBruteForceByKeysRatio` | 0.05 | Seek only selected FTS candidate ordinals for a sufficiently selective filter. |

Ratios must be finite and in `[0, 1]`. The alternative routes are exact, so the
thresholds affect cost rather than matching or score semantics.

`CollectionOptions.MaxBufferSize` separately controls each handle's DiskANN
node cache. Zero selects `DefaultMaxBufferSize` (64 MiB); less than one 4 KiB
sector disables node caching.

## Logging and Jieba

Set `Logger` to a `*slog.Logger`. Query lifecycle records use debug,
maintenance lifecycle records use info, and memory rejection uses warn.
`LogLevel` filters xvec records before the handler. A nil logger disables xvec
logging. xvec neither owns nor closes the logger. `LogLevelFatal` maps to slog
error and never terminates the process.

Jieba resources are resolved in this order:

1. field tokenizer configuration;
2. `ZVEC_JIEBA_DICT_DIR`;
3. `RuntimeConfig.JiebaDictionaryDir` or `SetDefaultJiebaDictDir`.

## Statistics

`CurrentRuntimeStats` reports configured/current/peak memory reservations and
waiters, plus active/peak/queued/completed query and maintenance tasks.
`Collection.Stats` reports live documents, immutable segments, mutable and
deleted documents, the encoded-memory estimate, and per-field index
completeness. All counters are observational point-in-time values.
