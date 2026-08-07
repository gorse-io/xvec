# Continuous verification

The v1 hardening gates are checked into `.github/workflows` so release quality
does not depend on one developer machine.

## Per-change CI

`ci.yml` runs for every pull request and push to `main`:

- the complete test suite executes natively on current GitHub-hosted Linux,
  macOS, and Windows runners with the Go version declared by `go.mod`;
- Linux additionally runs `go vet ./...`, the complete suite with CGO disabled,
  and the complete suite under the race detector;
- persistence subprocess tests are repeated explicitly, covering atomic-file
  replacement, DiskANN publication, and Collection DDL/Optimize crash recovery.

Jobs have explicit timeouts, read-only repository permissions, fail-fast is
disabled for the platform matrix, and superseded per-change runs are canceled.

## Scheduled fuzzing

`fuzz.yml` runs weekly and on manual dispatch. Independent jobs fuzz the native
collection schema and document codecs, manifest, WAL, segment and snapshot
decoders, SQL filter parser, and FTS parser for 60 seconds each. The jobs are
bounded to three concurrent runners and retain the normal Go fuzz failure
artifacts in the workflow log/workspace for reproduction before the run ends.
Every fuzz target also remains part of ordinary `go test` through its seed
corpus.

## Benchmark records

`benchmarks.yml` runs weekly and on manual dispatch on Linux amd64. It executes
all checked-in benchmarks three times with allocation reporting and uploads the
raw Go benchmark record for 90 days. The suite records exact and ANN query
latency, build cost, tokenizer/filter throughput, FTS posting and merge cost,
runtime admission, hybrid reranking, memory allocations, and index I/O.

Benchmark records are evidence for release review rather than a wall-clock
pass/fail threshold: shared hosted runners do not provide stable enough timing
for a meaningful hard cutoff. Deterministic recall, result, resource-bound,
artifact-size, and reopen invariants remain blocking tests in the ordinary CI
suite. A material benchmark movement must be explained before a release is
tagged.

The workflows use official GitHub actions and read the toolchain version from
`go.mod`. Scheduled workflows deliberately use non-hour-boundary UTC times to
reduce scheduler congestion.
