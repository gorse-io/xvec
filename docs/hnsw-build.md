# Dense HNSW construction

This independently gated v0.3 unit adds deterministic dense HNSW graph
construction to `internal/core`. Graph search, persistence/reopen, incremental
writes, and collection orchestration are now implemented by subsequent units.

## Build contract

`HNSWBuilder` accepts unique keys and finite FP32 vectors of one fixed
dimension. It owns cloned inputs and is one-shot only after a successful
`Build`; cancellation or another construction failure leaves it retryable. An
empty input produces a valid graph with no entry point and maximum level -1.

The options align with the pinned public baseline:

| Option | Default | Meaning |
| --- | ---: | --- |
| `M` | 50 | maximum neighbors on levels above zero |
| `EFConstruction` | 500 | bounded candidate set while inserting a node |
| `Seed` | 0 | deterministic level generator seed |

Level zero permits `2*M` neighbors, matching the baseline's default L0
multiplier. `M` is bounded so that this doubled degree remains below the native
16-bit neighbor-count ceiling, and `EFConstruction` must be at least `M`.

## Deterministic graph algorithm

Levels use the standard exponentially decreasing HNSW distribution, a fixed
SplitMix64 stream, and a maximum of fifteen layers (levels 0 through 14). For a
fixed seed and insertion order, level assignment and every adjacency list are
bit-for-bit stable across runs and platforms.

Nodes are inserted in input order. Construction greedily descends layers above
the new node, performs a bounded best-first `EFConstruction` search on each
shared layer, and selects at most the layer degree limit. Selection applies the
HNSW diversity rule: a candidate is retained only when it is strictly closer
to the new node than to every already selected neighbor. All comparisons are
metric-aware—higher is better for inner product, lower is better for L2,
cosine distance, and MIPS-L2—with lower node positions breaking equal scores.

Every selected edge is installed in both directions. When a reverse adjacency
is full, its old neighbors plus the new candidate are rescored from that node
and diversity-pruned back to the degree limit. The newest node becomes the
entry point only when it introduces a strictly higher level.

## Inspection boundary

`HNSWIndex` retains originals contiguously and currently implements
`DenseProvider`. `Vector`, `Level`, `EntryPoint`, `MaxLevel`, and `Neighbors`
allow tests and later codecs to inspect the graph without exposing mutable
storage. The separately documented search unit traverses this topology.

Tests cover all dense metrics, defaults and invalid parameters, empty and
single-node graphs, entry/level/degree/reference invariants, duplicate and
non-finite input rejection, cancellation retry, builder lifecycle, cloned
ownership, deterministic topology, bounded level sampling, and a construction
benchmark.
