# Sparse HNSW construction

This independently gated v0.3 unit adds deterministic sparse HNSW graph
construction to `internal/core`. Sparse graph search and persistence/reopen are
now implemented by subsequent units, as are incremental writes. Collection
routing is implemented by a later unit.

## Input and storage contract

`SparseHNSWBuilder` accepts unique keys and canonical sparse FP32 vectors.
Coordinate and value lengths must match, coordinates must be strictly
increasing, and values must be finite. Empty vectors are valid. The builder
owns cloned input in compressed sparse row form and transfers that storage to
one graph only after a successful build.

Sparse HNSW supports inner product exclusively, matching the public schema
constraint at the pinned C++ baseline. `DefaultSparseHNSWBuildOptions` uses the
same `M=50`, `EFConstruction=500`, level cap, and deterministic seed semantics
as dense HNSW.

## Graph construction

All levels are sampled before insertion with the shared SplitMix64 generator.
Each node then performs greedy upper-layer descent, bounded construction
search, diversity selection, and reverse-edge maintenance. Level zero permits
`2*M` outgoing edges; upper levels permit `M`. Full reverse lists are pruned
with the same deterministic heuristic and stable insertion-position tie break
used by dense HNSW.

`SparseHNSWIndex` exposes metric, length, build options, cloned vectors, entry
point, maximum/node levels, and cloned neighbor keys. Internal positions and
CSR backing slices are never returned.

Tests cover defaults and invalid metrics/options, canonical sparse validation,
empty and single-node graphs, provider ownership, deterministic topology,
entry/level/degree/reference/CSR invariants, builder lifecycle and
cancellation, and a construction benchmark.
