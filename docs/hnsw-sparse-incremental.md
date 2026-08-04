# Sparse HNSW incremental writes

This independently gated v0.3 unit makes built and reopened sparse HNSW
indexes implement the common `SparseStreamer` contract. Collection routing is
still a separate integration unit, so public collection requests do not yet
select sparse HNSW.

## Input and deterministic topology

`SparseHNSWIndex.AddSparse` accepts a unique key and a canonical sparse FP32
vector. Coordinate and value counts must match, coordinates must be strictly
increasing, and values must be finite. Empty vectors are valid. Accepted input
is copied into the index's CSR storage.

Each successful addition advances the persisted SplitMix64 level-generator
state exactly once and uses the same insertion, diversity pruning, reverse-edge
maintenance, and stable tie rules as one-shot construction. Streaming the same
ordered inputs therefore produces the same CSR data and graph topology as a
single build, including when additions continue after save and reopen. Failed
or canceled additions do not consume a sampled level.

## Atomic visibility and snapshots

Writers are serialized per index. An addition clones the current CSR vectors,
key map, levels, and adjacency lists; constructs the new node on that private
generation; checks cancellation; and then swaps all mutable fields under one
short write lock. Searchers and providers hold a read lock and can observe only
the old or new complete generation. A partially linked node or partial CSR row
is never published.

Save clones the currently published generation under the same read lock and
encodes the clone after releasing it. Concurrent additions can continue while
the snapshot is validated, checksummed, and atomically published. This
correctness-first implementation copies the graph per addition; later
collection integration may batch additions without changing the streaming
contract.

Tests cover deterministic equality with one-shot construction, continued
streaming after reopen, input ownership, empty-index insertion, validation and
duplicate errors, cancellation during graph cloning and traversal, rollback
identity, concurrent writers/searchers/snapshots under the race detector, and
final persistence identity.
