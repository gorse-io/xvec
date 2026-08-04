# Dense HNSW incremental writes

This independently gated v0.3 unit turns a built or reopened dense HNSW graph
into a `DenseIndex` by implementing `DenseStreamer.Add`. Each call accepts one
unique key and one finite FP32 vector of the graph's fixed dimension. The graph
owns the input; later caller mutation cannot change stored vectors.

## Deterministic insertion

The level generator resumes from the state stored by the builder or native Go
artifact. For the same seed and key/vector insertion sequence, building a
prefix and streaming the remainder produces the same levels, entry point,
neighbor order, and search results as one one-shot build.

An inserted node performs the same greedy upper-level descent, bounded
construction search, diversity selection, reverse-edge maintenance, and
per-level degree pruning as the builder. Duplicate keys, invalid vectors,
capacity violations, and cancellation return errors without consuming a
sampled level.

## Atomicity and concurrency

The current correctness-first implementation serializes additions, clones the
current graph under a read lock, and plans each insertion on that private
copy-on-write generation after releasing the graph lock. Only a completely
linked and validated operation replaces the live generation under a short
write lock. A cancellation during cloning or graph traversal therefore cannot
expose a key without its vector, level, or reverse edges. This deliberately
favors simple atomicity over write throughput; later performance work may
replace the full-generation copy with a smaller mutation journal without
changing the contract.

Providers, topology inspection, and searches hold a read lock, so concurrent
readers see either the old or new complete graph. Save clones a complete
generation under the read lock and releases it before encoding, syncing, and
atomic publication. Reopened graphs retain their level-generator state and can
continue accepting deterministic additions.

Tests cover empty-graph bootstrap, input ownership, one-shot topology identity,
continued streaming after reopen, pre- and mid-operation cancellation,
duplicate and malformed inputs, failed-operation byte identity, and concurrent
add/search/provider/save/reopen under the race detector.
