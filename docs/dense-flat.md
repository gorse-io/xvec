# Dense Flat index

The first exact vector index is implemented in `internal/core` as a pure-Go
dense Flat scan. It stores fixed-dimension FP32 vectors contiguously, owns a
copy of every streamed vector, and requires unique unsigned document keys.

The index implements the shared dense `Provider`, `Searcher`, and `Streamer`
contracts. A one-shot `Builder` creates the same streamable index, so later
segment builders and approximate indexes can use a common lifecycle without
changing query orchestration.

Search evaluates every vector with the selected L2, inner-product, cosine, or
MIPS-L2 metric. Results are exact and deterministic: the metric selects the
best score direction and equal scores are ordered by ascending key. Search uses
O(k) ranking memory beyond the candidate views and checks cancellation between
candidates. Concurrent searches share the index; incremental adds are
serialized and never expose partially appended vectors.

The extended query contract applies metric-aware radius and candidate filters
during the scan. Group-by search independently retains a bounded top-k for
each resolved scalar group so a dominant group cannot hide other groups before
ranking. Collection-level deleted-document masks and projected document
materialization are layered over these exact results.
