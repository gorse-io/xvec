# IVF construction

This v0.3 unit adds deterministic, unquantized IVF construction to
`internal/core`. Search, persistence/reopen, and incremental writes are now
implemented by later independently gated units; collection orchestration
remains. The collection layer therefore continues returning `ErrNotSupported`
for IVF rather than substituting Flat.

## Build contract

`IVFBuilder` accepts unique keys and finite FP32 vectors of one fixed dimension.
It clones every input, rejects a duplicate before training, and is a one-shot
builder after a successful `Build`. A canceled or failed build publishes
nothing and may be retried. An empty builder creates a valid empty layout
without calling k-means, which lets schema/index construction succeed on an
empty collection.

The build options mirror the public defaults:

| Option | Default | Meaning |
| --- | ---: | --- |
| `NList` | 1024 | requested centroid/list count |
| `NIterations` | 10 | maximum Lloyd updates |
| `Tolerance` | float32 epsilon | absolute objective stopping threshold |
| `Workers` | `GOMAXPROCS` | bounded assignment concurrency when non-positive |
| `Seed` | 0 | deterministic centroid sampling seed |

The effective centroid count is `min(NList, vectorCount)`. All vectors train the
centroids in this unit. The shared k-means framework assigns every original to
its metric-best centroid with stable lower-index ties. Positions are appended
to each inverted list in builder insertion order, so layout does not depend on
worker scheduling. Duplicate samples may result in an empty list because equal
centroid scores use the lower index.

## Streamable output

`IVFIndex` stores original vectors once in contiguous FP32 memory. Inverted
lists contain positions into that storage rather than duplicate vectors. It
implements `DenseProvider`, enabling the original-vector refiner before IVF
search is connected. `Vector`, `List`, and `Centroids` return clones; option
values, list membership, training objective, and iteration count are available
for inspection without exposing mutable state.

The separately documented search unit probes the nearest centroids and scans
only their position lists. The persistence unit durably preserves this layout
and reconstructs its derived maps and lists on reopen. The incremental unit
adds concurrency-safe online list assignment without retraining a full index.

Tests cover separated-cluster assignment, stable list order, input/accessor
ownership, empty indexes, list-count capping, deterministic output across
worker counts, lifecycle and error paths, cancellation retry, and inner-product
option propagation.
