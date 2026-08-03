# IVF search

This independently gated v0.3 unit makes the immutable IVF layout searchable.
It remains internal until persistence/reopen and incremental-write units are
complete, so collection queries still never substitute an in-memory index for
an unsupported durable IVF definition.

## Query flow

`IVFSearchOptions` combines the common `TopK`, `Radius`, and candidate `Filter`
controls with `NProbe`. `NProbe` must be positive; values larger than the
effective list count are capped. The generic `Search` and `SearchWithOptions`
contracts use the pinned public default of 10 probes, while `SearchIVF` accepts
the explicit value from future `IVFQueryParams` orchestration.

One query executes two metric-aware stages:

1. rank trained centroids with the index metric, breaking equal scores by the
   lower centroid index;
2. concatenate only the selected lists and score their retained original FP32
   vectors with the same metric.

Filter, radius, and final top-k are applied to original-vector scores, not to
centroid scores. Inner product uses a minimum-similarity radius; L2, cosine,
and MIPS-L2 use a maximum distance. Equal final scores use ascending document
keys. No candidate occurs in more than one list.

`ProbedLists` exposes the ordered centroid indexes for diagnostics and future
query statistics. An empty built index validates the query and returns non-nil
empty probe and result slices.

## Exactness boundary

With `NProbe == NList`, every stored vector is scanned exactly once and results
must match Flat byte-for-byte for all four metrics. With fewer probes, IVF is
approximate only because unselected lists are absent; vectors inside selected
lists still receive exact original-vector scores. Quantized list scoring and
optional original-vector refinement are connected in the later v0.3
integration unit.

Tests cover nearest-list selection, probe capping, partial-list results,
full-probe equivalence to Flat for L2/IP/cosine/MIPS-L2, filters, both radius
directions, stable ties, empty indexes, default query behavior, cancellation,
and invalid query controls.
