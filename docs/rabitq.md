# RaBitQ training and distance estimation

This v0.4 component implements the portable RaBitQ model, converter, and
distance estimator used by the HNSW-RaBitQ index. Its behavioral baseline is
zvec commit `58375ff` and the locked RaBitQ-Library commit
`858b0d6c480766d0e4f08fc5e02f34b53d698fad`.

The converter remains internal to `internal/core`. Public
`HNSWRaBitQIndexParams` is executable through Collection queries and uses this
component without falling back to another index. See the
[HNSW-RaBitQ index](hnsw-rabitq.md) for graph behavior, refinement, and native
persistence.

## Model training

`TrainRaBitQ` accepts L2, inner-product, or cosine vectors with dimensions from
64 through 4095. `TotalBits` is in `[1,9]`, including the sign bit, and defaults
to 7. The default centroid count is 16. Training performs:

1. finite-value and dimension validation;
2. optional deterministic reservoir sampling;
3. k-means centroid training, with spherical centroids for IP and cosine;
4. a four-round random-sign FHT/Kac rotation, padded to a multiple of 64; and
5. deterministic expected-scale training over 100 unit Gaussian samples when
   extra bits are enabled.

Cosine inputs are normalized before training, encoding, and query preparation.
A fixed seed produces the same model state regardless of worker count. Long
training and batch encoding accept `context.Context` and propagate
cancellation.

`RaBitQModelState` contains all state required to restore a converter:
unrotated centroids, the four rotation-sign rounds, metric, dimensions, bit
width, and extra-code scale. State and centroid getters return independent
copies. A model fingerprint prevents a query prepared for one model from
silently evaluating another model's code.

## Portable split codes

Each encoded vector selects its nearest centroid and stores:

- one sign bit per padded coordinate in `BinaryCode`;
- `TotalBits-1` magnitude bits per coordinate in `ExtraCode`; and
- the additive, rescale, and probabilistic error factors needed by the
  estimator.

Both bit streams use a native Go, least-significant-bit-first layout. They are
portable across the supported Go platforms but are intentionally not the
AVX-oriented memory layout used by the C++ library. `QuantizedValues` expands a
code for diagnostics and compatibility fixtures. Returned bit slices are
copies, so codes remain immutable to callers.

The factor equations, sign handling, default bit width, dimension limits, and
fixed-scale conversion match the locked library. The Go trainer uses a fixed
PRNG rather than Eigen's process-global random state so rebuilds are explicit
and reproducible. Native Go and C++ code bytes are not interchangeable.

## Estimates and bounds

`PrepareQuery` rotates a query once and caches its centroid-dependent terms.
The resulting immutable query is safe for concurrent estimates:

- `EstimateCoarse` uses only the one-bit sign code;
- `Estimate` uses all configured bits and is normally more accurate; and
- one-bit models return the same result from both methods.

All estimator outputs are lower-is-better internal distances. L2 is squared
Euclidean distance, IP is `1 - dot`, and cosine is `1 - cosine`. An HNSW layer
must convert the internal IP distance back to the public higher-is-better score
when it publishes results.

`LowerBound` and `UpperBound` expose the locked implementation's randomized
error envelope. The full-code envelope uses the one-bit error factor divided
by `2^(TotalBits-1)`. It is intended for candidate pruning after random
rotation and is not a deterministic per-vector guarantee.

## Verification

Tests include factors and expanded codes produced by the locked C++ headers,
worker-count determinism, restore equivalence, all bit widths, padded and
maximum dimensions, L2/IP/cosine behavior, zero residuals, ownership,
cancellation, model mismatch detection, probabilistic quality checks, fuzzing,
and a microbenchmark:

```sh
go test ./internal/core -run '^TestRaBitQ'
go test ./internal/core -run '^$' -bench '^BenchmarkRaBitQEncodeEstimate$' -benchmem
go test ./internal/core -run '^$' -fuzz '^FuzzRaBitQEncodeEstimate$'
```
