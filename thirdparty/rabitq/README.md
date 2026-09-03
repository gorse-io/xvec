# Scalar RaBitQ primitives

This package is a pure-Go, scalar port of the RaBitQ functionality used by
Alibaba zvec. It is based on
[VectorDB-NTU/RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library)
commit `540242ea0a68926f1b827bf1f9add844f07a427b`, the revision pinned by zvec.

The package is standalone: this change does not connect it to xvec's existing
index implementations. It uses no CGO, `unsafe`, architecture intrinsics, or
runtime SIMD dispatch.

Implemented upstream surface:

- `choose_rotator`, matrix rotation, FHT-Kac rotation, and raw rotator state
- `faster_config`
- `BinDataMap`, `BatchDataMap`, and `ExDataMap` byte layouts
- `quantize_split_single` and `quantize_split_batch`
- `SplitSingleQuery` and patched reusable `SplitBatchQuery::reset`
- `split_single_estdist`, `split_single_fulldist`, `split_batch_estdist`, and
  `split_distance_boosting`
- `select_excode_ipfunc`, `dot_product`, and `euclidean_sqr`

The extra-bit count is `0..8`, corresponding to zvec's total-bit range `1..9`.
Encoded dimensions are positive multiples of 64. zvec's FHT input dimension is
`64..4095` and is normally padded to at most `4096`; manually padded and matrix
rotator states may be larger. FastScan batches contain 32 vectors. The
batch estimator implements the high-accuracy 16-bit LUT path used by zvec;
SIMD implementations can be added later without changing the serialized
layouts.
Full-distance refinement and boosting require at least one extra bit; a
one-bit-only configuration uses the estimate path.

`NewFasterConfig` reproduces the pinned libstdc++ `mt19937` and
`normal_distribution<double>` sample sequence used by zvec's RaBitQ build.
Callers may also supply a previously persisted `FasterConfig.TConst` directly.
Matrix generation is statistically equivalent but not byte-identical to Eigen's
Householder-QR constructor; loading raw matrix state is the interoperability
path for existing C++ indexes.
