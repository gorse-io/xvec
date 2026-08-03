# Rotation preprocessing and original-vector refinement

The second v0.3 implementation unit adds two independent `internal/core`
building blocks used by quantized and approximate indexes: a reversible random
rotation preprocessor and an exact original-vector candidate refiner. They do
not enable an ANN collection index by themselves.

## FHT/Kac rotation

`FHTRotator` follows the default rotator at pinned C++ commit `58375ff`. It
keeps the input dimension unchanged and performs four rounds with independent
random sign bits:

- a power-of-two dimension uses sign flip, Walsh-Hadamard transform, and
  `1/sqrt(d)` normalization in every round;
- every other dimension alternates the transform over the leading and trailing
  largest power-of-two subranges, applies the baseline Kac walk after each
  round, and finishes with a `1/4` scale.

The inverse applies those operations in reverse order. Rotation is orthogonal
apart from ordinary FP32 rounding, so it preserves inner products, squared L2
distances, and norms. There is no hidden zero padding and the output has exactly
the input dimension.

New rotators draw sign bytes from `crypto/rand`. `Signs` returns a clone of the
complete four-round state, and `NewFHTRotatorFromSigns` restores it exactly.
Index persistence units must store this state and must use the same state for
documents and queries after reopen. The state and derived fields are immutable,
so one rotator supports concurrent transformations. Batch rotation preserves
input order, limits worker concurrency, and propagates context cancellation.

`RotationReformer` exposes the operation through the general reversible
`DenseReformer` contract. Rotation may precede INT8 or INT4 quantization; public
index-parameter validation now accepts both combinations, matching the pinned
baseline.

## Original-vector refiner

`OriginalVectorRefiner` resolves candidate keys through a `DenseProvider`,
ignores approximate scores, and recomputes the configured metric from retained
FP32 originals. Candidate keys are deduplicated, caller filters are reapplied,
exact radius semantics are enforced, and equal final scores use ascending keys.
A missing original vector is an explicit error rather than a partially refined
or silently approximate result.

`RefinedSearch` implements the two-stage flow. It requests
`floor(topK * scaleFactor)` candidates (with a minimum of one), applies the
filter but no approximate-score radius at the base stage, and then applies
filter, radius, and top-k to exact scores. Scale factors must be finite,
positive, and free of integer overflow.

Tests cover fixed rotation state, power-of-two and arbitrary dimensions,
inverse and norm preservation, concurrent and batch transforms, fuzzed round
trips, exact reranking, duplicate keys, filtering, radius, cancellation,
missing originals, and scale arithmetic.
