# K-means training framework

The third v0.3 implementation unit adds a reusable, pure-Go Lloyd k-means
trainer in `internal/core`. IVF, PQ, and later clustered quantizers consume the
same immutable model instead of carrying private training loops.

## Baseline-oriented defaults

`DefaultKMeansOptions` uses the behavior family from pinned C++ commit
`58375ff`:

- at most 20 Lloyd update rounds;
- absolute objective change below float32 epsilon stops training;
- initial centroids are sampled uniformly without replacement;
- a requested cluster count larger than the training set is reduced to the
  number of samples;
- assignment uses the configured vector metric and centroid updates use an
  FP64-accumulated arithmetic mean.

Unlike the baseline's process-random reservoir, the Go sampler uses a fixed
SplitMix64 algorithm and an explicit seed. A seed therefore produces the same
sampling sequence on every supported architecture and Go version. Ordered
accumulation then makes a training run independent of worker scheduling. The
optional k-means++ initializer chooses its first sample uniformly and uses
squared-L2 weights for subsequent spacing.

IVF passes its public `NIterations` value when it invokes this framework; the
framework's 20-round default does not override the IVF default of 10.

## Determinism and cancellation

Vector-to-centroid assignment is parallel and bounds concurrency through the
shared worker helper. Each worker writes only its input position. Centroid sums
are then accumulated in original input order using FP64, so worker scheduling
cannot change labels, centroids, counts, objective values, or convergence.

Training, initialization, assignment, validation, and centroid updates poll
the supplied context. Cancellation returns an error and never publishes a
partial model. Input vectors and optional initial centroids must be non-empty,
dimensionally uniform, and finite; score overflow is also an error.

## Empty clusters and spherical updates

Empty clusters use one explicit policy:

- `KMeansEmptyKeep` retains their preceding centers;
- `KMeansEmptyReseedFarthest` deterministically moves them to distinct
  worst-assigned samples and is the Go default;
- `KMeansEmptyDrop` removes them, matching baseline `purge_empty` behavior.

Reseeding or dropping prevents a false convergence decision for that round.
Optional spherical mode normalizes each non-zero updated centroid, which later
RaBitQ and angular-clustering paths can use. A zero mean remains the zero
vector with the established zero-vector metric semantics.

## Model contract

`KMeansModel` owns all centroid and count storage. Its accessors return clones,
and `Nearest` and `Classify` use stable lower-centroid-index tie breaking. The
reported objective is lower-is-better: distance scores are summed, while inner
product similarities are negated. Final labels, counts, and cost are always
recomputed against the final centroid set.

Tests cover exact small-cluster results, fixed reservoir state, k-means++ and
reservoir reproducibility across worker counts, empty policies, spherical inner
product, ownership, validation, cancellation, tie breaking, fuzzed training,
and a repeatable training benchmark.
