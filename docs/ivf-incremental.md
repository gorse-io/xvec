# IVF incremental writes

This v0.3 unit turns the built or reopened IVF layout into a `DenseIndex` by
implementing the common `DenseStreamer.Add` contract. It accepts one unique
key and one finite, fixed-dimension FP32 vector per call. Inputs are cloned,
and a failed or canceled call leaves every key, vector, list, count, and
objective value unchanged.

## Online assignment

An index trained with at least `NList` vectors keeps its trained centroid set
fixed. Each incremental vector is scored against those centroids with the
configured metric and appended to the best list, using the lower list number
for equal scores.

An empty collection has no centroids to search. To make indexes created on
empty or very small collections streamable, the centroid set grows while its
size is below configured `NList`: each new vector becomes the next centroid
and the first member of its own list. This is also applied to a trained index
whose original vector count was below `NList`. Once the configured count is
reached, normal fixed-centroid assignment begins. Existing memberships are not
rewritten and centroids are not retrained during streaming.

The maintained objective is the sum of every vector's score to its assigned
centroid, negated for inner product so lower remains better. The original
training iteration and convergence fields stay historical; an index
bootstrapped entirely by streaming reports zero training iterations.

## Concurrency and durability

`IVFIndex` serializes additions with a write lock. Search, providers, list
inspection, and training metadata use read locks, allowing concurrent readers
without observing partially appended records. Search holds a single snapshot
view through centroid selection and candidate scoring.

Save clones a complete index generation under the read lock, releases the
lock, then validates, encodes, syncs, and atomically publishes that snapshot.
Writers are therefore blocked only for the memory snapshot, not disk I/O. A
reopened generation preserves streamed centroids, membership, vectors, counts,
and search results.

This core streamer is insert-only, matching the existing `DenseStreamer`
contract. Collection update and deletion continue to use immutable document
versions and live-version filtering; they do not mutate an index key in place.

Tests cover empty bootstrap, list growth and capping, fixed-centroid assignment,
full-probe equivalence to Flat, cloning and error atomicity, save/reopen after
streaming, and concurrent add/search/provider/save/reopen under the race
detector.
