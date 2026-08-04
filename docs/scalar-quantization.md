# Scalar vector quantization

The first v0.3 implementation unit provides dependency-free FP16, INT8, and
INT4 scalar quantization primitives in `internal/core`. The completed v0.3 ANN
integration first connected them to dense Flat, IVF, and HNSW collection query
paths; current collection execution also supports Vamana and DiskANN while
retaining original vectors for optional refinement.

## Encodings

| Encoding | Code bytes | Reconstruction |
| --- | ---: | --- |
| FP16 | `2 * dimension` | IEEE 754 binary16, little-endian, round-to-nearest-even |
| INT8 | `dimension` | `inverseScale * int8(code) + offset` |
| INT4 | `dimension / 2` | the same affine formula with signed 4-bit codes |

INT8 maps each non-constant input vector to the signed range `[-127, 127]`.
INT4 maps it to `[-8, 7]`; element 0 occupies the low nibble and element 1 the
high nibble, matching the layout at pinned C++ commit `58375ff`. Because the
baseline codec reads pairs, INT4 dimensions must be even. Schema validation
rejects an odd INT4 dimension before any index build starts.

The integer range is chosen independently for every vector. The packed codes
therefore carry their own inverse scale and offset rather than depending on a
trained global codebook. Constant vectors use zero codes, a zero inverse scale,
and the constant as their offset. This deterministic special case reconstructs
the constant exactly and avoids unstable division by a zero range.

FP16 conversion rejects finite FP32 values that round to infinity. All three
encoders reject empty and non-finite inputs. Encoded data is immutable to
callers: accessors and decode operations return fresh slices.

## Distance kernels

The scalar kernels implement squared L2, inner product, cosine distance, and
the pinned localized-spherical MIPS-to-L2 score. For INT8 and INT4, code sums,
squared sums, and code dot products evaluate the same algebra as decoding both
vectors without allocating decoded buffers. FP16 uses the existing exact
metric implementation after binary16 expansion.

Integer queries are quantized per query with the candidate encoding. Batch
conversion preserves input order, bounds worker concurrency, propagates
context cancellation, and never aliases caller-owned vectors.

The implementation is covered by all-binary16-pattern round trips, pinned
integer code examples, decoded-distance equivalence tests, cancellation and
corruption checks, and a fuzz target.

## Collection execution

Dense FP32 Flat, HNSW, IVF, Vamana, and DiskANN fields accept FP16, INT8, or
INT4 scalar representations. INT8 and INT4 may apply a deterministic
dimension-preserving rotation before quantization. Collection Linear queries
use the matching scalar Flat truth path, while `UseRefiner` reranks candidates
against retained unmodified vectors.

For DiskANN, scalar quantization and product quantization have distinct jobs.
The optionally rotated scalar representation supplies graph-build vectors and
candidate scores. DiskANN then trains its mandatory 8-bit `PQChunks` model over
that representation solely to order the graph frontier. CreateIndex, writes,
Optimize, and reopen validate or reconstruct both layers without treating one
as a substitute for the other.
