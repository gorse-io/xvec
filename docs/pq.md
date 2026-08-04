# Product quantization

The v0.4 PQ component is the pure-Go codebook and distance-table layer needed
by the later DiskANN index. Its behavioral baseline is zvec commit 58375ff;
its model representation is native Go and is not a C++ artifact reader.

## Training and layout

TrainPQ accepts finite FP32 training vectors and PQOptions. It retains the
baseline fixed settings:

- eight bits and 256 centroid IDs per chunk;
- at most the first 200,000 training vectors;
- 12 Lloyd update rounds by default;
- contiguous dimension chunks; and
- auto chunk count equal to half the dimension.

When dimensions do not divide evenly, earlier chunks receive one extra
dimension. For example, dimension 10 split into three chunks has offsets
[0, 4, 7, 10]. An explicit chunk count must be between one and the vector
dimension. Dimension one therefore needs Chunks=1; its auto value is invalid,
matching the baseline divide-by-chunk boundary.

Each chunk is trained independently with the existing deterministic Go
k-means framework. The baseline length-32 KMC2 initializer is retained but
given a portable deterministic seed instead of process entropy. Assignment,
centroid update, iteration limit, chunk ordering, and output layout retain the
baseline semantics, and results are bit-for-bit stable across worker counts.
If a training set has fewer than 256 rows, unused pivot rows repeat centroid
zero; lower-ID tie breaking ensures those padding rows are never emitted.

PQModelState.Pivots uses the baseline full-pivot representation: it is a
256-by-dimension centroid-major matrix. Row c contains centroid c for every
chunk, and ChunkOffsets selects the independently trained portion. State and
RestorePQModel deep-copy and validate this complete representation.

## Codes and distance tables

Encode emits one unsigned byte per chunk. Decode reconstructs the selected
chunk portions into FP32. Codes carry a model fingerprint in memory, so a code
cannot be decoded or scored by different pivots even when dimensions and chunk
counts happen to match. Code safely restores raw bytes against one model.
Batch encoding preserves input order and supports cancellation.

DistanceTable stores 256 public scores per chunk in chunk-major order:

- L2 entries are squared subspace distances; and
- inner-product entries are subspace similarities.

Lookup adds the entry selected by each code byte. The result therefore equals
the public metric score between the query and the PQ-reconstructed vector,
apart from normal floating-point rounding. LookupBatch evaluates many codes
concurrently in stable input order.

The low-level PQ model deliberately accepts only L2 and inner product. Cosine
and MIPS-L2 DiskANN paths require their metric-specific normalization or
augmentation before PQ and will be connected in the DiskANN unit. Calling PQ
directly with those metrics returns an explicit unsupported-metric error.

## Verification

Tests lock down the baseline chunk split and pivot/table layout, L2 and inner
product fixtures, deterministic training, prefix sample cap, lossy codebook
training beyond 256 samples, ownership, restore and model mismatch behavior,
context cancellation, overflow, fuzz safety, and a zero-allocation lookup
benchmark:

    go test ./internal/core -run '^(TestPQ|ExampleTrainPQ)'
    go test ./internal/core -run '^$' -bench '^BenchmarkPQDistanceLookup$' -benchmem
    go test ./internal/core -run '^$' -fuzz '^FuzzRestorePQModel$'
