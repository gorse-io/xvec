# IVF-RaBitQ index

IVF-RaBitQ combines inverted-file list probing with portable RaBitQ codes for
FP32 dense vectors. It supports L2, inner-product, and cosine metrics and uses
`IndexTypeIVFRaBitQ`, whose stable value is `7` to match zvec-go v0.7.0.

## Construction

`IVFRaBitQIndexParams` exposes the zvec-go IVF-RaBitQ controls:

- `NList` selects the requested number of coarse inverted lists;
- `TotalBits` selects the RaBitQ width in `[1, 9]` and defaults to `7`; an
  explicit zero also selects the native default;
- `SampleCount` limits deterministic training samples; zero uses all vectors.

The builder validates and owns the input vectors, trains the IVF centroids and
RaBitQ model, assigns every vector to one coarse list, and stores one immutable
RaBitQ code per vector. Empty and small inputs cap the effective list count to
the number of vectors. FP32 dimensions must be in `[64, 4095]`.

## Query

`IVFRaBitQQueryParams` exposes:

- `NProbe`, the number of nearest coarse lists to scan;
- `ScaleFactor`, the candidate multiplier used before exact refinement;
- `Radius`, `Linear`, and `UseRefiner` through `QueryOptions`.

Non-linear search selects the nearest `NProbe` IVF centroids and estimates only
the RaBitQ codes in those lists. Indexes with at most 1,000 vectors scan all
encoded clusters, matching zvec's small-index threshold. `Linear` skips IVF selection and scans all
RaBitQ codes; it does not silently replace the index with Flat. Filters and
metric-aware radius are applied to approximate scores. With `UseRefiner`, xvec
requests `floor(TopK * ScaleFactor)` candidates and reranks them from retained
original FP32 vectors.

The same query parameter type is accepted by ordinary vector queries,
`SubQuery` entries in `MultiQuery`, and group-by vector queries. Group-by scans
the selected IVF lists directly so a global top-k cannot hide requested groups.
As in zvec-go v0.7.0, IVF-RaBitQ group-by rejects `UseRefiner`.

## Persistence

The native `.zvi` artifact stores the IVF model and postings, RaBitQ model state,
and codes in one versioned CRC32C-protected record. Writes use xvec's atomic
file publication helper. Decoding validates the record size, checksum, model
fingerprint, dimensions, code cardinality, and per-code lengths before the
index becomes visible. The format is native Go and is intentionally not binary
compatible with C++ zvec artifacts.

Flush persists newly immutable segment indexes. Close/reopen restores the same
model, codes, and list assignments; Optimize deterministically rebuilds them
from retained original vectors.

## Verification

```sh
go test ./internal/core/algorithm -run '^TestIVFRaBitQ'
go test ./ -run 'IVFRaBitQ' -count=1
```
