# CreateIndex

`Collection.CreateIndex(ctx, column, params, options)` validates an index
against the target field, backfills the complete current live-document
snapshot, and publishes the updated collection schema through a new immutable
manifest generation. The collection write lock covers all three phases, so
writes cannot cross the schema commit point.

The currently executable index types are:

- `FlatIndexParams` and `HNSWIndexParams` on supported dense or sparse vector
  fields;
- `IVFIndexParams` on supported dense vector fields;
- `HNSWRaBitQIndexParams` on FP32 dense vector fields with 64–4095
  dimensions and L2, IP, or cosine scoring;
- `VamanaIndexParams` on supported dense vector fields;
- `DiskANNIndexParams` on FP32 or FP16 dense vector fields;
- `InvertIndexParams` on every filterable scalar and array field, including
  BINARY and ARRAY_BINARY;
- `FTSIndexParams` on string fields, with the configured tokenizer and token
  filters validated against the complete live snapshot.

Dense Flat/HNSW/IVF definitions may use FP16, INT8, or INT4 scalar codes where
the vector data type permits them; INT8/INT4 may enable rotation. Sparse
Flat/HNSW supports unquantized or FP16-rounded values. HNSW-RaBitQ trains and
validates its centroid/rotation model and graph during backfill. Vamana
performs deterministic RobustPrune graph construction and supports the same
FP16/INT8/INT4 scalar representations as dense HNSW. DiskANN constructs its
graph and internal PQ codes during backfill and supports FP16/INT8/INT4 public
scalar representations independently of its `PQChunks` traversal encoding.
IVF SOAR returns `ErrNotSupported`. A vector index on a
scalar field, scalar index on a vector field, invalid metric/type combination,
nil parameters, or negative concurrency returns `ErrInvalidArgument`. A
missing column returns `ErrNotFound`. Different non-vector index types cannot
coexist on one column.

`CreateIndexOptions.Concurrency` bounds parallel work in backfills that expose
it, including IVF, RaBitQ, and DiskANN PQ training and assignment. Vamana construction is
deterministic and cancellation-aware; its current Go builder is serial. Zero
uses the runtime's default worker count. The operation is idempotent when the
column already has equal parameters; no new manifest is published in that
case. Supplying different parameters of the same implemented type rebuilds
and atomically replaces the definition.

```go
params := zvec.NewInvertIndexParams()
params.EnableExtendedWildcard = true
err := collection.CreateIndex(ctx, "title", params,
    zvec.CreateIndexOptions{Concurrency: 4},
)
```

The native Flat, HNSW, HNSW-RaBitQ, IVF, Vamana, DiskANN, INVERT, and FTS
collection search structures remain snapshot-local: query execution
reconstructs them from live documents.
CreateIndex persists validated parameters, not a C++-compatible or
segment-attached standalone index artifact. Backfill constructs the requested
runtime representation, including quantization overflow and rotation checks;
IVF, HNSW-RaBitQ, and DiskANN also complete deterministic training, while
Vamana completes graph construction and medoid selection. Existing WAL data, new
writes, Close/reopen, and later Flush all use the newly published schema.
If backfill, encoding, training, cancellation, or pre-commit manifest
publication fails, the previous in-memory and on-disk schema remains active.
An error after the atomic CURRENT replacement is reported, but the committed
schema remains the source of truth.
