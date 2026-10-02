// SPDX-License-Identifier: Apache-2.0

package xvec

import (
	"context"
	"path/filepath"
	"testing"

	core "github.com/gorse-io/xvec/internal/core/algorithm"
	"github.com/stretchr/testify/require"
)

func TestCollectionWriterAppendPreservesLeasedSnapshot(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("native",
		FieldSchema{Name: "vector", DataType: DataTypeVectorFP16, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeL2)},
	)
	path := filepath.Join(t.TempDir(), "native")
	c, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	insert := func(key string, x float32) {
		t.Helper()
		v := VectorFP16{Float16FromFloat32(x), 0}
		_, err := c.Insert(ctx, []Document{{PrimaryKey: key, Fields: map[string]any{"vector": v}}})
		require.NoError(t, err)
		v[0] = Float16FromFloat32(100)
	}
	current := func() *collectionSegmentRuntime {
		for _, runtime := range c.segmentIndexes {
			return runtime
		}
		return nil
	}
	insert("a", 1)
	first := current()
	insert("b", 2)
	require.Same(t, first, current(), "unleased append should extend the existing runtime")
	c.mu.RLock()
	snapshot, release, err := c.acquireQuerySnapshotLocked(ctx)
	c.mu.RUnlock()
	require.NoError(t, err)
	insert("c", 3)
	second := current()
	require.NotSame(t, first, second, "a leased runtime must remain immutable")
	require.Len(t, first.documents, 2)
	require.Len(t, first.documentOrdinals, 2)
	field, _ := schema.Field("vector")
	results, err := c.searchVectorSegments(ctx, "test", field, VectorFP16{Float16FromFloat32(1), 0}, nil, 10, nil,
		snapshot.segments, snapshot.runtimes, snapshot.liveFilter)
	require.NoError(t, err)
	require.Len(t, results, 2, "the old snapshot must not see newly appended Flat rows")
	require.Equal(t, float32(0), results[0].Score, "Insert must copy the caller's FP16 bits")
	release()
	insert("d", 4)
	require.Same(t, second, current())
	require.Len(t, second.documentOrdinals, 4)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{Concurrency: 2}))
	require.NoError(t, c.Close())
	c, err = Open(ctx, path, CollectionOptions{ReadOnly: true, EnableMmap: true})
	require.NoError(t, err)
	docs, err := c.Query(ctx, VectorQuery{Field: "vector", DenseVector: VectorFP16{Float16FromFloat32(1), 0}, TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b", "c", "d"}, documentKeys(docs))
}

func TestCollectionWriterAppendRetainsFlatQuantization(t *testing.T) {
	ctx := context.Background()
	params := NewFlatIndexParams(MetricTypeL2)
	params.Quantize = QuantizeTypeInt8
	schema := NewCollectionSchema("quantized", FieldSchema{Name: "vector", DataType: DataTypeVectorFP32, Dimension: 2, Index: params})
	c, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "quantized"), schema, NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	for _, key := range []string{"a", "b"} {
		_, err := c.Insert(ctx, []Document{{PrimaryKey: key, Fields: map[string]any{"vector": VectorFP32{1, 2}}}})
		require.NoError(t, err)
		for _, runtime := range c.segmentIndexes {
			require.IsType(t, &core.ScalarQuantizedFlatIndex{}, runtime.indexes.denseNative["vector"])
		}
	}
}
