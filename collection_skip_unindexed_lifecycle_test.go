// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package xvec

import (
	"context"
	"testing"

	"github.com/gorse-io/xvec/internal/db"
	"github.com/stretchr/testify/require"
)

func TestSkipUnindexedSegmentsDoesNotDecodeExcludedPayload(t *testing.T) {
	ctx := context.Background()
	c := skipTestCollection(t, FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)})
	_, err := c.Insert(ctx, []Document{{PrimaryKey: "indexed", Fields: map[string]any{"v": VectorFP32{1, 0}}}})
	require.NoError(t, err)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
	// Bypass the codec to plant a decoding trap in an excluded tail. Restricted
	// queries must neither decode this segment nor require a prepared runtime.
	_, err = c.store.Insert(ctx, []db.WriteInput{{PrimaryKey: "tail", Payload: []byte("invalid document encoding")}})
	require.NoError(t, err)
	results, err := c.Query(ctx, VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"indexed"}, documentKeys(results))
	_, err = c.Query(ctx, VectorQuery{TopK: 10})
	require.Error(t, err, "the decoding trap must fail a full snapshot")
}

func TestSkipUnindexedSegmentsReopenAndCreateIndex(t *testing.T) {
	ctx := context.Background()
	index := NewHNSWIndexParams(MetricTypeIP)
	c := skipTestCollection(t, FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 2, Index: index})
	_, err := c.Insert(ctx, []Document{{PrimaryKey: "tail", Fields: map[string]any{"v": VectorFP32{1, 0}}}})
	require.NoError(t, err)
	query := VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 10}
	results, err := c.Query(ctx, query)
	require.NoError(t, err)
	require.Empty(t, results)
	path := c.path
	require.NoError(t, c.Close())
	for _, skip := range []bool{true, false} {
		options := NewCollectionOptions()
		options.ReadOnly, options.SkipUnindexedSegments = true, skip
		reopened, err := Open(ctx, path, options)
		require.NoError(t, err)
		results, err = reopened.Query(ctx, query)
		require.NoError(t, err)
		if skip {
			require.Empty(t, results)
		} else {
			require.Len(t, results, 1)
		}
		require.NoError(t, reopened.Close())
	}
	options := NewCollectionOptions()
	options.SkipUnindexedSegments = true
	reopened, err := Open(ctx, path, options)
	require.NoError(t, err)
	defer func() { require.NoError(t, reopened.Close()) }()
	results, err = reopened.Query(ctx, query)
	require.NoError(t, err)
	require.Empty(t, results)
	require.NoError(t, reopened.CreateIndex(ctx, "v", index, CreateIndexOptions{}))
	results, err = reopened.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, results, 1, "index publication must invalidate the restricted snapshot")
}
