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
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSkipUnindexedSegments(t *testing.T) {
	ctx := context.Background()
	for _, skip := range []bool{false, true} {
		t.Run(map[bool]string{false: "default", true: "skip"}[skip], func(t *testing.T) {
			opts := NewCollectionOptions()
			opts.SkipUnindexedSegments = skip
			hnsw := NewHNSWIndexParams(MetricTypeIP)
			hnsw.M, hnsw.EFConstruction = 4, 16
			schema := NewCollectionSchema("skip", FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 2, Index: hnsw}, FieldSchema{Name: "flat", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewFlatIndexParams(MetricTypeIP)}, FieldSchema{Name: "group", DataType: DataTypeString})
			path := filepath.Join(t.TempDir(), "collection")
			c, err := CreateAndOpen(ctx, path, schema, opts)
			require.NoError(t, err)
			defer func() { require.NoError(t, c.Close()) }()
			doc := func(key string, v VectorFP32) Document {
				return Document{PrimaryKey: key, Fields: map[string]any{"v": v, "flat": v, "group": key}}
			}
			_, err = c.Insert(ctx, []Document{doc("indexed", VectorFP32{1, 0})})
			require.NoError(t, err)
			require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
			_, err = c.Insert(ctx, []Document{doc("tail", VectorFP32{2, 0})})
			require.NoError(t, err)
			before := c.indexBuildCount
			results, err := c.Query(ctx, VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 10})
			require.NoError(t, err)
			if skip {
				require.Equal(t, []string{"indexed"}, documentKeys(results))
			} else {
				require.Equal(t, []string{"tail", "indexed"}, documentKeys(results))
			}
			require.Equal(t, before, c.indexBuildCount)
			require.Len(t, c.segmentIndexes, 2, "restricted queries must not evict full runtimes")
			results, err = c.Query(ctx, VectorQuery{Field: "flat", DenseVector: VectorFP32{1, 0}, TopK: 10})
			require.NoError(t, err)
			require.Equal(t, []string{"tail", "indexed"}, documentKeys(results))
			results, err = c.Query(ctx, VectorQuery{TopK: 10})
			require.NoError(t, err)
			require.Len(t, results, 2)
			results, err = c.Query(ctx, VectorQuery{Field: "v", PrimaryKey: "tail", TopK: 10})
			require.NoError(t, err)
			if skip {
				require.Equal(t, []string{"indexed"}, documentKeys(results))
			}
			groups, err := c.GroupByQuery(ctx, GroupByVectorQuery{Field: "v", PrimaryKey: "tail", GroupByField: "group", GroupCount: 10, TopKPerGroup: 1})
			require.NoError(t, err)
			if skip {
				require.Len(t, groups, 1)
			} else {
				require.Len(t, groups, 2)
			}
			_, err = c.Update(ctx, []Document{{PrimaryKey: "indexed", Fields: map[string]any{"v": VectorFP32{3, 0}}}})
			require.NoError(t, err)
			results, err = c.Query(ctx, VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 10})
			require.NoError(t, err)
			if skip {
				require.Empty(t, results, "excluded updates must not resurrect older indexed versions")
			}
			_, err = c.Query(ctx, VectorQuery{Field: "v", DenseVector: VectorFP32{1}, TopK: 10})
			require.Error(t, err, "validate vectors even without eligible candidates")
			require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
			results, err = c.Query(ctx, VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 10})
			require.NoError(t, err)
			require.Equal(t, []string{"indexed", "tail"}, documentKeys(results))
			require.NoError(t, c.Close())
			c, err = Open(ctx, path, opts)
			require.NoError(t, err)
			results, err = c.Query(ctx, VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 10})
			require.NoError(t, err)
			require.Equal(t, []string{"indexed", "tail"}, documentKeys(results))
		})
	}
}
