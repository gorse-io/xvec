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
	"fmt"
	"path/filepath"
	"sync"
	"testing"

	core "github.com/gorse-io/xvec/internal/core/algorithm"
	"github.com/stretchr/testify/require"
)

func TestHNSWRuntimeSharesFlatAndDefersExact(t *testing.T) {
	ctx := context.Background()
	for _, quantize := range []QuantizeType{QuantizeTypeUndefined, QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		t.Run(fmt.Sprint(quantize), func(t *testing.T) {
			params := NewHNSWIndexParams(MetricTypeL2)
			params.M, params.EFConstruction, params.Quantize = 4, 16, quantize
			params.Quantizer.EnableRotate = quantize == QuantizeTypeInt4 || quantize == QuantizeTypeInt8
			field := FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Nullable: true, Index: params}
			schema := NewCollectionSchema("shared_hnsw", field)
			documents := annDenseDocuments(48)
			for j := range documents {
				documents[j].DocID = uint64(j + 1)
			}
			documents = append(documents, Document{DocID: 1000, PrimaryKey: "missing", Fields: map[string]any{}})
			spec, err := resolveCollectionVectorIndex(field, "test", "")
			require.NoError(t, err)
			eagerFlat, err := buildCollectionDenseFlat(ctx, schema.Name, field, documents, spec)
			require.NoError(t, err)
			eagerExact, err := buildDenseFlatIndex(ctx, field, spec.metric, documents)
			require.NoError(t, err)
			query := []float32{.75, .25, -.5, .125}
			options := core.SearchOptions{TopK: 12, Filter: func(key uint64) bool { return key%3 != 0 }, Radius: 10}
			wantFlat, err := eagerFlat.SearchWithOptions(ctx, query, options)
			require.NoError(t, err)
			wantExact, err := eagerExact.SearchWithOptions(ctx, query, options)
			require.NoError(t, err)
			groups := core.GroupByOptions{GroupCount: 4, TopKPerGroup: 2, Filter: options.Filter, Radius: 10, Resolve: func(key uint64) (string, bool) { return fmt.Sprint(key % 4), true }}
			wantGroups, err := eagerFlat.(core.DenseGroupSearcher).SearchGroups(ctx, query, groups)
			require.NoError(t, err)

			var artifacts map[string]string
			for _, reopen := range []bool{false, true} {
				indexes, err := buildCollectionRuntimeIndexes(ctx, schema, documents, 2, 0, false, artifacts)
				require.NoError(t, err)
				lazy, ok := indexes.denseExact[field.Name].(*lazyCollectionDenseFlatIndex)
				require.True(t, ok)
				require.Nil(t, lazy.index, "exact storage must not be built for ANN queries")
				require.Equal(t, 48, lazy.Len())
				flat := indexes.denseFlat[field.Name]
				if quantize == QuantizeTypeUndefined {
					require.Same(t, lazy, flat)
				} else {
					require.IsType(t, &core.ScalarQuantizedFlatIndex{}, flat)
					got, err := flat.SearchWithOptions(ctx, query, options)
					require.NoError(t, err)
					require.Equal(t, wantFlat, got)
					gotGroups, err := flat.(core.DenseGroupSearcher).SearchGroups(ctx, query, groups)
					require.NoError(t, err)
					require.Equal(t, wantGroups, gotGroups)
					require.Nil(t, lazy.index, "quantized linear and grouped searches should not allocate exact storage")
				}
				// Concurrent first use must publish one complete exact fallback, and a
				// canceled attempt must not prevent a later successful initialization.
				canceled, cancel := context.WithCancel(ctx)
				cancel()
				_, err = lazy.SearchWithOptions(canceled, query, options)
				require.ErrorIs(t, err, context.Canceled)
				require.Nil(t, lazy.index)
				results := make([][]core.Result, 8)
				errs := make([]error, 8)
				var wg sync.WaitGroup
				for j := range results {
					wg.Go(func() { results[j], errs[j] = lazy.SearchWithOptions(ctx, query, options) })
				}
				wg.Wait()
				for j := range results {
					require.NoError(t, errs[j])
					require.Equal(t, wantExact, results[j])
				}
				require.NotNil(t, lazy.index)
				if !reopen {
					path := filepath.Join(t.TempDir(), "hnsw")
					require.NoError(t, indexes.denseNative[field.Name].(interface {
						Save(context.Context, string) error
					}).Save(ctx, path))
					artifacts = map[string]string{collectionIndexArtifactKey(field.Name, collectionVectorArtifactKind(IndexTypeHNSW)): path}
				}
				require.NoError(t, indexes.Close())
			}
		})
	}
}
