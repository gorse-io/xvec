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

func TestImmutableHNSWDocumentsSharedAcrossQuerySnapshots(t *testing.T) {
	params := NewHNSWIndexParams(MetricTypeL2)
	params.M, params.EFConstruction, params.Quantize = 4, 16, QuantizeTypeInt4
	testImmutableDocumentsSharedAcrossQuerySnapshots(t, params)
}

func TestImmutableFlatDocumentsSharedAcrossQuerySnapshots(t *testing.T) {
	for _, quantize := range []QuantizeType{QuantizeTypeUndefined, QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		t.Run(fmt.Sprint(quantize), func(t *testing.T) {
			params := NewFlatIndexParams(MetricTypeL2)
			params.Quantize = quantize
			params.Quantizer.EnableRotate = quantize == QuantizeTypeInt4 || quantize == QuantizeTypeInt8
			testImmutableDocumentsSharedAcrossQuerySnapshots(t, params)
		})
	}
}

func testImmutableDocumentsSharedAcrossQuerySnapshots(t *testing.T, params IndexParams) {
	t.Helper()
	ctx := context.Background()
	schema := NewCollectionSchema("snapshot_memory", FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: params}, FieldSchema{Name: "rating", DataType: DataTypeInt32})
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "collection"), schema, NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, collection.Close()) }()
	_, err = collection.Insert(ctx, annDenseDocuments(32))
	require.NoError(t, err)
	require.NoError(t, collection.Optimize(ctx, OptimizeOptions{}))
	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0, 0, 0}, TopK: 100, Projection: Projection{IncludeVectors: true}}
	results, err := collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, results, 32)
	before := collection.querySnapshot.Load()
	require.NotEmpty(t, before.segments)
	require.False(t, before.segments[0].mutable)
	key := before.documents[0].PrimaryKey
	vector := before.documents[0].Fields["embedding"].(VectorFP32)
	// Public result mutation must not reach vectors borrowed by the index.
	results[0].Fields["embedding"].(VectorFP32)[0] += 100
	_, err = collection.Delete(ctx, []string{key})
	require.NoError(t, err)
	results, err = collection.Query(ctx, query)
	require.NoError(t, err)
	require.Len(t, results, 31)
	after := collection.querySnapshot.Load()
	require.Same(t, before.runtimes[0], after.runtimes[0])
	require.Same(t, &before.segments[0].documents[0], &after.segments[0].documents[0])
	require.Same(t, &vector[0], &after.segments[0].documents[0].Fields["embedding"].(VectorFP32)[0])
	for _, document := range results {
		require.NotEqual(t, key, document.PrimaryKey)
	}
	// Updating a vector must create a new generation, not mutate a shared row.
	updatedKey := results[0].PrimaryKey
	_, err = collection.Update(ctx, []Document{{PrimaryKey: updatedKey, Fields: map[string]any{"embedding": VectorFP32{100, 0, 0, 0}}}})
	require.NoError(t, err)
	query.DenseVector = VectorFP32{100, 0, 0, 0}
	results, err = collection.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, updatedKey, results[0].PrimaryKey)
	require.Equal(t, VectorFP32{100, 0, 0, 0}, results[0].Fields["embedding"])
}

func TestQuantizedFlatRuntimeDefersExact(t *testing.T) {
	ctx := context.Background()
	for _, quantize := range []QuantizeType{QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		t.Run(fmt.Sprint(quantize), func(t *testing.T) {
			params := NewFlatIndexParams(MetricTypeL2)
			params.Quantize = quantize
			params.Quantizer.EnableRotate = quantize != QuantizeTypeFP16
			field := FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: params}
			schema := NewCollectionSchema("flat_memory", field)
			documents := annDenseDocuments(40)
			for j := range documents {
				documents[j].DocID = uint64(j + 1)
			}
			indexes, err := buildCollectionRuntimeIndexes(ctx, schema, documents, 2, 0, false, nil)
			require.NoError(t, err)
			defer func() { require.NoError(t, indexes.Close()) }()
			lazy, ok := indexes.denseExact[field.Name].(*lazyCollectionDenseFlatIndex)
			require.True(t, ok)
			require.Nil(t, lazy.index)
			query := []float32{.75, .25, -.5, .125}
			flat := indexes.denseFlat[field.Name]
			require.Same(t, flat, indexes.denseNative[field.Name])
			_, err = flat.SearchWithOptions(ctx, query, core.SearchOptions{TopK: 12})
			require.NoError(t, err)
			groups := core.GroupByOptions{GroupCount: 4, TopKPerGroup: 2, Resolve: func(key uint64) (string, bool) { return fmt.Sprint(key % 4), true }}
			_, err = flat.(core.DenseGroupSearcher).SearchGroups(ctx, query, groups)
			require.NoError(t, err)
			require.Nil(t, lazy.index, "quantized searches must not materialize exact FP32 storage")
			exact, err := buildDenseFlatIndex(ctx, field, core.MetricL2, documents)
			require.NoError(t, err)
			want, err := exact.(core.DenseGroupSearcher).SearchGroups(ctx, query, groups)
			require.NoError(t, err)
			got, err := lazy.SearchGroups(ctx, query, groups)
			require.NoError(t, err)
			require.Equal(t, want, got)
			require.NotNil(t, lazy.index, "grouped refinement must still support exact scoring")
		})
	}
}
