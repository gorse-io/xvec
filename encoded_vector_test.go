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
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestReadOnlyQuantizedFlatEncodedVectors(t *testing.T) {
	ctx := context.Background()
	for _, quantize := range []QuantizeType{QuantizeTypeFP16, QuantizeTypeInt8, QuantizeTypeInt4} {
		for _, useMmap := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/mmap=%v", quantize, useMmap), func(t *testing.T) {
				params := NewFlatIndexParams(MetricTypeL2)
				params.Quantize = quantize
				params.Quantizer.EnableRotate = quantize != QuantizeTypeFP16
				schema := NewCollectionSchema("encoded_flat", FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Nullable: true, Index: params}, FieldSchema{Name: "rating", DataType: DataTypeInt32})
				path := filepath.Join(t.TempDir(), "collection")
				writer, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
				require.NoError(t, err)
				docs := annDenseDocuments(40)
				docs = append(docs, Document{PrimaryKey: "missing", Fields: map[string]any{"rating": int32(0)}})
				_, err = writer.Insert(ctx, docs)
				require.NoError(t, err)
				require.NoError(t, writer.Optimize(ctx, OptimizeOptions{}))
				queries := []VectorQuery{}
				expected := [][]Document{}
				for _, refine := range []bool{false, true} {
					queryParams := NewFlatQueryParams()
					queryParams.UseRefiner = refine
					queryParams.ScaleFactor = 100
					for _, projection := range []Projection{{OutputFields: []string{}}, {IncludeVectors: true}} {
						for _, byKey := range []bool{false, true} {
							query := VectorQuery{Field: "embedding", TopK: 8, Params: queryParams, Projection: projection, Filter: "rating >= 1", DenseVector: VectorFP32{.3, -.5, 1, .7}}
							if byKey {
								query.DenseVector = nil
								query.PrimaryKey = docs[7].PrimaryKey
							}
							want, err := writer.Query(ctx, query)
							require.NoError(t, err)
							queries = append(queries, query)
							expected = append(expected, want)
						}
					}
				}
				groupParams := NewFlatQueryParams()
				groupParams.UseRefiner = true
				groupQuery := GroupByVectorQuery{Field: "embedding", DenseVector: VectorFP32{.3, -.5, 1, .7}, Params: groupParams, GroupByField: "rating", GroupCount: 3, TopKPerGroup: 2, Projection: Projection{IncludeVectors: true}}
				wantGroups, err := writer.GroupByQuery(ctx, groupQuery)
				require.NoError(t, err)
				require.NoError(t, writer.Close())
				reader, err := Open(ctx, path, CollectionOptions{ReadOnly: true, EnableMmap: useMmap})
				require.NoError(t, err)
				defer func() { require.NoError(t, reader.Close()) }()
				for i, query := range queries {
					got, err := reader.Query(ctx, query)
					require.NoError(t, err)
					require.Equal(t, expected[i], got)
					if query.Projection.IncludeVectors && len(got) > 0 {
						got[0].Fields["embedding"].(VectorFP32)[0] += 100
						again, err := reader.Query(ctx, query)
						require.NoError(t, err)
						require.Equal(t, expected[i], again)
					}
				}
				snapshot := reader.querySnapshot.Load()
				require.IsType(t, encodedVectorFP32{}, snapshot.documents[0].Fields["embedding"])
				exact := snapshot.runtimes[0].indexes.denseExact["embedding"].(*lazyCollectionDenseFlatIndex)
				require.Nil(t, exact.index, "ordinary refinement must read originals without building a second exact index")
				require.Empty(t, exact.candidates)
				require.NotNil(t, exact.reader)
				groups, err := reader.GroupByQuery(ctx, groupQuery)
				require.NoError(t, err)
				require.Equal(t, wantGroups, groups)
				require.NotNil(t, exact.index)
			})
		}
	}
}

func TestEncodedFP32ValidationAndIndependentDecode(t *testing.T) {
	v := make(encodedVectorFP32, 8)
	binary.LittleEndian.PutUint32(v, math.Float32bits(1))
	binary.LittleEndian.PutUint32(v[4:], math.Float32bits(-2))
	require.NoError(t, v.validate())
	decoded, err := v.decode()
	require.NoError(t, err)
	require.Equal(t, VectorFP32{1, -2}, decoded)
	decoded[0] = 100
	again, err := v.decode()
	require.NoError(t, err)
	require.Equal(t, float32(1), again[0])
	require.Error(t, v[:7].validate())
	require.Error(t, v.readInto(make([]float32, 1)))
	binary.LittleEndian.PutUint32(v, math.Float32bits(float32(math.NaN())))
	require.Error(t, v.validate())
	_, err = v.decode()
	require.Error(t, err)
}

func TestReadOnlyEncodedVectorsSurviveConcurrentClose(t *testing.T) {
	ctx := context.Background()
	params := NewFlatIndexParams(MetricTypeL2)
	params.Quantize = QuantizeTypeFP16
	schema := NewCollectionSchema("encoded_close", FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 4, Index: params}, FieldSchema{Name: "rating", DataType: DataTypeInt32})
	path := filepath.Join(t.TempDir(), "collection")
	writer, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	_, err = writer.Insert(ctx, annDenseDocuments(8))
	require.NoError(t, err)
	require.NoError(t, writer.Optimize(ctx, OptimizeOptions{}))
	require.NoError(t, writer.Close())
	reader, err := Open(ctx, path, CollectionOptions{ReadOnly: true, EnableMmap: true})
	require.NoError(t, err)
	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0, 0, 0}, TopK: 3, Projection: Projection{IncludeVectors: true}}
	expected, err := reader.Query(ctx, query)
	require.NoError(t, err)
	snapshot := reader.querySnapshot.Load()
	runtime := snapshot.runtimes[0]
	require.IsType(t, encodedVectorFP32{}, snapshot.documents[0].Fields["embedding"])
	blocking := &blockingCollectionDenseIndex{collectionDenseIndex: runtime.indexes.denseFlat["embedding"], started: make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{})}
	runtime.indexes.denseNative["embedding"] = blocking
	runtime.indexes.denseFlat["embedding"] = blocking
	type queryResult struct {
		docs []Document
		err  error
	}
	done := make(chan queryResult, 1)
	go func() { docs, err := reader.Query(ctx, query); done <- queryResult{docs, err} }()
	select {
	case <-blocking.started:
	case <-time.After(5 * time.Second):
		t.Fatal("query did not start")
	}
	closed := make(chan error, 1)
	go func() { closed <- reader.Close() }()
	select {
	case err := <-closed:
		t.Fatalf("closed before query: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(blocking.release)
	result := <-done
	require.NoError(t, result.err)
	require.Equal(t, expected, result.docs)
	require.NoError(t, <-closed)
	// Projected vectors are owned and still readable after mappings were released.
	require.Equal(t, expected[0].Fields["embedding"], result.docs[0].Fields["embedding"])
}
