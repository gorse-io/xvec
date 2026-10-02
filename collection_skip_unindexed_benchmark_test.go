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
	"testing"

	"github.com/stretchr/testify/require"
)

// Compare an indexed corpus plus a larger WAL-backed tail. Cold includes query
// snapshot reconstruction; warm measures searching the retained snapshot.
func BenchmarkSkipUnindexedSegments(b *testing.B) {
	for _, skip := range []bool{false, true} {
		for _, cold := range []bool{false, true} {
			b.Run(fmt.Sprintf("skip=%t/cold=%t", skip, cold), func(b *testing.B) {
				ctx := context.Background()
				hnsw := NewHNSWIndexParams(MetricTypeIP)
				hnsw.M, hnsw.EFConstruction = 8, 32
				options := NewCollectionOptions()
				options.SkipUnindexedSegments = skip
				c, err := CreateAndOpen(ctx, filepath.Join(b.TempDir(), "collection"), NewCollectionSchema("benchmark", FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 32, Index: hnsw}), options)
				require.NoError(b, err)
				b.Cleanup(func() { require.NoError(b, c.Close()) })
				documents := func(start, count int) []Document {
					docs := make([]Document, count)
					for i := range docs {
						v := make(VectorFP32, 32)
						for j := range v {
							v[j] = float32((start+i+j)%97) / 97
						}
						docs[i] = Document{PrimaryKey: fmt.Sprint(start + i), Fields: map[string]any{"v": v}}
					}
					return docs
				}
				_, err = c.Insert(ctx, documents(0, 512))
				require.NoError(b, err)
				require.NoError(b, c.Optimize(ctx, OptimizeOptions{}))
				_, err = c.Insert(ctx, documents(512, 4096))
				require.NoError(b, err)
				vector := make(VectorFP32, 32)
				vector[0] = 1
				query := VectorQuery{Field: "v", DenseVector: vector, TopK: 10}
				results, err := c.Query(ctx, query)
				require.NoError(b, err)
				require.Len(b, results, 10)
				before := c.indexBuildCount
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if cold {
						c.mu.Lock()
						c.invalidateQuerySnapshotLocked()
						c.mu.Unlock()
					}
					_, err = c.Query(ctx, query)
					if err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				require.Equal(b, before, c.indexBuildCount)
			})
		}
	}
}
