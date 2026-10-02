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

// Model Gorse's dense-vector batches with indexed category and timestamp fields.
// The seed also measures preparing indexes over an existing mutable segment.
func BenchmarkCollectionUpsertBatch(b *testing.B) {
	for _, size := range []int{100, 1000} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			ctx := context.Background()
			schema := NewCollectionSchema("upsert",
				FieldSchema{Name: "vector", DataType: DataTypeVectorFP16, Dimension: 512, Index: NewHNSWIndexParams(MetricTypeL2)},
				FieldSchema{Name: "categories", DataType: DataTypeArrayString, Index: NewInvertIndexParams()},
				FieldSchema{Name: "timestamp", DataType: DataTypeInt64, Index: NewInvertIndexParams()},
			)
			docs := make([]Document, size)
			for i := range docs {
				v := make(VectorFP16, 512)
				v[0] = Float16FromFloat32(float32(i))
				docs[i] = Document{PrimaryKey: fmt.Sprint(i), Fields: map[string]any{
					"vector": v, "categories": StringArray{"all", fmt.Sprint(i % 10)}, "timestamp": int64(i),
				}}
			}
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				c, err := CreateAndOpen(ctx, filepath.Join(b.TempDir(), "collection"), schema, NewCollectionOptions())
				require.NoError(b, err)
				_, err = c.Insert(ctx, docs)
				require.NoError(b, err)
				b.StartTimer()
				_, err = c.Upsert(ctx, docs)
				b.StopTimer()
				require.NoError(b, err)
				require.NoError(b, c.Close())
			}
		})
	}
}
