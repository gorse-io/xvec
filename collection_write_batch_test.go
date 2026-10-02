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
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCollectionWriteBatchPreparesRuntimeOnce(t *testing.T) {
	for _, operator := range []Operator{OperatorUpsert, OperatorUpdate} {
		t.Run(operator.String(), func(t *testing.T) {
			ctx := context.Background()
			schema := testPublicCollectionSchema()
			schema.Fields[1].Index = NewInvertIndexParams()
			schema.Fields[2].Index = NewInvertIndexParams()
			schema.Fields[3].Index = NewHNSWIndexParams(MetricTypeIP)
			path := filepath.Join(t.TempDir(), "collection")
			c, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, c.Close()) })
			docs := make([]Document, 64)
			for i := range docs {
				docs[i] = testPublicDocument(fmt.Sprint(i), "title", "old", int32(i), 1, []float32{1, 0})
			}
			_, err = c.Insert(ctx, docs)
			require.NoError(t, err)
			// Retain a cold-built query snapshot to check that a write invalidates it.
			_, err = c.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 64, Filter: "category='old'"})
			require.NoError(t, err)
			for i := range docs {
				docs[i] = Document{PrimaryKey: fmt.Sprint(i), Fields: map[string]any{"category": "new", "rating": int32(100 + i)}}
			}
			before := c.indexBuildCount
			results, err := c.writeDocuments(ctx, operator, docs)
			require.NoError(t, err)
			require.Len(t, results, len(docs))
			require.Equal(t, before+1, c.indexBuildCount, "mutable runtime should be prepared once per batch")
			check := func() {
				got, err := c.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 64, Filter: "category='new' AND rating>=100"})
				require.NoError(t, err)
				require.Len(t, got, len(docs))
				old, err := c.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 64, Filter: "category='old'"})
				require.NoError(t, err)
				require.Empty(t, old)
			}
			check()
			require.NoError(t, c.Close())
			c, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)
			check()
		})
	}
}

func TestCollectionWriteBatchDuplicateKeysAndPartialErrors(t *testing.T) {
	for _, operator := range []Operator{OperatorUpsert, OperatorUpdate} {
		t.Run(operator.String(), func(t *testing.T) {
			ctx := context.Background()
			c, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "collection"), testPublicCollectionSchema(), NewCollectionOptions())
			require.NoError(t, err)
			defer c.Close()
			_, err = c.Insert(ctx, []Document{testPublicDocument("a", "original", "old", 1, 1, []float32{1, 0})})
			require.NoError(t, err)
			before := c.indexBuildCount
			results, err := c.writeDocuments(ctx, operator, []Document{
				{PrimaryKey: "a", Fields: map[string]any{"title": "updated"}},
				{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{1}}}, // Invalid intervening update must not overwrite a.
				{PrimaryKey: "missing", Fields: map[string]any{"title": "incomplete"}},
				{PrimaryKey: "a", Fields: map[string]any{"rating": int32(9)}},
			})
			var batchErr *BatchWriteError
			require.ErrorAs(t, err, &batchErr)
			require.Equal(t, 2, batchErr.Failed)
			require.NoError(t, results[0].Err)
			require.Error(t, results[1].Err)
			require.Error(t, results[2].Err)
			require.NoError(t, results[3].Err)
			require.Greater(t, results[3].DocID, results[0].DocID)
			require.Equal(t, before+1, c.indexBuildCount)
			docs, err := c.Fetch(ctx, []string{"a", "missing"}, Projection{IncludeVectors: true})
			require.NoError(t, err)
			require.Equal(t, "updated", docs[0].Fields["title"])
			require.Equal(t, int32(9), docs[0].Fields["rating"])
			require.Equal(t, VectorFP32{1, 0}, docs[0].Fields["embedding"])
			require.Nil(t, docs[1])
		})
	}
}

// Cancel deterministically during a write without depending on machine timing.
type writeBatchCancelContext struct {
	context.Context
	checks atomic.Int64
	cancel context.CancelFunc
}

func (c *writeBatchCancelContext) Err() error {
	if c.checks.Add(1) == 128 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestCollectionUpsertBatchCancellationPublishesCommittedPrefix(t *testing.T) {
	ctx := context.Background()
	c, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "collection"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer c.Close()
	docs := make([]Document, 128)
	for i := range docs {
		docs[i] = testPublicDocument(fmt.Sprint(i), "title", "all", 1, 1, []float32{1, 0})
	}
	parent, cancel := context.WithCancel(ctx)
	defer cancel()
	writeCtx := &writeBatchCancelContext{Context: parent, cancel: cancel}
	results, err := c.Upsert(writeCtx, docs)
	require.ErrorIs(t, err, context.Canceled)
	committed := 0
	for _, result := range results {
		if result.Err == nil {
			committed++
		} else {
			require.ErrorIs(t, result.Err, context.Canceled)
		}
	}
	require.Greater(t, committed, 0)
	require.Less(t, committed, len(docs))
	got, err := c.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: len(docs)})
	require.NoError(t, err)
	require.Len(t, got, committed)
	require.Equal(t, uint64(1), c.indexBuildCount)
}

func TestCollectionUpsertBatchSegmentCapacityPublishesCommittedPrefix(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("rollover",
		FieldSchema{Name: "vector", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)},
		FieldSchema{Name: "group", DataType: DataTypeString, Index: NewInvertIndexParams()},
	)
	schema.MaxDocsPerSegment = MinMaxDocsPerSegment
	c, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "collection"), schema, NewCollectionOptions())
	require.NoError(t, err)
	defer c.Close()
	docs := make([]Document, MinMaxDocsPerSegment+1)
	for i := range docs {
		docs[i] = Document{PrimaryKey: fmt.Sprint(i), Fields: map[string]any{"vector": VectorFP32{float32(i), 0}, "group": "all"}}
	}
	results, err := c.Upsert(ctx, docs)
	require.Error(t, err)
	require.Len(t, results, len(docs))
	for _, result := range results[:MinMaxDocsPerSegment] {
		require.NoError(t, result.Err)
	}
	require.ErrorIs(t, results[MinMaxDocsPerSegment].Err, ErrResourceExhausted)
	require.Equal(t, uint64(1), c.indexBuildCount)
	query := VectorQuery{Field: "vector", DenseVector: VectorFP32{1, 0}, TopK: 2, Filter: "group='all'"}
	got, err := c.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{fmt.Sprint(MinMaxDocsPerSegment - 1), fmt.Sprint(MinMaxDocsPerSegment - 2)}, documentKeys(got))
	// A full segment requires explicit Flush, as before this optimization.
	require.NoError(t, c.Flush(ctx))
	_, err = c.Upsert(ctx, docs[MinMaxDocsPerSegment:])
	require.NoError(t, err)
	got, err = c.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{fmt.Sprint(MinMaxDocsPerSegment), fmt.Sprint(MinMaxDocsPerSegment - 1)}, documentKeys(got))
}
