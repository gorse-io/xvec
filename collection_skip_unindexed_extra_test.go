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
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/gorse-io/xvec/internal/db/index/common"
	"github.com/stretchr/testify/require"
)

func skipTestCollection(t *testing.T, fields ...FieldSchema) *Collection {
	t.Helper()
	opts := NewCollectionOptions()
	opts.SkipUnindexedSegments = true
	c, err := CreateAndOpen(context.Background(), filepath.Join(t.TempDir(), "collection"), NewCollectionSchema("skip", fields...), opts)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, c.Close()) })
	return c
}

func TestSkipUnindexedSegmentsSparseFilterAndFTS(t *testing.T) {
	ctx := context.Background()
	hnsw := NewHNSWIndexParams(MetricTypeIP)
	hnsw.M, hnsw.EFConstruction = 4, 16
	c := skipTestCollection(t, FieldSchema{Name: "s", DataType: DataTypeSparseVectorFP32, Index: hnsw}, FieldSchema{Name: "text", DataType: DataTypeString, Index: NewFTSIndexParams()}, FieldSchema{Name: "group", DataType: DataTypeString})
	vector := SparseVectorFP32{Indices: []uint32{1}, Values: []float32{1}}
	doc := func(key string) Document {
		return Document{PrimaryKey: key, Fields: map[string]any{"s": vector, "text": "apple", "group": key}}
	}
	_, err := c.Insert(ctx, []Document{doc("indexed")})
	require.NoError(t, err)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
	_, err = c.Insert(ctx, []Document{doc("tail")})
	require.NoError(t, err)
	params := NewHNSWQueryParams()
	params.Linear = true
	results, err := c.Query(ctx, VectorQuery{Field: "s", SparseVector: vector, TopK: 10, Params: params, Filter: "group != 'absent'", Projection: Projection{IncludeVectors: true}})
	require.NoError(t, err)
	require.Equal(t, []string{"indexed"}, documentKeys(results))
	require.Equal(t, vector, results[0].Fields["s"])
	results, err = c.Query(ctx, VectorQuery{Field: "s", PrimaryKey: "tail", TopK: 10})
	require.NoError(t, err)
	require.Equal(t, []string{"indexed"}, documentKeys(results))
	groups, err := c.GroupByQuery(ctx, GroupByVectorQuery{Field: "s", SparseVector: vector, GroupByField: "group", GroupCount: 10, TopKPerGroup: 1, Params: params, Filter: "group != 'absent'"})
	require.NoError(t, err)
	require.Len(t, groups, 1)
	results, err = c.Query(ctx, VectorQuery{Field: "text", FTS: &FTSClause{Match: "apple"}, TopK: 10})
	require.NoError(t, err)
	require.Len(t, results, 2)
	fetched, err := c.Fetch(ctx, []string{"tail"}, Projection{IncludeVectors: true})
	require.NoError(t, err)
	require.NotNil(t, fetched[0])
	require.Equal(t, uint64(2), c.Stats().DocumentCount)
	_, err = c.Delete(ctx, []string{"indexed"})
	require.NoError(t, err)
	results, err = c.Query(ctx, VectorQuery{Field: "s", SparseVector: vector, TopK: 10})
	require.NoError(t, err)
	require.Empty(t, results)
	require.NoError(t, c.Flush(ctx))
	results, err = c.Query(ctx, VectorQuery{Field: "s", SparseVector: vector, TopK: 10})
	require.NoError(t, err)
	require.Empty(t, results, "flushed but unbuilt immutable tail must remain excluded")
}

func TestSkipUnindexedSegmentsValidation(t *testing.T) {
	ctx := context.Background()
	c := skipTestCollection(t, FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)}, FieldSchema{Name: "g", DataType: DataTypeString})
	for name, query := range map[string]VectorQuery{
		"dimension":  {Field: "v", DenseVector: VectorFP32{1}, TopK: 1},
		"nan":        {Field: "v", DenseVector: VectorFP32{float32(math.NaN()), 1}, TopK: 1},
		"type":       {Field: "v", SparseVector: SparseVectorFP32{}, TopK: 1},
		"params":     {Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 1, Params: NewIVFQueryParams()},
		"filter":     {Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 1, Filter: "unknown = 1"},
		"projection": {Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 1, Projection: Projection{OutputFields: []string{"unknown"}}},
		"primary":    {Field: "v", PrimaryKey: "missing", TopK: 1},
	} {
		t.Run(name, func(t *testing.T) { _, err := c.Query(ctx, query); require.Error(t, err) })
	}
	_, err := c.GroupByQuery(ctx, GroupByVectorQuery{Field: "v", DenseVector: VectorFP32{1}, GroupByField: "g", GroupCount: 1, TopKPerGroup: 1})
	require.Error(t, err)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = c.Query(cancelled, VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 1})
	require.Error(t, err)
	ivf := skipTestCollection(t, FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewIVFIndexParams(MetricTypeIP)}, FieldSchema{Name: "g", DataType: DataTypeString})
	_, err = ivf.GroupByQuery(ctx, GroupByVectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, GroupByField: "g", GroupCount: 1, TopKPerGroup: 1})
	require.Error(t, err, "unsupported group params must be validated without candidates")
}

func TestSkipUnindexedSegmentsFieldMetadataEligibility(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("skip", FieldSchema{Name: "a", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)}, FieldSchema{Name: "b", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)})
	path := filepath.Join(t.TempDir(), "collection")
	c, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	_, err = c.Insert(ctx, []Document{{PrimaryKey: "one", Fields: map[string]any{"a": VectorFP32{1, 0}, "b": VectorFP32{1, 0}}}})
	require.NoError(t, err)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
	require.NoError(t, c.Close())
	versions, err := common.OpenVersionManager(ctx, path)
	require.NoError(t, err)
	_, err = versions.Update(ctx, func(m *common.Manifest) error {
		artifacts := m.SegmentIndexSnapshots[0].Artifacts
		kept := artifacts[:0]
		for _, artifact := range artifacts {
			if artifact.Field != "b" {
				kept = append(kept, artifact)
			}
		}
		m.SegmentIndexSnapshots[0].Artifacts = kept
		return nil
	})
	require.NoError(t, err)
	opts := NewCollectionOptions()
	opts.SkipUnindexedSegments = true
	c, err = Open(ctx, path, opts)
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	results, err := c.Query(ctx, VectorQuery{Field: "a", DenseVector: VectorFP32{1, 0}, TopK: 10})
	require.NoError(t, err)
	require.Len(t, results, 1)
	results, err = c.Query(ctx, VectorQuery{Field: "b", DenseVector: VectorFP32{1, 0}, TopK: 10})
	require.NoError(t, err)
	require.Empty(t, results)
	require.Len(t, c.vectorQuerySnapshots, 2)
}

func TestSkipUnindexedSegmentsPublishedCorruption(t *testing.T) {
	ctx := context.Background()
	c := skipTestCollection(t, FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)})
	_, err := c.Insert(ctx, []Document{{PrimaryKey: "one", Fields: map[string]any{"v": VectorFP32{1, 0}}}})
	require.NoError(t, err)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
	path := c.path
	artifact := c.store.Manifest().SegmentIndexSnapshots[0].Artifacts[0].File
	require.NoError(t, c.Close())
	require.NoError(t, os.WriteFile(filepath.Join(path, filepath.FromSlash(artifact)), []byte("corrupt"), 0600))
	opts := NewCollectionOptions()
	opts.SkipUnindexedSegments = true
	reopened, err := Open(ctx, path, opts)
	require.Error(t, err)
	if reopened != nil {
		require.NoError(t, reopened.Close())
	}
}

func TestSkipUnindexedSegmentsConcurrentQueriesAndWrites(t *testing.T) {
	ctx := context.Background()
	c := skipTestCollection(t, FieldSchema{Name: "v", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)})
	_, err := c.Insert(ctx, []Document{{PrimaryKey: "indexed", Fields: map[string]any{"v": VectorFP32{1, 0}}}})
	require.NoError(t, err)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
	var wg sync.WaitGroup
	errors := make(chan error, 32)
	for worker := 0; worker < 3; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				_, err := c.Query(ctx, VectorQuery{Field: "v", DenseVector: VectorFP32{1, 0}, TopK: 10})
				if err != nil {
					errors <- err
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			_, err := c.Upsert(ctx, []Document{{PrimaryKey: fmt.Sprint(i), Fields: map[string]any{"v": VectorFP32{2, 0}}}})
			if err != nil {
				errors <- err
			}
		}
	}()
	wg.Wait()
	close(errors)
	for err := range errors {
		require.NoError(t, err)
	}
}
