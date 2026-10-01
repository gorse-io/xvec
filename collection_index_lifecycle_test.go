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
	"sync"
	"testing"
	"time"

	core "github.com/gorse-io/xvec/internal/core/algorithm"
	"github.com/stretchr/testify/require"
)

func TestCollectionANNLifecycleUsesFlatUntilMaintenance(t *testing.T) {
	ctx := context.Background()
	schema := testPublicCollectionSchema()
	schema.Fields[3].Index = NewHNSWIndexParams(MetricTypeIP)
	schema.Fields[4].Index = NewHNSWIndexParams(MetricTypeIP)
	path := filepath.Join(t.TempDir(), "lifecycle")
	collection, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, collection.Close()) }()
	_, err = collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "one", 1, 1, []float32{1, 0})})
	require.NoError(t, err)
	var first *collectionSegmentRuntime
	for _, r := range collection.segmentIndexes {
		first = r
	}
	require.NotNil(t, first)
	raw := first.indexes.denseNative["embedding"]
	require.IsType(t, &core.DenseFlatIndex{}, raw)
	require.IsType(t, &core.SparseFlatIndex{}, first.indexes.sparseNative["sparse"])
	require.Zero(t, collection.Stats().IndexCompleteness["embedding"])

	// Append incrementally; a previously acquired query snapshot must not see it.
	snapshot, release, err := collection.acquireQuerySnapshotLocked(ctx)
	require.NoError(t, err)
	defer release()
	_, err = collection.Insert(ctx, []Document{testPublicDocument("b", "bravo", "two", 2, 2, []float32{2, 0})})
	require.NoError(t, err)
	for _, r := range collection.segmentIndexes {
		require.Same(t, raw, r.indexes.denseNative["embedding"])
	}
	field, _ := schema.Field("embedding")
	oldResults, err := collection.searchVectorSegments(ctx, "test", field, VectorFP32{1, 0}, nil, 10, nil, snapshot.segments, snapshot.runtimes, snapshot.liveFilter)
	require.NoError(t, err)
	require.Equal(t, []core.Result{{Key: snapshot.documents[0].DocID, Score: 1}}, oldResults)
	release()

	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10}
	checkQuery := func(want []string) {
		t.Helper()
		before := collection.indexBuildCount
		docs, err := collection.Query(ctx, query)
		require.NoError(t, err)
		require.Equal(t, want, documentKeys(docs))
		require.Equal(t, before, collection.indexBuildCount, "query initialized indexes")
		_, err = collection.MultiQuery(ctx, MultiQuery{TopK: 10, Queries: []SubQuery{{Field: "embedding", DenseVector: VectorFP32{1, 0}}, {Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}}}})
		require.NoError(t, err)
		require.Equal(t, before, collection.indexBuildCount)
	}
	checkQuery([]string{"b", "a"})
	require.NoError(t, collection.Flush(ctx))
	require.Empty(t, collection.store.Manifest().SegmentIndexSnapshots, "Flush built ANN artifacts")
	checkQuery([]string{"b", "a"})
	require.NoError(t, collection.Close())
	collection, err = Open(ctx, path, NewCollectionOptions())
	require.NoError(t, err)
	for _, r := range collection.segmentIndexes {
		require.True(t, r.indexes.writerFlat["embedding"])
	}
	checkQuery([]string{"b", "a"})
	require.NoError(t, collection.Optimize(ctx, OptimizeOptions{}))
	require.Equal(t, float32(1), collection.Stats().IndexCompleteness["embedding"])
	for _, r := range collection.segmentIndexes {
		require.IsType(t, &core.HNSWIndex{}, r.indexes.denseNative["embedding"])
	}
	checkQuery([]string{"b", "a"})
	_, err = collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{3, 0}, "sparse": SparseVectorFP32{Indices: []uint32{2}, Values: []float32{3}}}}})
	require.NoError(t, err)
	require.Equal(t, float32(.5), collection.Stats().IndexCompleteness["embedding"])
	checkQuery([]string{"a", "b"})
	_, err = collection.Delete(ctx, []string{"b"})
	require.NoError(t, err)
	checkQuery([]string{"a"})
	require.NoError(t, collection.CreateIndex(ctx, "embedding", schema.Fields[3].Index, CreateIndexOptions{}))
	require.Equal(t, float32(1), collection.Stats().IndexCompleteness["embedding"])
	require.Zero(t, collection.Stats().IndexCompleteness["sparse"], "CreateIndex built an unrelated vector field")
	checkQuery([]string{"a"})
	require.NoError(t, collection.Close())
	collection, err = Open(ctx, path, CollectionOptions{ReadOnly: true, EnableMmap: true})
	require.NoError(t, err)
	checkQuery([]string{"a"})
}

func TestCollectionUnoptimizedANNQueryDoesNotTrain(t *testing.T) {
	ctx := context.Background()
	for _, index := range []IndexParams{NewHNSWIndexParams(MetricTypeL2), NewIVFIndexParams(MetricTypeL2), NewVamanaIndexParams(MetricTypeL2), NewDiskANNIndexParams(MetricTypeL2), NewHNSWRaBitQIndexParams(MetricTypeL2), NewIVFRaBitQIndexParams(MetricTypeL2)} {
		t.Run(index.IndexType().String(), func(t *testing.T) {
			schema := NewCollectionSchema("ann", FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 64, Index: index}, FieldSchema{Name: "group", DataType: DataTypeString})
			path := filepath.Join(t.TempDir(), "ann")
			c, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
			require.NoError(t, err)
			_, err = c.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"embedding": lifecycleDenseVector(1), "group": "all"}}, {PrimaryKey: "b", Fields: map[string]any{"embedding": lifecycleDenseVector(2), "group": "all"}}})
			require.NoError(t, err)
			before := c.indexBuildCount
			docs, err := c.Query(ctx, VectorQuery{Field: "embedding", DenseVector: lifecycleDenseVector(1), TopK: 2})
			require.NoError(t, err)
			require.Equal(t, []string{"a", "b"}, documentKeys(docs))
			require.Equal(t, before, c.indexBuildCount)
			groups, err := c.GroupByQuery(ctx, GroupByVectorQuery{Field: "embedding", DenseVector: lifecycleDenseVector(1), GroupByField: "group", GroupCount: 1, TopKPerGroup: 2})
			require.NoError(t, err)
			require.Len(t, groups, 1)
			require.Equal(t, before, c.indexBuildCount)
			require.NoError(t, c.Close()) // WAL recovery without Flush also prepares Flat at Open.
			c, err = Open(ctx, path, NewCollectionOptions())
			require.NoError(t, err)
			before = c.indexBuildCount
			_, err = c.Query(ctx, VectorQuery{Field: "embedding", DenseVector: lifecycleDenseVector(1), TopK: 2})
			require.NoError(t, err)
			require.Equal(t, before, c.indexBuildCount)
			require.NoError(t, c.Close())
		})
	}
}

func lifecycleDenseVector(first float32) VectorFP32 {
	vector := make(VectorFP32, 64)
	vector[0] = first
	return vector
}

// Unit tests of native searchers explicitly perform maintenance first.
func buildIndexedCollectionRuntimeForTest(t *testing.T, ctx context.Context, schema CollectionSchema, documents []Document) (*collectionRuntimeIndexes, error) {
	t.Helper()
	native, err := buildCollectionArtifactIndexes(ctx, schema, documents, 2, 0)
	if err != nil {
		return nil, err
	}
	c := &Collection{path: t.TempDir(), schema: schema}
	artifacts, _, err := c.writeSegmentRuntimeArtifacts(ctx, 1, native)
	closeErr := native.Close()
	if err != nil {
		return nil, err
	}
	if closeErr != nil {
		return nil, closeErr
	}
	paths := make(map[string]string)
	for _, artifact := range artifacts {
		paths[collectionIndexArtifactKey(artifact.Field, artifact.Kind)] = filepath.Join(c.path, filepath.FromSlash(artifact.File))
	}
	return buildCollectionRuntimeIndexes(ctx, schema, documents, 2, 0, false, paths)
}

func TestCollectionNativeBuildAllowsQueriesAndWrites(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("concurrent_build", FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)})
	c, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "build"), schema, NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	_, err = c.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{1, 0}}}})
	require.NoError(t, err)
	require.NoError(t, c.Flush(ctx))
	entered, resume, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	releaseBuild := func() { releaseOnce.Do(func() { close(resume) }) }
	defer func() { releaseBuild(); <-finished }()
	built := make(chan error, 1)
	go func() {
		defer close(finished)
		c.maintenanceMu.Lock()
		defer c.maintenanceMu.Unlock()
		built <- c.buildAndPublishNativeIndexes(ctx, 1, func(ctx context.Context, schema CollectionSchema, documents []Document, workers int, maxBufferSize uint32) (*collectionRuntimeIndexes, error) {
			close(entered)
			<-resume
			return buildCollectionArtifactIndexes(ctx, schema, documents, workers, maxBufferSize)
		})
	}()
	<-entered
	// Keep construction blocked until both operations have finished. Success
	// cannot depend on the scheduler completing the ANN build quickly enough.
	query := VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 10}
	queried := make(chan error, 1)
	go func() { _, err := c.Query(ctx, query); queried <- err }()
	select {
	case err := <-queried:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("query waited for ANN construction")
	}
	written := make(chan error, 1)
	go func() {
		_, err := c.Insert(ctx, []Document{{PrimaryKey: "b", Fields: map[string]any{"embedding": VectorFP32{2, 0}}}})
		written <- err
	}()
	select {
	case err := <-written:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("write waited for ANN construction")
	}
	releaseBuild()
	require.NoError(t, <-built)
	require.Equal(t, float32(.5), c.Stats().IndexCompleteness["embedding"])
	count := c.indexBuildCount
	docs, err := c.Query(ctx, query)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(docs))
	require.Equal(t, count, c.indexBuildCount)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
	require.Equal(t, float32(1), c.Stats().IndexCompleteness["embedding"])
}

func TestCollectionFailedNativeBuildKeepsFlatQueryable(t *testing.T) {
	ctx := context.Background()
	schema := NewCollectionSchema("failed_build", FieldSchema{Name: "embedding", DataType: DataTypeVectorFP32, Dimension: 2, Index: NewHNSWIndexParams(MetricTypeIP)})
	c, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "build"), schema, NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	_, err = c.Insert(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"embedding": VectorFP32{1, 0}}}})
	require.NoError(t, err)
	require.NoError(t, c.Flush(ctx))
	before := c.store.Manifest()
	c.maintenanceMu.Lock()
	err = c.buildAndPublishNativeIndexes(ctx, 1, func(context.Context, CollectionSchema, []Document, int, uint32) (*collectionRuntimeIndexes, error) {
		return nil, context.Canceled
	})
	c.maintenanceMu.Unlock()
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, before, c.store.Manifest())
	count := c.indexBuildCount
	docs, err := c.Query(ctx, VectorQuery{Field: "embedding", DenseVector: VectorFP32{1, 0}, TopK: 1})
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, documentKeys(docs))
	require.Equal(t, count, c.indexBuildCount)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
}

func TestCollectionIndexDDLRetainsUnchangedNativeArtifacts(t *testing.T) {
	ctx := context.Background()
	schema := testPublicCollectionSchema()
	schema.Fields[3].Index = NewHNSWIndexParams(MetricTypeIP)
	schema.Fields[4].Index = NewHNSWIndexParams(MetricTypeIP)
	path := filepath.Join(t.TempDir(), "ddl")
	c, err := CreateAndOpen(ctx, path, schema, NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, c.Close()) }()
	_, err = c.Insert(ctx, []Document{testPublicDocument("a", "alpha", "one", 1, 1, []float32{1, 0}), testPublicDocument("b", "bravo", "two", 2, 2, []float32{2, 0})})
	require.NoError(t, err)
	require.NoError(t, c.Optimize(ctx, OptimizeOptions{}))
	sparseArtifact := ""
	for _, snapshot := range c.store.Manifest().SegmentIndexSnapshots {
		for _, artifact := range snapshot.Artifacts {
			if artifact.Field == "sparse" {
				sparseArtifact = artifact.File
			}
		}
	}
	require.NotEmpty(t, sparseArtifact)
	changed := NewHNSWIndexParams(MetricTypeIP)
	changed.M, changed.EFConstruction = 4, 20
	require.NoError(t, c.CreateIndex(ctx, "embedding", changed, CreateIndexOptions{}))
	require.Equal(t, float32(1), c.Stats().IndexCompleteness["sparse"])
	require.NoError(t, c.CreateIndex(ctx, "rating", NewInvertIndexParams(), CreateIndexOptions{}))
	require.Equal(t, float32(1), c.Stats().IndexCompleteness["embedding"])
	require.NoError(t, c.DropIndex(ctx, "embedding"))
	require.Equal(t, float32(1), c.Stats().IndexCompleteness["sparse"])
	found := false
	for _, snapshot := range c.store.Manifest().SegmentIndexSnapshots {
		for _, artifact := range snapshot.Artifacts {
			if artifact.Field == "sparse" {
				found = true
				require.Equal(t, sparseArtifact, artifact.File)
			}
		}
	}
	require.True(t, found, "DDL discarded the unrelated sparse graph")
	require.NoError(t, c.Close())
	c, err = Open(ctx, path, CollectionOptions{ReadOnly: true})
	require.NoError(t, err)
	for _, segment := range c.segmentIndexes {
		require.IsType(t, &core.SparseHNSWIndex{}, segment.indexes.sparseNative["sparse"])
	}
	before := c.indexBuildCount
	docs, err := c.Query(ctx, VectorQuery{Field: "sparse", SparseVector: SparseVectorFP32{Indices: []uint32{2}, Values: []float32{1}}, Filter: "rating >= 1", TopK: 2})
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, documentKeys(docs))
	require.Equal(t, before, c.indexBuildCount)
}
