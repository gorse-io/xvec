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
	"io"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCollectionIteratorRejectsMaintenanceWhileOpen(t *testing.T) {
	testCases := []struct {
		name      string
		operation func(context.Context, *Collection) error
	}{
		{name: "add column", operation: func(ctx context.Context, collection *Collection) error {
			return collection.AddColumn(ctx, NewField("added", DataTypeInt32), "1", AddColumnOptions{})
		}},
		{name: "alter column", operation: func(ctx context.Context, collection *Collection) error {
			return collection.AlterColumn(ctx, "rating", "renamed", nil, AlterColumnOptions{})
		}},
		{name: "drop column", operation: func(ctx context.Context, collection *Collection) error {
			return collection.DropColumn(ctx, "rating")
		}},
		{name: "create index", operation: func(ctx context.Context, collection *Collection) error {
			return collection.CreateIndex(ctx, "embedding", NewHNSWIndexParams(MetricTypeIP), CreateIndexOptions{})
		}},
		{name: "drop index", operation: func(ctx context.Context, collection *Collection) error {
			return collection.DropIndex(ctx, "embedding")
		}},
		{name: "optimize", operation: func(ctx context.Context, collection *Collection) error {
			return collection.Optimize(ctx, OptimizeOptions{})
		}},
		{name: "destroy", operation: func(ctx context.Context, collection *Collection) error {
			return collection.Destroy(ctx)
		}},
		{name: "close", operation: func(_ context.Context, collection *Collection) error {
			return collection.Close()
		}},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := context.Background()
			collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "iterator"), testPublicCollectionSchema(), NewCollectionOptions())
			require.NoError(t, err)
			iterator, err := collection.CreateIterator(ctx, NewIteratorOptions())
			require.NoError(t, err)

			err = testCase.operation(ctx, collection)
			require.ErrorIs(t, err, ErrFailedPrecondition)
			iterator.Close()
			require.NoError(t, collection.Close())
		})
	}
}

func TestCollectionIteratorSnapshotProjection(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "iterator"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, collection.Close()) }()

	_, err = collection.Insert(ctx, []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("deleted", "deleted", "low", 2, 2, []float32{0, 1}),
	})
	require.NoError(t, err)
	_, err = collection.Update(ctx, []Document{{PrimaryKey: "a", Fields: map[string]any{"title": "current"}}})
	require.NoError(t, err)
	_, err = collection.Delete(ctx, []string{"deleted"})
	require.NoError(t, err)

	options := NewIteratorOptions()
	options.Projection = Projection{OutputFields: []string{"title"}}
	iterator, err := collection.CreateIterator(ctx, options)
	require.NoError(t, err)
	defer iterator.Close()

	require.NoError(t, collection.Flush(ctx))
	_, err = collection.Delete(ctx, []string{"a"})
	require.NoError(t, err)
	_, err = collection.Insert(ctx, []Document{testPublicDocument("late", "late", "high", 3, 3, []float32{1, 1})})
	require.NoError(t, err)

	document, err := iterator.Next()
	require.NoError(t, err)
	require.Equal(t, "a", document.PrimaryKey)
	require.Equal(t, map[string]any{"title": "current"}, document.Fields)
	_, err = iterator.Next()
	require.ErrorIs(t, err, io.EOF)
}

func TestCollectionIteratorEmptyProjection(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "iterator"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, collection.Close()) }()
	_, err = collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0})})
	require.NoError(t, err)

	options := NewIteratorOptions()
	options.Projection = Projection{OutputFields: []string{}}
	iterator, err := collection.CreateIterator(ctx, options)
	require.NoError(t, err)
	defer iterator.Close()
	document, err := iterator.Next()
	require.NoError(t, err)
	require.Equal(t, "a", document.PrimaryKey)
	require.Empty(t, document.Fields)
}

func TestCollectionIteratorValidatesInputs(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "iterator"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)

	var nilCollection *Collection
	_, err = nilCollection.CreateIterator(ctx, NewIteratorOptions())
	require.ErrorIs(t, err, ErrInvalidArgument)
	_, err = collection.CreateIterator(nil, NewIteratorOptions())
	require.ErrorIs(t, err, ErrInvalidArgument)

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = collection.CreateIterator(cancelled, NewIteratorOptions())
	require.ErrorIs(t, err, context.Canceled)

	invalid := NewIteratorOptions()
	invalid.Projection.OutputFields = []string{"missing"}
	_, err = collection.CreateIterator(ctx, invalid)
	require.ErrorIs(t, err, ErrInvalidArgument)

	require.NoError(t, collection.Close())
	_, err = collection.CreateIterator(ctx, NewIteratorOptions())
	require.ErrorIs(t, err, ErrFailedPrecondition)
}

func TestCollectionIteratorReleasesEachMaintenanceSlot(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "iterator"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	_, err = collection.Insert(ctx, []Document{testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0})})
	require.NoError(t, err)

	first, err := collection.CreateIterator(ctx, NewIteratorOptions())
	require.NoError(t, err)
	second, err := collection.CreateIterator(ctx, NewIteratorOptions())
	require.NoError(t, err)
	firstDocument, err := first.Next()
	require.NoError(t, err)
	secondDocument, err := second.Next()
	require.NoError(t, err)
	require.Equal(t, firstDocument, secondDocument)

	first.Close()
	require.ErrorIs(t, collection.Close(), ErrFailedPrecondition)
	second.Close()
	require.NoError(t, collection.Close())
}

func TestCollectionIteratorDefaults(t *testing.T) {
	ctx := context.Background()
	collection, err := CreateAndOpen(ctx, filepath.Join(t.TempDir(), "iterator"), testPublicCollectionSchema(), NewCollectionOptions())
	require.NoError(t, err)
	defer func() { require.NoError(t, collection.Close()) }()

	_, err = collection.Insert(ctx, []Document{
		testPublicDocument("a", "alpha", "low", 1, 1, []float32{1, 0}),
		testPublicDocument("b", "bravo", "high", 2, 2, []float32{0, 1}),
	})
	require.NoError(t, err)

	iterator, err := collection.CreateIterator(ctx, NewIteratorOptions())
	require.NoError(t, err)

	seen := make(map[string]bool)
	for {
		document, err := iterator.Next()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		seen[document.PrimaryKey] = true
		require.Contains(t, document.Fields, "title")
		require.Contains(t, document.Fields, "embedding")
	}
	require.Equal(t, map[string]bool{"a": true, "b": true}, seen)

	iterator.Close()
	iterator.Close()
	_, err = iterator.Next()
	require.ErrorIs(t, err, io.EOF)
}
