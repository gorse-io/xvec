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

package db

import (
	"context"
	"testing"

	"github.com/gorse-io/xvec/internal/db/index/common"
	"github.com/stretchr/testify/require"
)

func TestVisitSelectedSegmentSnapshotsSkipsPayloadVisit(t *testing.T) {
	ctx := context.Background()
	store, err := CreateCollection(ctx, t.TempDir(), testCollectionSchema, CollectionOptions{SegmentMaxDocuments: 2})
	require.NoError(t, err)
	defer func() { require.NoError(t, store.Close()) }()
	_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "sealed", Payload: []byte("payload")}})
	require.NoError(t, err)
	require.NoError(t, store.Flush(ctx))
	_, err = store.Insert(ctx, []WriteInput{{PrimaryKey: "writing", Payload: []byte("tail")}})
	require.NoError(t, err)
	immutable := store.manager.ImmutableSegments()[0]
	// Closing an excluded segment makes any attempted VisitDocuments return an
	// error. Metadata remains available, so selection can still skip it safely.
	require.NoError(t, immutable.Close())
	var selected, visited int
	require.NoError(t, store.VisitSelectedSegmentSnapshots(ctx, func(metadata common.SegmentMetadata, mutable bool) bool {
		selected++
		return mutable
	}, func(snapshot SegmentSnapshot) error {
		visited++
		require.True(t, snapshot.Mutable)
		require.Equal(t, "writing", snapshot.Documents[0].PrimaryKey)
		return nil
	}))
	require.Equal(t, 2, selected)
	require.Equal(t, 1, visited)
	require.Error(t, store.VisitSegmentSnapshots(ctx, func(SegmentSnapshot) error { return nil }), "the closed-segment trap must detect payload visits")
	require.NoError(t, store.VisitSelectedSegmentSnapshots(ctx, func(common.SegmentMetadata, bool) bool { return false }, func(SegmentSnapshot) error { t.Fatal("excluded payload visited"); return nil }))
	require.Error(t, store.VisitSelectedSegmentSnapshots(nil, nil, func(SegmentSnapshot) error { return nil }))
	require.Error(t, store.VisitSelectedSegmentSnapshots(ctx, nil, nil))
}
