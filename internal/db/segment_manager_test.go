// Copyright 2026-present the zvec-go project
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

	"github.com/stretchr/testify/require"
)

func TestSegmentManagerLifecycleAndLookup(t *testing.T) {
	primary := NewPrimaryKeyMap()
	deletes := NewDeleteStore()
	manager := NewSegmentManager(primary, deletes)
	late := testImmutableSegment(2, 10, "ten", "eleven")
	early := testImmutableSegment(1, 0, "zero", "one")
	{
		err := manager.AddImmutable(late)
		require.NoError(t, err)
	}
	{
		err := manager.AddImmutable(early)
		require.NoError(t, err)
	}
	{
		got := manager.ImmutableMetadata()
		require.Len(t, got, 2)
		require.True(t, got[0].ID == 1)
		require.True(t, got[1].ID == 2)
	}
	{
		err := manager.AddImmutable(early)
		require.Error(t, err,
			"duplicate segment succeeded")
	}

	overlap := testImmutableSegment(9, 1, "overlap")
	{
		err := manager.AddImmutable(overlap)
		require.Error(t, err,
			"overlapping segment succeeded")
	}

	writing, _ := NewWriteSegment(3, 20, 5)
	{
		err := manager.SetWriting(writing)
		require.NoError(t, err)
	}

	written, err := writing.Append(context.Background(), "twenty", []byte("live"))
	require.NoError(t, err)
	{
		err := manager.SetWriting(writing)
		require.Error(t, err,
			"second writing segment succeeded")
	}

	reservedOverlap := testImmutableSegment(4, 24, "reserved")
	{
		err := manager.AddImmutable(reservedOverlap)
		require.Error(t, err,
			"reserved write range overlap succeeded")
	}

	_, _, _ = primary.Put(context.Background(), "zero", 0)
	_, _, _ = primary.Put(context.Background(), "twenty", written.DocID)
	{
		doc, found, err := manager.DocumentByPrimaryKey("zero")
		require.NoError(t, err)
		require.True(t, found)
		require.True(t, doc.DocID == 0)
	}
	{
		doc, found, err := manager.DocumentByPrimaryKey("twenty")
		require.NoError(t, err)
		require.True(t, found)
		require.True(t, string(doc.Payload) == "live")
	}

	_, _ = deletes.MarkDeleted(context.Background(), 0)
	{
		_, found := manager.Document(0)
		require.False(t, found,
			"deleted document is visible by ID")
	}
	{
		_, found, err := manager.DocumentByPrimaryKey("zero")
		require.NoError(t, err)
		require.False(t, found,
			"deleted document is visible by key")
	}
	{
		cleared := manager.ClearWriting()
		require.Same(t, writing, cleared,
			"clear writing returned wrong segment")
		require.Nil(t, manager.Writing(),
			"clear writing returned wrong segment")
	}

	removed, err := manager.RemoveImmutable(2)
	require.NoError(t, err)
	require.Same(t, late, removed)
	{
		_, err := manager.RemoveImmutable(2)
		require.ErrorIs(t, err, ErrSegmentNotFound)
	}
	require.Same(t, primary, manager.PrimaryKeys(),
		"manager replaced stores")
	require.Same(t, deletes, manager.Deletes(),
		"manager replaced stores")
}

func TestSegmentManagerRejectsWritingReservedOverlap(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	{
		err := manager.AddImmutable(testImmutableSegment(1, 10, "ten", "eleven"))
		require.NoError(t, err)
	}

	writing, _ := NewWriteSegment(2, 9, 2)
	{
		err := manager.SetWriting(writing)
		require.Error(t, err,
			"overlapping empty write segment succeeded")
	}
}

func TestSegmentManagerRejectsEmptyImmutableSegment(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	require.Error(t, manager.AddImmutable(testImmutableSegment(1, 0)))
}

func TestSegmentManagerValidatesStalePrimaryLocation(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	segment := testImmutableSegment(1, 0, "zero")
	{
		err := manager.AddImmutable(segment)
		require.NoError(t, err)
	}

	_, _, _ = manager.PrimaryKeys().Put(context.Background(), "wrong", 0)
	{
		_, found, err := manager.DocumentByPrimaryKey("wrong")
		require.NoError(t, err)
		require.False(t, found,
			"stale primary-key location returned another document")
	}

	_, _, _ = manager.PrimaryKeys().Put(context.Background(), "zero", 99)
	{
		_, found, err := manager.DocumentByPrimaryKey("zero")
		require.NoError(t, err)
		require.False(t, found,
			"missing segment location returned a document")
	}
}

func TestSegmentManagerFetch(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	segment := testImmutableSegment(1, 5, "five", "six")
	{
		err := manager.AddImmutable(segment)
		require.NoError(t, err)
	}

	_, _, _ = manager.PrimaryKeys().Put(context.Background(), "five", 5)
	_, _, _ = manager.PrimaryKeys().Put(context.Background(), "six", 6)
	_, _ = manager.Deletes().MarkDeleted(context.Background(), 6)
	results, err := manager.Fetch(context.Background(), []string{"missing", "five", "six", "five"})
	require.NoError(t, err)
	require.Len(t, results, 4)
	require.Nil(t, results[0].Document)
	require.NotNil(t, results[1].Document)
	require.Nil(t, results[2].Document)
	require.NotNil(t, results[3].Document)
	require.True(t, results[1].Document.DocID == 5)
	require.True(t, string(results[1].Document.Payload) == "five")

	results[1].Document.Payload[0] = 'X'
	again, err := manager.Fetch(context.Background(), []string{"five"})
	require.NoError(t, err)
	require.True(t, string(again[0].Document.Payload) == "five")

	empty, err := manager.Fetch(context.Background(), nil)
	require.NoError(t, err)
	require.NotNil(t, empty)
	require.Len(t, empty, 0)
}

func TestSegmentManagerFetchCancellation(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	results, err := manager.Fetch(canceled, []string{"one", "two"})
	require.ErrorIs(t, err, context.Canceled)
	require.Len(t, results, 2)
	require.ErrorIs(t, results[0].Err, context.Canceled)
	require.ErrorIs(t, results[1].Err, context.Canceled)
	{
		_, err := manager.Fetch(nil, []string{"one"})
		require.Error(t, err,
			"nil fetch context succeeded")
	}

	var nilManager *SegmentManager
	{
		_, err := nilManager.Fetch(context.Background(), nil)
		require.Error(t, err,
			"nil manager fetch succeeded")
	}
}

func TestNewSegmentManagerCreatesStores(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	require.NotNil(t, manager.PrimaryKeys(),
		"default stores are nil")
	require.NotNil(t, manager.Deletes(),
		"default stores are nil")
	require.Empty(t, manager.ImmutableMetadata())
}

func testImmutableSegment(id, minDocID uint64, keys ...string) *ImmutableSegment {
	docs := make([]StoredDocument, len(keys))
	for index, key := range keys {
		docs[index] = StoredDocument{DocID: minDocID + uint64(index), PrimaryKey: key, Payload: []byte(key)}
	}
	metadata := segmentMetadata(id, docs, []string{"segment.seg"})
	return newImmutableSegment(metadata, docs)
}
