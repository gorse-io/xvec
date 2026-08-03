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
	"errors"
	"reflect"
	"testing"
)

func TestSegmentManagerLifecycleAndLookup(t *testing.T) {
	primary := NewPrimaryKeyMap()
	deletes := NewDeleteStore()
	manager := NewSegmentManager(primary, deletes)
	late := testImmutableSegment(2, 10, "ten", "eleven")
	early := testImmutableSegment(1, 0, "zero", "one")
	if err := manager.AddImmutable(late); err != nil {
		t.Fatal(err)
	}
	if err := manager.AddImmutable(early); err != nil {
		t.Fatal(err)
	}
	if got := manager.ImmutableMetadata(); len(got) != 2 || got[0].ID != 1 || got[1].ID != 2 {
		t.Fatalf("sorted metadata = %#v", got)
	}
	if err := manager.AddImmutable(early); err == nil {
		t.Fatal("duplicate segment succeeded")
	}
	overlap := testImmutableSegment(9, 1, "overlap")
	if err := manager.AddImmutable(overlap); err == nil {
		t.Fatal("overlapping segment succeeded")
	}

	writing, _ := NewWriteSegment(3, 20, 5)
	if err := manager.SetWriting(writing); err != nil {
		t.Fatal(err)
	}
	written, err := writing.Append(context.Background(), "twenty", []byte("live"))
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.SetWriting(writing); err == nil {
		t.Fatal("second writing segment succeeded")
	}
	reservedOverlap := testImmutableSegment(4, 24, "reserved")
	if err := manager.AddImmutable(reservedOverlap); err == nil {
		t.Fatal("reserved write range overlap succeeded")
	}

	_, _, _ = primary.Put(context.Background(), "zero", DocumentLocation{SegmentID: 1, DocID: 0})
	_, _, _ = primary.Put(context.Background(), "twenty", DocumentLocation{SegmentID: 3, DocID: written.DocID})
	if doc, found := manager.DocumentByPrimaryKey("zero"); !found || doc.DocID != 0 {
		t.Fatalf("immutable lookup = %#v, %v", doc, found)
	}
	if doc, found := manager.DocumentByPrimaryKey("twenty"); !found || string(doc.Payload) != "live" {
		t.Fatalf("writing lookup = %#v, %v", doc, found)
	}
	_, _ = deletes.MarkDeleted(context.Background(), 0)
	if _, found := manager.Document(0); found {
		t.Fatal("deleted document is visible by ID")
	}
	if _, found := manager.DocumentByPrimaryKey("zero"); found {
		t.Fatal("deleted document is visible by key")
	}

	if cleared := manager.ClearWriting(); cleared != writing || manager.Writing() != nil {
		t.Fatal("clear writing returned wrong segment")
	}
	removed, err := manager.RemoveImmutable(2)
	if err != nil || removed != late {
		t.Fatalf("remove = %p, %v", removed, err)
	}
	if _, err := manager.RemoveImmutable(2); !errors.Is(err, ErrSegmentNotFound) {
		t.Fatalf("second remove = %v", err)
	}
	if manager.PrimaryKeys() != primary || manager.Deletes() != deletes {
		t.Fatal("manager replaced stores")
	}
}

func TestSegmentManagerRejectsWritingReservedOverlap(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	if err := manager.AddImmutable(testImmutableSegment(1, 10, "ten", "eleven")); err != nil {
		t.Fatal(err)
	}
	writing, _ := NewWriteSegment(2, 9, 2)
	if err := manager.SetWriting(writing); err == nil {
		t.Fatal("overlapping empty write segment succeeded")
	}
}

func TestSegmentManagerValidatesStalePrimaryLocation(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	segment := testImmutableSegment(1, 0, "zero")
	if err := manager.AddImmutable(segment); err != nil {
		t.Fatal(err)
	}
	_, _, _ = manager.PrimaryKeys().Put(context.Background(), "wrong", DocumentLocation{SegmentID: 1, DocID: 0})
	if _, found := manager.DocumentByPrimaryKey("wrong"); found {
		t.Fatal("stale primary-key location returned another document")
	}
	_, _, _ = manager.PrimaryKeys().Put(context.Background(), "zero", DocumentLocation{SegmentID: 99, DocID: 0})
	if _, found := manager.DocumentByPrimaryKey("zero"); found {
		t.Fatal("missing segment location returned a document")
	}
}

func TestNewSegmentManagerCreatesStores(t *testing.T) {
	manager := NewSegmentManager(nil, nil)
	if manager.PrimaryKeys() == nil || manager.Deletes() == nil {
		t.Fatal("default stores are nil")
	}
	if !reflect.DeepEqual(manager.ImmutableMetadata(), []SegmentMetadata{}) && manager.ImmutableMetadata() != nil {
		t.Fatalf("initial metadata = %#v", manager.ImmutableMetadata())
	}
}

func testImmutableSegment(id, minDocID uint64, keys ...string) *ImmutableSegment {
	docs := make([]StoredDocument, len(keys))
	for index, key := range keys {
		docs[index] = StoredDocument{DocID: minDocID + uint64(index), PrimaryKey: key, Payload: []byte(key)}
	}
	metadata := segmentMetadata(id, docs, []string{"segment.seg"})
	return newImmutableSegment(metadata, docs)
}
