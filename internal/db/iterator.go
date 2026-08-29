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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/gorse-io/xvec/internal/db/index/common"
	segmentstore "github.com/gorse-io/xvec/internal/db/index/segment"
)

// DocumentIterator lazily reads documents from an isolated IDMap snapshot.
type DocumentIterator struct {
	mu          sync.Mutex
	collection  *CollectionStore
	primaryMap  *common.PrimaryKeyMap
	primaryKeys *common.PrimaryKeySnapshotIterator
	tempRoot    string
	closed      bool
}

// CreateIterator captures the collection's current primary-key mapping without
// copying document payloads.
func (c *CollectionStore) CreateIterator(ctx context.Context) (*DocumentIterator, error) {
	if c == nil {
		return nil, errors.New("db: nil collection")
	}
	if ctx == nil {
		return nil, errors.New("db: nil iterator context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, ErrCollectionClosed
	}
	tempRoot, err := os.MkdirTemp(c.dir, ".iterator-idmap-*")
	if err != nil {
		return nil, fmt.Errorf("db: create iterator snapshot directory: %w", err)
	}
	fail := func(failure error, primaryMap *common.PrimaryKeyMap) (*DocumentIterator, error) {
		var closeErr error
		if primaryMap != nil {
			closeErr = primaryMap.Close()
		}
		return nil, errors.Join(failure, closeErr, os.RemoveAll(tempRoot))
	}
	primaryMap, err := c.manager.PrimaryKeys().CreateDetachedSnapshot(
		ctx, filepath.Join(tempRoot, "checkpoint"), filepath.Join(tempRoot, "working"),
	)
	if err != nil {
		return fail(err, nil)
	}
	primaryKeys, err := primaryMap.NewSnapshotIterator()
	if err != nil {
		return fail(err, primaryMap)
	}
	return &DocumentIterator{
		collection: c, primaryMap: primaryMap, primaryKeys: primaryKeys, tempRoot: tempRoot,
	}, nil
}

// Next returns the next retained document from the snapshot. found is false at
// exhaustion.
func (i *DocumentIterator) Next() (document segmentstore.StoredDocument, found bool, err error) {
	if i == nil {
		return segmentstore.StoredDocument{}, false, nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return segmentstore.StoredDocument{}, false, nil
	}
	primaryKey, docID, found, err := i.primaryKeys.Next()
	if err != nil || !found {
		return segmentstore.StoredDocument{}, found, err
	}

	i.collection.mu.RLock()
	defer i.collection.mu.RUnlock()
	if i.collection.closed {
		return segmentstore.StoredDocument{}, false, ErrCollectionClosed
	}
	document, found = i.collection.manager.RetainedDocument(docID)
	if !found {
		return segmentstore.StoredDocument{}, false, fmt.Errorf("%w: snapshot document %d is missing", ErrCollectionCorrupt, docID)
	}
	if document.PrimaryKey != primaryKey {
		return segmentstore.StoredDocument{}, false, fmt.Errorf(
			"%w: snapshot document %d primary key is %q, expected %q",
			ErrCollectionCorrupt, docID, document.PrimaryKey, primaryKey,
		)
	}
	return document, true, nil
}

// Close releases the IDMap snapshot. It is idempotent.
func (i *DocumentIterator) Close() error {
	if i == nil {
		return nil
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return nil
	}
	i.closed = true
	i.collection = nil
	iteratorErr := i.primaryKeys.Close()
	i.primaryKeys = nil
	mapErr := i.primaryMap.Close()
	i.primaryMap = nil
	removeErr := os.RemoveAll(i.tempRoot)
	i.tempRoot = ""
	return errors.Join(iteratorErr, mapErr, removeErr)
}
