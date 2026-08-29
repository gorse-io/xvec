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
	"io"
	"slices"
	"sync"

	"github.com/gorse-io/xvec/internal/db"
)

// IteratorOptions controls document projection during collection iteration.
type IteratorOptions struct {
	Projection Projection
}

// NewIteratorOptions returns iterator defaults aligned with zvec: all scalar
// fields and all vector fields are included.
func NewIteratorOptions() IteratorOptions {
	return IteratorOptions{Projection: Projection{IncludeVectors: true}}
}

// DocumentIterator lazily traverses an isolated snapshot of live documents.
type DocumentIterator struct {
	mu         sync.Mutex
	snapshot   *db.DocumentIterator
	schema     CollectionSchema
	projection Projection
	path       string
	closed     bool
	collection *Collection
}

// CreateIterator captures the current live-document snapshot. Writes committed
// after this method returns are not visible to the iterator.
func (c *Collection) CreateIterator(ctx context.Context, options IteratorOptions) (*DocumentIterator, error) {
	const op = "create iterator"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, wrapCollectionError(op, c.Path(), err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	if err := options.Projection.Validate(c.schema); err != nil {
		return nil, err
	}
	snapshot, err := c.store.CreateIterator(ctx)
	if err != nil {
		return nil, wrapCollectionError(op, c.path, err)
	}
	c.activeIterators++
	projection := options.Projection
	projection.OutputFields = slices.Clone(options.Projection.OutputFields)
	return &DocumentIterator{
		snapshot: snapshot, schema: c.schema.Clone(), projection: projection,
		path: c.path, collection: c,
	}, nil
}

// Next returns the next document or io.EOF when iteration is complete or the
// iterator has been closed.
func (i *DocumentIterator) Next() (*Document, error) {
	if i == nil {
		return nil, io.EOF
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return nil, io.EOF
	}
	stored, found, err := i.snapshot.Next()
	if err != nil {
		return nil, iteratorInternalError(i.path, err)
	}
	if !found {
		return nil, io.EOF
	}
	document, err := decodeStoredDocument(stored)
	if err != nil {
		return nil, iteratorInternalError(i.path, err)
	}
	if err := document.Validate(i.schema); err != nil {
		return nil, iteratorInternalError(i.path, fmt.Errorf("stored document %d violates schema: %w", stored.DocID, err))
	}
	projected, err := ProjectDocument(document, i.schema, i.projection)
	if err != nil {
		return nil, iteratorInternalError(i.path, err)
	}
	return &projected, nil
}

func iteratorInternalError(path string, err error) error {
	return &Error{Code: ErrorCodeInternal, Op: "iterate documents", Path: path, Message: "iterator snapshot is invalid", Err: err}
}

// Close releases the iterator snapshot. It is idempotent.
func (i *DocumentIterator) Close() {
	if i == nil {
		return
	}
	i.mu.Lock()
	if i.closed {
		i.mu.Unlock()
		return
	}
	i.closed = true
	if i.snapshot != nil {
		_ = i.snapshot.Close()
		i.snapshot = nil
	}
	collection := i.collection
	i.collection = nil
	i.mu.Unlock()

	if collection != nil {
		collection.mu.Lock()
		collection.activeIterators--
		collection.mu.Unlock()
	}
}
