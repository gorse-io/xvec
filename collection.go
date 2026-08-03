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

package zvec

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"

	"github.com/gorse-io/zvec/internal/db"
)

// DefaultMaxBufferSize is the baseline-compatible future disk-cache budget.
const DefaultMaxBufferSize uint32 = 64 << 20

// CollectionOptions controls one collection handle. EnableMmap is persisted
// when creating a collection; MaxBufferSize is reserved for later disk-index
// caches. A zero MaxBufferSize selects DefaultMaxBufferSize.
type CollectionOptions struct {
	ReadOnly      bool
	EnableMmap    bool
	MaxBufferSize uint32
	WALSyncEvery  uint64
}

// NewCollectionOptions returns baseline-compatible handle defaults.
func NewCollectionOptions() CollectionOptions {
	return CollectionOptions{EnableMmap: true, MaxBufferSize: DefaultMaxBufferSize}
}

func (o CollectionOptions) normalized() CollectionOptions {
	if o.MaxBufferSize == 0 {
		o.MaxBufferSize = DefaultMaxBufferSize
	}
	return o
}

// CollectionStats is a point-in-time in-memory collection summary.
type CollectionStats struct {
	DocumentCount     uint64
	IndexCompleteness map[string]float32
}

// Collection is one open native Go collection. Its methods are safe for
// concurrent use; mutations are serialized so queries see complete versions.
type Collection struct {
	mu      sync.RWMutex
	store   *db.CollectionStore
	path    string
	schema  CollectionSchema
	options CollectionOptions
	closed  bool
}

// CreateAndOpen creates a native Go collection and opens its sole writable
// handle. The format is intentionally incompatible with C++ collections.
func CreateAndOpen(ctx context.Context, path string, schema CollectionSchema, options CollectionOptions) (*Collection, error) {
	if ctx == nil {
		return nil, invalidArgument("create collection", "context is nil")
	}
	if path == "" {
		return nil, invalidArgument("create collection", "path is empty")
	}
	if options.ReadOnly {
		return nil, &Error{Code: ErrorCodeInvalidArgument, Op: "create collection", Path: path, Message: "a collection cannot be created read-only"}
	}
	if err := schema.Validate(); err != nil {
		return nil, err
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, wrapCollectionError("create collection", path, err)
	}
	encodedSchema, err := marshalCollectionSchema(schema)
	if err != nil {
		return nil, wrapCollectionError("create collection", absolute, err)
	}
	options = options.normalized()
	store, err := db.CreateCollection(ctx, absolute, encodedSchema, db.CollectionOptions{
		EnableMmap:          options.EnableMmap,
		SegmentMaxDocuments: schema.MaxDocsPerSegment,
		WAL:                 db.WALOptions{SyncEvery: options.WALSyncEvery},
	})
	if err != nil {
		return nil, wrapCollectionError("create collection", absolute, err)
	}
	return &Collection{
		store: store, path: absolute, schema: schema.Clone(), options: options,
	}, nil
}

// Open opens the version named by CURRENT and replays the valid WAL prefix.
func Open(ctx context.Context, path string, options CollectionOptions) (*Collection, error) {
	if ctx == nil {
		return nil, invalidArgument("open collection", "context is nil")
	}
	if path == "" {
		return nil, invalidArgument("open collection", "path is empty")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, wrapCollectionError("open collection", path, err)
	}
	options = options.normalized()
	store, err := db.OpenCollection(ctx, absolute, db.CollectionOptions{
		ReadOnly: options.ReadOnly,
		WAL:      db.WALOptions{SyncEvery: options.WALSyncEvery},
	})
	if err != nil {
		return nil, wrapCollectionError("open collection", absolute, err)
	}
	manifest := store.Manifest()
	schema, err := unmarshalCollectionSchema(manifest.Schema)
	if err != nil {
		_ = store.Close()
		return nil, &Error{Code: ErrorCodeInternal, Op: "open collection", Path: absolute, Message: "collection schema is corrupt", Err: err}
	}
	if schema.MaxDocsPerSegment != manifest.SegmentMaxDocuments {
		_ = store.Close()
		return nil, &Error{Code: ErrorCodeInternal, Op: "open collection", Path: absolute, Message: "schema and manifest segment capacities differ"}
	}
	options.EnableMmap = manifest.EnableMmap
	return &Collection{
		store: store, path: absolute, schema: schema, options: options,
	}, nil
}

// Path returns the absolute collection directory.
func (c *Collection) Path() string {
	if c == nil {
		return ""
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.path
}

// Schema returns an independent schema copy.
func (c *Collection) Schema() CollectionSchema {
	if c == nil {
		return CollectionSchema{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.schema.Clone()
}

// Options returns the effective options for this handle.
func (c *Collection) Options() CollectionOptions {
	if c == nil {
		return CollectionOptions{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.options
}

// Stats returns live document count and current index completeness.
func (c *Collection) Stats() CollectionStats {
	if c == nil {
		return CollectionStats{IndexCompleteness: map[string]float32{}}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	stats := CollectionStats{IndexCompleteness: make(map[string]float32)}
	if c.store != nil {
		stats.DocumentCount = c.store.DocumentCount()
	}
	for _, field := range c.schema.Fields {
		index := field.EffectiveIndex()
		if indexParamsNil(index) {
			continue
		}
		completeness := float32(0)
		if flatFieldSupported(field) {
			completeness = 1
		}
		stats.IndexCompleteness[field.Name] = completeness
	}
	return stats
}

// Flush atomically publishes the current write segment and rotates its WAL.
func (c *Collection) Flush(ctx context.Context) error {
	if c == nil {
		return invalidArgument("flush collection", "collection is nil")
	}
	if ctx == nil {
		return invalidArgument("flush collection", "context is nil")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked("flush collection"); err != nil {
		return err
	}
	return wrapCollectionError("flush collection", c.path, c.store.Flush(ctx))
}

// Close releases files and the cross-process collection lock. It is
// idempotent; WAL-backed writes remain recoverable without an explicit Flush.
func (c *Collection) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return wrapCollectionError("close collection", c.path, c.store.Close())
}

// Destroy closes the handle and recursively removes only its validated
// collection directory. Read-only handles cannot destroy collections.
func (c *Collection) Destroy(ctx context.Context) error {
	if c == nil {
		return invalidArgument("destroy collection", "collection is nil")
	}
	if ctx == nil {
		return invalidArgument("destroy collection", "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError("destroy collection", c.Path(), err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.store == nil {
		return &Error{Code: ErrorCodeFailedPrecondition, Op: "destroy collection", Path: c.path, Message: "collection is closed"}
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: "destroy collection", Path: c.path, Message: "read-only collection cannot be destroyed"}
	}
	if !safeDestroyPath(c.path) {
		return &Error{Code: ErrorCodeInvalidArgument, Op: "destroy collection", Path: c.path, Message: "refusing to remove an unsafe collection path"}
	}
	c.closed = true
	closeErr := c.store.Close()
	removeErr := os.RemoveAll(c.path)
	return wrapCollectionError("destroy collection", c.path, errors.Join(closeErr, removeErr))
}

func (c *Collection) requireOpenLocked(op string) error {
	if c.closed || c.store == nil {
		return &Error{Code: ErrorCodeFailedPrecondition, Op: op, Path: c.path, Message: "collection is closed"}
	}
	return nil
}

func safeDestroyPath(path string) bool {
	if path == "" || !filepath.IsAbs(path) {
		return false
	}
	clean := filepath.Clean(path)
	volume := filepath.VolumeName(clean)
	root := volume + string(os.PathSeparator)
	return clean != root && clean != volume && filepath.Dir(clean) != clean
}
