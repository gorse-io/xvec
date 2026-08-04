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
	"fmt"

	dbsql "github.com/gorse-io/zvec/internal/db/sql"
)

// Optimize atomically compacts the current live snapshot into maximally sized
// contiguous-ID segments, reclaims superseded/deleted versions, rebuilds the
// implemented vector/INVERT runtime state, and removes obsolete storage files.
func (c *Collection) Optimize(ctx context.Context, options OptimizeOptions) error {
	const op = "optimize collection"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	if options.Concurrency < 0 {
		return invalidArgument(op, "Concurrency cannot be negative")
	}
	needed, err := c.store.OptimizationNeeded(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if !needed {
		// Publication is the durability boundary. A process can stop after the
		// new manifest becomes current but before obsolete files are removed;
		// a later no-op optimization must finish that safe cleanup.
		return wrapCollectionError(op, c.path, c.store.PruneObsoleteArtifacts(ctx))
	}
	for _, field := range c.schema.Fields {
		if err := optimizableField(field, c.path); err != nil {
			return err
		}
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if err := c.rewriteCollectionDocumentsLocked(ctx, op, c.schema.Clone(), documents, options.Concurrency, func(*Document) error {
		return nil
	}); err != nil {
		return err
	}
	return wrapCollectionError(op, c.path, c.store.PruneObsoleteArtifacts(ctx))
}

func optimizableField(field FieldSchema, path string) error {
	index := field.EffectiveIndex()
	if indexParamsNil(index) {
		return nil
	}
	switch index.IndexType() {
	case IndexTypeFlat, IndexTypeHNSW, IndexTypeHNSWRaBitQ, IndexTypeIVF, IndexTypeDiskANN, IndexTypeVamana:
		if !field.DataType.IsVector() {
			return invalidArgument("optimize collection", "scalar field %q cannot use %s", field.Name, index.IndexType())
		}
		_, err := resolveCollectionVectorIndex(field, "optimize collection", path)
		return err
	case IndexTypeInvert:
		if field.DataType.IsVector() {
			return invalidArgument("optimize collection", "vector field %q cannot use INVERT", field.Name)
		}
		kind, _, supported := filterValueKind(field.DataType)
		if !supported || kind == dbsql.ValueBinary {
			return notSupported("optimize collection", path, fmt.Sprintf("INVERT is not implemented for %s field %q", field.DataType, field.Name))
		}
		return nil
	default:
		return notSupported("optimize collection", path, fmt.Sprintf("index %s on field %q is not implemented", index.IndexType(), field.Name))
	}
}
