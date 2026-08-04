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
	"fmt"
	"reflect"

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/db"
	dbsql "github.com/gorse-io/zvec/internal/db/sql"
)

// CreateIndex validates and backfills a currently implemented index, then
// atomically publishes the new schema in a manifest generation. At this stage
// Vector and INVERT indexes are snapshot-local runtime indexes, so backfill
// validates the complete live snapshot and publication persists their
// parameters.
func (c *Collection) CreateIndex(ctx context.Context, column string, index IndexParams, options CreateIndexOptions) error {
	const op = "create index"
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
	if column == "" {
		return invalidArgument(op, "column is empty")
	}
	if indexParamsNil(index) {
		return invalidArgument(op, "index parameters are nil")
	}
	if options.Concurrency < 0 {
		return invalidArgument(op, "Concurrency cannot be negative")
	}
	fieldIndex := -1
	for position := range c.schema.Fields {
		if c.schema.Fields[position].Name == column {
			fieldIndex = position
			break
		}
	}
	if fieldIndex < 0 {
		return &Error{Code: ErrorCodeNotFound, Op: op, Path: c.path, Message: fmt.Sprintf("field %q does not exist", column)}
	}
	if err := validateIndexParams(index); err != nil {
		return err
	}

	normalized := index.cloneIndexParams()
	nextSchema := c.schema.Clone()
	oldField := c.schema.Fields[fieldIndex]
	nextSchema.Fields[fieldIndex].Index = normalized
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	if err := supportedCreateIndex(nextSchema.Fields[fieldIndex], normalized, c.path); err != nil {
		return err
	}
	if equalIndexParams(oldField.Index, normalized) {
		return nil
	}
	if !oldField.DataType.IsVector() && !indexParamsNil(oldField.Index) && oldField.Index.IndexType() != normalized.IndexType() {
		return notSupported(op, c.path, fmt.Sprintf(
			"field %q already has %s and cannot also use %s", column, oldField.Index.IndexType(), normalized.IndexType(),
		))
	}
	if err := c.validateIndexBackfillLocked(ctx, nextSchema.Fields[fieldIndex], options.Concurrency); err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	return c.publishSchemaLocked(ctx, op, nextSchema)
}

// DropIndex atomically clears a scalar index or restores a vector field to the
// baseline unquantized Flat/IP definition. Existing documents are unchanged.
func (c *Collection) DropIndex(ctx context.Context, column string) error {
	const op = "drop index"
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
	if column == "" {
		return invalidArgument(op, "column is empty")
	}
	fieldIndex := -1
	for position := range c.schema.Fields {
		if c.schema.Fields[position].Name == column {
			fieldIndex = position
			break
		}
	}
	if fieldIndex < 0 {
		return &Error{Code: ErrorCodeNotFound, Op: op, Path: c.path, Message: fmt.Sprintf("field %q does not exist", column)}
	}
	oldField := c.schema.Fields[fieldIndex]
	if !oldField.DataType.IsVector() && indexParamsNil(oldField.Index) {
		return nil
	}
	defaultFlat := NewFlatIndexParams(MetricTypeIP)
	if oldField.DataType.IsVector() && equalIndexParams(oldField.EffectiveIndex(), defaultFlat) {
		return nil
	}
	nextSchema := c.schema.Clone()
	if oldField.DataType.IsVector() {
		nextSchema.Fields[fieldIndex].Index = defaultFlat
	} else {
		nextSchema.Fields[fieldIndex].Index = nil
	}
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	if oldField.DataType.IsVector() {
		if err := c.validateIndexBackfillLocked(ctx, nextSchema.Fields[fieldIndex], 0); err != nil {
			return wrapCollectionError(op, c.path, err)
		}
	}
	return c.publishSchemaLocked(ctx, op, nextSchema)
}

func (c *Collection) publishSchemaLocked(ctx context.Context, op string, nextSchema CollectionSchema) error {
	encoded, err := marshalCollectionSchema(nextSchema)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	committed, publishErr := c.store.PublishSchema(ctx, encoded)
	if committed {
		c.schema = nextSchema
	}
	if publishErr != nil {
		return wrapCollectionError(op, c.path, publishErr)
	}
	if !committed {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "schema publication did not commit"}
	}
	return nil
}

func (c *Collection) rewriteCollectionDocumentsLocked(
	ctx context.Context,
	op string,
	nextSchema CollectionSchema,
	documents []Document,
	workers int,
	transform func(*Document) error,
) error {
	if transform == nil {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "document rewrite transform is nil"}
	}
	rewritten := make([]db.StoredDocument, len(documents))
	if err := ailego.ParallelFor(ctx, len(documents), workers, func(_ context.Context, index int) error {
		document := documents[index]
		if transformErr := transform(&document); transformErr != nil {
			return fmt.Errorf("document %d: %w", document.DocID, transformErr)
		}
		if validateErr := document.Validate(nextSchema); validateErr != nil {
			return fmt.Errorf("document %d: %w", document.DocID, validateErr)
		}
		payload, encodeErr := marshalDocumentPayload(document.Fields)
		if encodeErr != nil {
			return fmt.Errorf("document %d: %w", document.DocID, encodeErr)
		}
		rewritten[index] = db.StoredDocument{
			DocID: document.DocID, PrimaryKey: document.PrimaryKey, Payload: payload,
		}
		return nil
	}); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return wrapCollectionError(op, c.path, err)
		}
		return invalidArgument(op, "document rewrite failed: %v", err)
	}
	encodedSchema, err := marshalCollectionSchema(nextSchema)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	committed, rewriteErr := c.store.RewriteDocuments(ctx, encodedSchema, rewritten)
	if committed {
		c.schema = nextSchema
	}
	if rewriteErr != nil {
		return wrapCollectionError(op, c.path, rewriteErr)
	}
	if !committed {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "document rewrite did not commit"}
	}
	return nil
}

func supportedCreateIndex(nextField FieldSchema, index IndexParams, path string) error {
	const op = "create index"
	switch index.IndexType() {
	case IndexTypeInvert:
		if nextField.DataType.IsVector() {
			return invalidArgument(op, "vector field %q cannot use INVERT", nextField.Name)
		}
		kind, _, supported := filterValueKind(nextField.DataType)
		if !supported || kind == dbsql.ValueBinary {
			return notSupported(op, path, fmt.Sprintf("INVERT is not implemented for %s field %q", nextField.DataType, nextField.Name))
		}
		return nil
	case IndexTypeFlat, IndexTypeHNSW, IndexTypeHNSWRaBitQ, IndexTypeIVF:
		if !nextField.DataType.IsVector() {
			return invalidArgument(op, "scalar field %q cannot use %s", nextField.Name, index.IndexType())
		}
		_, err := resolveCollectionVectorIndex(nextField, op, path)
		return err
	default:
		return notSupported(op, path, fmt.Sprintf("index %s on field %q is not implemented", index.IndexType(), nextField.Name))
	}
}

func equalIndexParams(left, right IndexParams) bool {
	if indexParamsNil(left) || indexParamsNil(right) {
		return indexParamsNil(left) && indexParamsNil(right)
	}
	return reflect.DeepEqual(left.cloneIndexParams(), right.cloneIndexParams())
}

func (c *Collection) validateIndexBackfillLocked(ctx context.Context, field FieldSchema, workers int) error {
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return err
	}
	switch field.Index.IndexType() {
	case IndexTypeInvert:
		kind, array, supported := filterValueKind(field.DataType)
		if !supported {
			return fmt.Errorf("unsupported INVERT data type %s", field.DataType)
		}
		params := field.Index.cloneIndexParams().(InvertIndexParams)
		index, err := dbsql.NewInvertedIndex(dbsql.Field{
			Name: field.Name, Kind: kind, Array: array, Nullable: field.Nullable,
			Filterable: true, Indexed: true,
			RangeOptimized: params.EnableRangeOptimization, ExtendedWildcard: params.EnableExtendedWildcard,
		})
		if err != nil {
			return err
		}
		if err := ailego.ParallelFor(ctx, len(documents), workers, func(_ context.Context, position int) error {
			document := &documents[position]
			raw, found := document.Fields[field.Name]
			value, err := toFilterValue(index.Field(), raw, found)
			if err != nil {
				return fmt.Errorf("document %d field %q: %w", document.DocID, field.Name, err)
			}
			return index.Add(uint64(position), value)
		}); err != nil {
			return err
		}
		return index.Seal()
	case IndexTypeFlat, IndexTypeHNSW, IndexTypeHNSWRaBitQ, IndexTypeIVF:
		spec, err := resolveCollectionVectorIndex(field, "create index", c.path)
		if err != nil {
			return err
		}
		if field.DataType.IsDenseVector() {
			switch spec.indexType {
			case IndexTypeFlat:
				_, err = buildCollectionDenseFlat(ctx, c.schema.Name, field, documents, spec)
			case IndexTypeHNSW:
				_, err = buildCollectionDenseHNSW(ctx, c.schema.Name, field, documents, spec)
			case IndexTypeHNSWRaBitQ:
				_, err = buildCollectionDenseHNSWRaBitQ(ctx, field, documents, spec, workers)
			case IndexTypeIVF:
				_, err = buildCollectionDenseIVF(ctx, c.schema.Name, field, documents, spec, workers)
			}
			return err
		}
		_, err = buildCollectionSparseIndex(ctx, field, documents, spec, false)
		return err
	default:
		return fmt.Errorf("unsupported index type %s", field.Index.IndexType())
	}
}
