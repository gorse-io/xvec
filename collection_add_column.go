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

	"github.com/gorse-io/zvec/internal/ailego"
	"github.com/gorse-io/zvec/internal/db"
)

// AddColumn atomically adds a basic numeric field and backfills every live
// document with a baseline-compatible arithmetic expression. An empty
// expression is allowed only for nullable fields and writes explicit NULLs.
func (c *Collection) AddColumn(ctx context.Context, field FieldSchema, expression string, options AddColumnOptions) error {
	const op = "add column"
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
	if err := field.Validate(); err != nil {
		return err
	}
	if !addColumnDataTypeSupported(field.DataType) {
		return invalidArgument(op, "only basic numeric data types are supported, got %s", field.DataType)
	}
	if _, found := c.schema.Field(field.Name); found {
		return invalidArgument(op, "field %q already exists", field.Name)
	}
	if expression == "" && !field.Nullable {
		return invalidArgument(op, "non-nullable field %q requires a backfill expression", field.Name)
	}

	nextSchema := c.schema.Clone()
	nextSchema.Fields = append(nextSchema.Fields, field.Clone())
	if err := nextSchema.Validate(); err != nil {
		return err
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if len(documents) == 0 {
		return c.publishSchemaLocked(ctx, op, nextSchema)
	}

	var compiled *columnExpression
	if expression != "" {
		compiled, err = parseColumnExpression(expression, c.schema)
		if err != nil {
			return invalidArgument(op, "invalid expression %q: %v", expression, err)
		}
	}
	rewritten := make([]db.StoredDocument, len(documents))
	if err := ailego.ParallelFor(ctx, len(documents), options.Concurrency, func(_ context.Context, index int) error {
		document := documents[index]
		var value any
		if compiled != nil {
			evaluated, evaluateErr := compiled.evaluate(document.Fields, field.DataType)
			if evaluateErr != nil {
				return fmt.Errorf("document %d: %w", document.DocID, evaluateErr)
			}
			value = evaluated
		}
		if value == nil && !field.Nullable {
			return fmt.Errorf("document %d expression evaluates to NULL for non-nullable field %q", document.DocID, field.Name)
		}
		document.Fields[field.Name] = value
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
		return invalidArgument(op, "backfill expression %q failed: %v", expression, err)
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
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "column rewrite did not commit"}
	}
	return nil
}
