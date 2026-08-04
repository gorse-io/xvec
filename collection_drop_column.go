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
	"slices"
)

// DropColumn atomically removes one basic numeric field from the schema and
// every live document payload. Nonnumeric fields are reserved for their owning
// index milestones and are rejected until those implementations can migrate
// their auxiliary state safely.
func (c *Collection) DropColumn(ctx context.Context, column string) error {
	const op = "drop column"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	releaseRuntime, err := c.beginRuntimeTask(ctx, runtimeOptimizeTask, op, 8)
	if err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	defer releaseRuntime()
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
	position := -1
	for index := range c.schema.Fields {
		if c.schema.Fields[index].Name == column {
			position = index
			break
		}
	}
	if position < 0 {
		return &Error{Code: ErrorCodeNotFound, Op: op, Path: c.path, Message: fmt.Sprintf("field %q does not exist", column)}
	}
	field := c.schema.Fields[position]
	if !addColumnDataTypeSupported(field.DataType) {
		return invalidArgument(op, "only basic numeric columns can be dropped, got %s", field.DataType)
	}

	nextSchema := c.schema.Clone()
	nextSchema.Fields = slices.Delete(nextSchema.Fields, position, position+1)
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
	return c.rewriteCollectionDocumentsLocked(ctx, op, nextSchema, documents, 0, func(document *Document) error {
		delete(document.Fields, column)
		return nil
	})
}
