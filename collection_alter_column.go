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
)

// AlterColumn atomically renames or replaces one basic numeric field. A
// replacement may change its name, numeric type, nullability, and INVERT index
// parameters. Rename and replacement forms are mutually exclusive.
func (c *Collection) AlterColumn(ctx context.Context, column, rename string, field *FieldSchema, options AlterColumnOptions) error {
	const op = "alter column"
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
	oldField := c.schema.Fields[position]
	if !addColumnDataTypeSupported(oldField.DataType) {
		return invalidArgument(op, "only basic numeric columns can be altered, got %s", oldField.DataType)
	}
	if rename != "" && field != nil {
		return invalidArgument(op, "rename and replacement field cannot both be specified")
	}

	var nextField FieldSchema
	if rename != "" {
		if _, exists := c.schema.Field(rename); exists {
			return invalidArgument(op, "field %q already exists", rename)
		}
		nextField = oldField.Clone()
		nextField.Name = rename
	} else {
		if field == nil {
			return invalidArgument(op, "replacement field is nil")
		}
		nextField = field.Clone()
		if nextField.Name != column {
			if _, exists := c.schema.Field(nextField.Name); exists {
				return invalidArgument(op, "field %q already exists", nextField.Name)
			}
		}
		if oldField.Nullable && !nextField.Nullable {
			return invalidArgument(op, "nullable field %q cannot be changed to non-nullable", column)
		}
	}
	if err := nextField.Validate(); err != nil {
		return err
	}
	if !addColumnDataTypeSupported(nextField.DataType) {
		return invalidArgument(op, "only basic numeric data types are supported, got %s", nextField.DataType)
	}
	if equalFieldSchema(oldField, nextField) {
		return nil
	}

	nextSchema := c.schema.Clone()
	nextSchema.Fields[position] = nextField.Clone()
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
	return c.rewriteCollectionDocumentsLocked(ctx, op, nextSchema, documents, options.Concurrency, func(document *Document) error {
		value, found := document.Fields[column]
		delete(document.Fields, column)
		if !found {
			return nil
		}
		if value == nil {
			document.Fields[nextField.Name] = nil
			return nil
		}
		number, convertErr := columnNumberFromValue(value)
		if convertErr != nil {
			return convertErr
		}
		converted, convertErr := number.cast(nextField.DataType)
		if convertErr != nil {
			return convertErr
		}
		document.Fields[nextField.Name] = converted
		return nil
	})
}

func equalFieldSchema(left, right FieldSchema) bool {
	return left.Name == right.Name && left.DataType == right.DataType && left.Nullable == right.Nullable &&
		left.Dimension == right.Dimension && equalIndexParams(left.Index, right.Index)
}
