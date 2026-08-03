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

import "context"

// CreateIndexOptions controls index-build concurrency. Zero lets the library
// select an appropriate worker count.
type CreateIndexOptions struct{ Concurrency int }

// AddColumnOptions controls column backfill concurrency.
type AddColumnOptions struct{ Concurrency int }

// AlterColumnOptions controls column migration concurrency.
type AlterColumnOptions struct{ Concurrency int }

// OptimizeOptions controls segment-optimization concurrency.
type OptimizeOptions struct{ Concurrency int }

// DropColumn will atomically remove a field in v0.2.
func (c *Collection) DropColumn(ctx context.Context, column string) error {
	return c.unsupportedMutation(ctx, "drop column", "DropColumn requires the v0.2 DDL executor")
}

// Optimize will merge segments and reclaim deleted versions in v0.2.
func (c *Collection) Optimize(ctx context.Context, options OptimizeOptions) error {
	return c.unsupportedMutation(ctx, "optimize collection", "Optimize requires the v0.2 segment merger")
}

func (c *Collection) unsupportedMutation(ctx context.Context, op, message string) error {
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if err := ctx.Err(); err != nil {
		return wrapCollectionError(op, c.Path(), err)
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireOpenLocked(op); err != nil {
		return err
	}
	if c.options.ReadOnly {
		return &Error{Code: ErrorCodePermissionDenied, Op: op, Path: c.path, Message: "collection is read-only"}
	}
	return notSupported(op, c.path, message)
}
