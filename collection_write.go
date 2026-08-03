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

	"github.com/gorse-io/zvec/internal/db"
)

// WriteResult reports one document mutation in input order.
type WriteResult struct {
	PrimaryKey string
	DocID      uint64
	Err        error
}

// Insert durably writes new primary keys. Valid documents in a mixed batch
// are committed even when other entries fail validation or already exist.
func (c *Collection) Insert(ctx context.Context, documents []Document) ([]WriteResult, error) {
	return c.writeDocuments(ctx, OperatorInsert, documents)
}

// Upsert inserts new documents and partially updates existing documents.
func (c *Collection) Upsert(ctx context.Context, documents []Document) ([]WriteResult, error) {
	return c.writeDocuments(ctx, OperatorUpsert, documents)
}

// Update partially replaces fields on existing documents while retaining all
// unspecified fields from the current version.
func (c *Collection) Update(ctx context.Context, documents []Document) ([]WriteResult, error) {
	return c.writeDocuments(ctx, OperatorUpdate, documents)
}

func (c *Collection) writeDocuments(ctx context.Context, operator Operator, documents []Document) ([]WriteResult, error) {
	op := operator.String()
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(documents) == 0 {
		return nil, invalidArgument(op, "document batch is empty")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	results := make([]WriteResult, len(documents))
	batchError := &BatchWriteError{}
	for index, document := range documents {
		results[index].PrimaryKey = document.PrimaryKey
		if err := ctx.Err(); err != nil {
			for remaining := index; remaining < len(documents); remaining++ {
				wrapped := wrapCollectionError(op, c.path, err)
				results[remaining] = WriteResult{PrimaryKey: documents[remaining].PrimaryKey, Err: wrapped}
				batchError.add(wrapped)
			}
			break
		}
		prepared, err := c.prepareWriteDocumentLocked(ctx, operator, document)
		if err != nil {
			wrapped := wrapCollectionError(op, c.path, err)
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		payload, err := marshalDocumentPayload(prepared.Fields)
		if err != nil {
			wrapped := wrapCollectionError(op, c.path, err)
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		if len(payload) > db.MaxDocumentPayloadSize {
			wrapped := &Error{
				Code: ErrorCodeResourceExhausted, Op: op, Path: c.path,
				Message: fmt.Sprintf("document payload is %d bytes, maximum %d", len(payload), db.MaxDocumentPayloadSize),
			}
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		dbResults, batchErr := c.callStoreWriteLocked(ctx, operator, db.WriteInput{
			PrimaryKey: prepared.PrimaryKey, Payload: payload,
		})
		var itemErr error
		if len(dbResults) == 1 {
			results[index].DocID = dbResults[0].DocID
			itemErr = dbResults[0].Err
		}
		if itemErr == nil {
			itemErr = batchErr
		}
		if itemErr != nil {
			wrapped := wrapCollectionError(op, c.path, itemErr)
			results[index].Err = wrapped
			batchError.add(wrapped)
		}
	}
	if batchError.Failed != 0 {
		return results, batchError
	}
	return results, nil
}

func (c *Collection) prepareWriteDocumentLocked(ctx context.Context, operator Operator, document Document) (Document, error) {
	clone, err := document.Clone()
	if err != nil {
		return Document{}, err
	}
	switch operator {
	case OperatorInsert:
		if err := clone.Validate(c.schema); err != nil {
			return Document{}, err
		}
		return clone, nil
	case OperatorUpsert, OperatorUpdate:
		if err := validateDocumentAgainstSchema(clone, c.schema, true); err != nil {
			return Document{}, err
		}
		current, found, err := c.fetchOneLocked(ctx, clone.PrimaryKey)
		if err != nil {
			return Document{}, err
		}
		if !found {
			if operator == OperatorUpdate {
				return Document{}, fmt.Errorf("%w: %q", db.ErrPrimaryKeyNotFound, clone.PrimaryKey)
			}
			if err := clone.Validate(c.schema); err != nil {
				return Document{}, err
			}
			return clone, nil
		}
		for name, value := range clone.Fields {
			current.Fields[name] = value
		}
		current.Score = 0
		current.DocID = 0
		if err := current.Validate(c.schema); err != nil {
			return Document{}, err
		}
		return current, nil
	default:
		return Document{}, invalidArgument("write documents", "unsupported operator %d", operator)
	}
}

func (c *Collection) callStoreWriteLocked(ctx context.Context, operator Operator, input db.WriteInput) ([]db.WriteResult, error) {
	switch operator {
	case OperatorInsert:
		return c.store.Insert(ctx, []db.WriteInput{input})
	case OperatorUpsert:
		return c.store.Upsert(ctx, []db.WriteInput{input})
	case OperatorUpdate:
		return c.store.Update(ctx, []db.WriteInput{input})
	default:
		return nil, errors.New("zvec: unsupported write operator")
	}
}

// Delete durably removes the current versions of the requested primary keys.
func (c *Collection) Delete(ctx context.Context, primaryKeys []string) ([]WriteResult, error) {
	const op = "DELETE"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(primaryKeys) == 0 {
		return nil, invalidArgument(op, "primary-key batch is empty")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	results := make([]WriteResult, len(primaryKeys))
	batchError := &BatchWriteError{}
	for index, primaryKey := range primaryKeys {
		results[index].PrimaryKey = primaryKey
		if err := ctx.Err(); err != nil {
			for remaining := index; remaining < len(primaryKeys); remaining++ {
				wrapped := wrapCollectionError(op, c.path, err)
				results[remaining] = WriteResult{PrimaryKey: primaryKeys[remaining], Err: wrapped}
				batchError.add(wrapped)
			}
			break
		}
		if _, err := (Document{PrimaryKey: primaryKey}).Clone(); err != nil {
			wrapped := wrapCollectionError(op, c.path, err)
			results[index].Err = wrapped
			batchError.add(wrapped)
			continue
		}
		dbResults, batchErr := c.store.Delete(ctx, []string{primaryKey})
		var itemErr error
		if len(dbResults) == 1 {
			results[index].DocID = dbResults[0].DocID
			itemErr = dbResults[0].Err
		}
		if itemErr == nil {
			itemErr = batchErr
		}
		if itemErr != nil {
			wrapped := wrapCollectionError(op, c.path, itemErr)
			results[index].Err = wrapped
			batchError.add(wrapped)
		}
	}
	if batchError.Failed != 0 {
		return results, batchError
	}
	return results, nil
}

// DeleteByFilter durably removes every live document for which filter
// evaluates to SQL TRUE. Selection and WAL-backed deletion are serialized
// under the collection write lock, so a matched version cannot be replaced
// between those two phases.
func (c *Collection) DeleteByFilter(ctx context.Context, filter string) error {
	const op = "delete by filter"
	if c == nil {
		return invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return invalidArgument(op, "context is nil")
	}
	if filter == "" {
		return invalidArgument(op, "filter is empty")
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
	plan, err := buildFilterPlan(filter, c.schema)
	if err != nil {
		return invalidArgument(op, "invalid filter: %v", err)
	}
	documents, err := c.liveDocumentsLocked(ctx)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	matched, err := evaluateFilterDocuments(ctx, plan, documents)
	if err != nil {
		return wrapFilterEvaluationError(op, c.path, err)
	}
	primaryKeys := make([]string, 0, len(documents))
	for _, document := range documents {
		if err := ctx.Err(); err != nil {
			return wrapCollectionError(op, c.path, err)
		}
		if matched(document.DocID) {
			primaryKeys = append(primaryKeys, document.PrimaryKey)
		}
	}
	if len(primaryKeys) == 0 {
		return nil
	}
	results, err := c.store.Delete(ctx, primaryKeys)
	if err != nil {
		return wrapCollectionError(op, c.path, err)
	}
	if len(results) != len(primaryKeys) {
		return &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "storage returned an incomplete delete result"}
	}
	for _, result := range results {
		if result.Err != nil {
			return wrapCollectionError(op, c.path, result.Err)
		}
	}
	return nil
}

// Fetch returns one independently owned document pointer per requested key.
// Missing or deleted keys have a nil entry and are not errors.
func (c *Collection) Fetch(ctx context.Context, primaryKeys []string, projection Projection) ([]*Document, error) {
	const op = "fetch"
	if c == nil {
		return nil, invalidArgument(op, "collection is nil")
	}
	if ctx == nil {
		return nil, invalidArgument(op, "context is nil")
	}
	if len(primaryKeys) == 0 {
		return nil, invalidArgument(op, "primary-key batch is empty")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if err := c.requireOpenLocked(op); err != nil {
		return nil, err
	}
	if err := projection.Validate(c.schema); err != nil {
		return nil, err
	}
	for _, primaryKey := range primaryKeys {
		if _, err := (Document{PrimaryKey: primaryKey}).Clone(); err != nil {
			return nil, err
		}
	}
	fetched, err := c.store.Fetch(ctx, primaryKeys)
	results := make([]*Document, len(fetched))
	for index, item := range fetched {
		if item.Err != nil || item.Document == nil {
			continue
		}
		document, decodeErr := decodeStoredDocument(*item.Document)
		if decodeErr != nil {
			return results, wrapCollectionError(op, c.path, decodeErr)
		}
		if validateErr := document.Validate(c.schema); validateErr != nil {
			return results, &Error{Code: ErrorCodeInternal, Op: op, Path: c.path, Message: "stored document violates collection schema", Err: validateErr}
		}
		projected, projectErr := ProjectDocument(document, c.schema, projection)
		if projectErr != nil {
			return results, projectErr
		}
		results[index] = &projected
	}
	if err != nil {
		return results, wrapCollectionError(op, c.path, err)
	}
	return results, nil
}

func (c *Collection) fetchOneLocked(ctx context.Context, primaryKey string) (Document, bool, error) {
	results, err := c.store.Fetch(ctx, []string{primaryKey})
	if err != nil {
		return Document{}, false, err
	}
	if len(results) != 1 || results[0].Document == nil {
		return Document{}, false, nil
	}
	document, err := decodeStoredDocument(*results[0].Document)
	return document, err == nil, err
}

func decodeStoredDocument(stored db.StoredDocument) (Document, error) {
	fields, err := unmarshalDocumentPayload(stored.Payload)
	if err != nil {
		return Document{}, fmt.Errorf("decode document %d: %w", stored.DocID, err)
	}
	return Document{
		PrimaryKey: stored.PrimaryKey, Fields: fields, DocID: stored.DocID,
	}, nil
}
