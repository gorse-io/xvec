// SPDX-License-Identifier: Apache-2.0

package xvec

import (
	"context"

	"github.com/gorse-io/xvec/internal/db"
	"github.com/gorse-io/xvec/internal/db/index/common"
)

// The caller holds mu exclusively. A runtime with only its collection owner
// can extend its documents and ordinal map in place, like zvec's writing
// segment. Leased snapshots and scalar/FTS indexes use immutable replacements.
func (c *Collection) appendOwnedMutableRuntimeLocked(ctx context.Context) (bool, error) {
	for _, field := range c.schema.Fields {
		if !field.DataType.IsVector() && field.IndexType() != IndexTypeUndefined {
			return false, nil
		}
		if field.DataType.IsVector() && field.IndexType() == IndexTypeFlat {
			spec, err := resolveCollectionVectorIndex(field, "append writer runtime", c.path)
			if err != nil {
				return false, err
			}
			if spec.quantize != QuantizeTypeUndefined {
				return false, nil
			}
		}
	}
	schemaKey, err := collectionRuntimeKeyFor(c.schema, nil)
	if err != nil {
		return false, err
	}
	c.indexMu.RLock()
	cached := make(map[uint64]*collectionSegmentRuntime, len(c.segmentIndexes))
	for id, runtime := range c.segmentIndexes {
		cached[id] = runtime
	}
	c.indexMu.RUnlock()
	appended := false
	err = c.store.VisitSelectedSegmentSnapshots(ctx, func(_ common.SegmentMetadata, mutable bool) bool {
		return mutable
	}, func(snapshot db.SegmentSnapshot) error {
		runtime := cached[snapshot.Metadata.ID]
		if runtime == nil || runtime.refs.Load() != 1 || runtime.key.schemaHash != schemaKey.schemaHash ||
			len(runtime.indexes.scalar) != 0 || len(runtime.indexes.fts) != 0 {
			return nil
		}
		start := len(runtime.documents)
		if start == 0 || start != runtime.key.count || start >= len(snapshot.Documents) ||
			snapshot.Documents[start-1].DocID != runtime.key.maxDocID {
			return nil
		}
		// Validate the complete suffix before mutating any shared Flat index.
		suffix := make([]Document, len(snapshot.Documents)-start)
		for index, stored := range snapshot.Documents[start:] {
			if err := ctx.Err(); err != nil {
				return err
			}
			document, err := decodeStoredDocumentWithBorrowedVectors(stored, nil)
			if err != nil {
				return err
			}
			if err := document.Validate(c.schema); err != nil {
				return err
			}
			suffix[index] = document
		}
		documents := append(runtime.documents, suffix...)
		for _, field := range c.schema.Fields {
			if !field.DataType.IsVector() {
				continue
			}
			spec, err := resolveCollectionVectorIndex(field, "append writer runtime", c.path)
			if err != nil {
				return err
			}
			if err := runtime.indexes.appendWriterFlat(ctx, field, spec, documents, runtime.indexes); err != nil {
				return err
			}
		}
		for index, document := range suffix {
			runtime.documentOrdinals[document.DocID] = start + index
		}
		runtime.documents = documents
		runtime.key.count = len(documents)
		runtime.key.maxDocID = snapshot.Metadata.MaxDocID
		runtime.indexes.key = runtime.key
		c.indexBuildCount++
		appended = true
		return nil
	})
	return appended, err
}
