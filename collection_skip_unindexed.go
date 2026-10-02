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
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/gorse-io/xvec/internal/db/index/common"
)

// acquireVectorQuerySnapshotLocked keeps restricted views separate from the full
// FTS/maintenance snapshot and never changes the prepared runtime cache.
func (c *Collection) acquireVectorQuerySnapshotLocked(ctx context.Context, fieldName string) (*collectionQuerySnapshot, func(), error) {
	if !c.options.SkipUnindexedSegments || fieldName == "" {
		return c.acquireQuerySnapshotLocked(ctx)
	}
	field, found := c.schema.Field(fieldName)
	if !found || !field.DataType.IsVector() {
		return nil, nil, invalidArgument("query", "vector field %q does not exist", fieldName)
	}
	spec, err := resolveCollectionVectorIndex(field, "query", c.path)
	if err != nil {
		return nil, nil, err
	}
	if spec.indexType == IndexTypeFlat {
		return c.acquireQuerySnapshotLocked(ctx)
	}
	c.querySnapshotMu.Lock()
	defer c.querySnapshotMu.Unlock()
	snapshot := c.vectorQuerySnapshots[fieldName]
	if snapshot == nil {
		schemaKey, err := collectionRuntimeKeyFor(c.schema, nil)
		if err != nil {
			return nil, nil, err
		}
		// Copy committed metadata before entering the store visitor: the selector
		// must not recursively acquire the store lock or touch index files.
		schemaHash := hex.EncodeToString(schemaKey.schemaHash[:])
		committed := make(map[uint64]common.SegmentIndexSnapshotMetadata)
		for _, indexSnapshot := range c.store.Manifest().SegmentIndexSnapshots {
			committed[indexSnapshot.SegmentID] = indexSnapshot
		}
		segments, err := c.segmentDocumentsSelectedLocked(ctx, func(metadata common.SegmentMetadata, mutable bool) bool {
			if mutable {
				return false
			}
			indexSnapshot := committed[metadata.ID]
			if indexSnapshot.SegmentID != metadata.ID || indexSnapshot.SchemaSHA256 != schemaHash ||
				indexSnapshot.DocumentCount != metadata.DocCount || indexSnapshot.MinDocumentID != metadata.MinDocID ||
				indexSnapshot.MaxDocumentID != metadata.MaxDocID {
				return false
			}
			for _, artifact := range indexSnapshot.Artifacts {
				if artifact.Field == fieldName && artifact.Kind == collectionVectorArtifactKind(spec.indexType) && artifact.File != "" {
					return true
				}
			}
			return false
		})
		if err != nil {
			return nil, nil, err
		}
		documents, err := c.liveDocumentsFromSelectedSegmentsLocked(ctx, segments, false)
		if err != nil {
			return nil, nil, err
		}
		runtimes, err := c.cachedSegmentRuntimeIndexesLocked(ctx, segments)
		if err != nil {
			return nil, nil, err
		}
		liveFilter, err := evaluateSegmentFilters(ctx, nil, documents, segments, runtimes, c.runtimeConfig().InvertToForwardScanRatio)
		if err != nil {
			return nil, nil, err
		}
		snapshot = &collectionQuerySnapshot{schema: c.schema.Clone(), documents: documents, documentOrdinals: indexDocumentOrdinals(documents), segments: segments, runtimes: runtimes, liveFilter: liveFilter}
		snapshot.retainRuntimes()
		if c.vectorQuerySnapshots == nil {
			c.vectorQuerySnapshots = make(map[string]*collectionQuerySnapshot)
		}
		c.vectorQuerySnapshots[fieldName] = snapshot
	}
	snapshot.retainRuntimes()
	c.queryLeases.Add(1)
	var once sync.Once
	return snapshot, func() { once.Do(func() { _ = c.releaseSnapshotRuntimes(snapshot); c.queryLeases.Done() }) }, nil
}

// Resolve the authoritative source version before releasing the collection lock.
// An excluded source is fetched by key, never scanned as a candidate segment.
func (c *Collection) resolveQueryVectorByKeyLocked(ctx context.Context, field FieldSchema, key, op string) (DenseVector, SparseVector, error) {
	fetched, err := c.store.Fetch(ctx, []string{key})
	if err != nil {
		return nil, nil, err
	}
	if fetched[0].Err != nil {
		return nil, nil, fetched[0].Err
	}
	if fetched[0].Document == nil {
		return resolveSnapshotQueryVector(nil, field, key, op)
	}
	document, err := decodeStoredDocument(*fetched[0].Document)
	if err != nil {
		return nil, nil, err
	}
	return resolveSnapshotQueryVector([]Document{document}, field, key, op)
}

func validateCollectionQueryVector(op string, field FieldSchema, dense DenseVector, sparse SparseVector) error {
	if field.DataType.IsDenseVector() {
		if !isNilInterface(sparse) {
			return invalidArgument(op, "dense field %q cannot use a sparse query vector", field.Name)
		}
		_, err := validateDenseQueryVector(field, dense)
		return err
	}
	if !isNilInterface(dense) {
		return invalidArgument(op, "sparse field %q cannot use a dense query vector", field.Name)
	}
	_, err := validateSparseQueryVector(field, sparse)
	return err
}

func validateCollectionGroupQueryConfig(op, path string, vectorIndex collectionVectorIndex, params collectionQueryConfig) error {
	if params.options.UseRefiner && vectorIndex.indexType == IndexTypeIVFRaBitQ {
		return notSupported(op, path, "IVF-RaBitQ group-by does not support refinement")
	}
	if !params.options.Linear && vectorIndex.indexType != IndexTypeFlat {
		if params.options.UseRefiner && (vectorIndex.indexType == IndexTypeHNSW || vectorIndex.indexType == IndexTypeHNSWRaBitQ) {
			return notSupported(op, path, fmt.Sprintf("%s group-by with a refiner requires Linear", vectorIndex.indexType))
		}
		if vectorIndex.indexType != IndexTypeHNSW && vectorIndex.indexType != IndexTypeHNSWRaBitQ && vectorIndex.indexType != IndexTypeIVFRaBitQ {
			return notSupported(op, path, fmt.Sprintf("group-by is not supported for %s graph traversal", vectorIndex.indexType))
		}
	}
	return nil
}
